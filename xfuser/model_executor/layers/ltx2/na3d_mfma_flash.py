# SPDX-License-Identifier: Apache-2.0
"""Triton flash-style 3D neighbourhood attention for LTX-2.5 VAE decoder.

Works on AMD (MFMA via ROCm) and NVIDIA (Tensor Cores via CUDA); Triton's JIT
handles both backends.  Confirmed on gfx950 (MI350X), gfx942 (MI300X), and B200.

Optimizations over the original na3d_eager_attn.py tiled SDPA implementation:
  1. BF16 tl.dot -> hardware matrix units (MFMA on AMD, TC on NVIDIA).
  2. BLOCK_KV = next_pow2(BLOCK_Q + KW - 1): single W-chunk per iteration.
  3. KT/KH/KW as tl.constexpr → static flat loop for Triton's software pipeliner.
  4. tl.exp -> tl.exp2 (folding log2e into the running max).
  5. Fused QKV GEMM in the processor: one (SEQ,C)×(C,3C) read instead of three.
  6. @triton.autotune over BLOCK_Q, num_stages, num_warps — keyed on (KT,KH,KW,W).
     Empirical optimum for LTX-2.5 decode shapes: BLOCK_Q=16, num_stages=2,
     num_warps=4 (confirmed by exhaustive search; see _MFMA_CONFIGS).
  7. int64 cast on pid_bnh to prevent signed-overflow when
     pid_bnh × stride_bnh exceeds 2^31 (full non-tiled 12.5M-token sequence).

AMDGCN_USE_BUFFER_OPS: leave unset (default) to support large tensors. Set to "1"
for tiled-only workloads (tensors < 2 GB) to use hardware-bounds-checked loads.
"""
from __future__ import annotations
import math
import os
import weakref
import triton
import triton.language as tl
import torch
import torch.nn.functional as F

_LOG2E: float = math.log2(math.e)   # ≈ 1.4426950408889634


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------
# For all LTX-2.5 kernel widths (KW=5,7,11):
#   BLOCK_Q=16 -> BLOCK_KV = next_pow2(16+KW-1) = 32  (KW≤17)
#   BLOCK_Q=32 -> BLOCK_KV = next_pow2(32+KW-1) = 64  (KW≤33)
# The W >= BLOCK_Q constraint is enforced by the pruner (det shapes with W=16
# cannot use BLOCK_Q=32).
_MFMA_CONFIGS = [
    # Autotune sweeps BLOCK_Q, num_stages, and num_warps.
    # For LTX-2.5 decode shapes the winner is BLOCK_Q=16, num_stages=2, num_warps=4.
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 2}, num_warps=4),
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 2}, num_warps=8),
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 4}, num_warps=4),
    triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 32, 'num_stages': 4}, num_warps=8),
    # BLOCK_Q=32: halves program count but doubles BLOCK_KV (more masked compute).
    triton.Config({'BLOCK_Q': 32, 'BLOCK_KV': 64, 'num_stages': 2}, num_warps=4),
    triton.Config({'BLOCK_Q': 32, 'BLOCK_KV': 64, 'num_stages': 2}, num_warps=8),
]


def _prune_mfma_configs(configs, named_args, **kwargs):
    """Drop BLOCK_Q > W configs (W >= BLOCK_Q required for same-(t,h) guarantee)."""
    W = named_args['W']
    return [c for c in configs if c.kwargs['BLOCK_Q'] <= W]


# ---------------------------------------------------------------------------
# Inner kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_MFMA_CONFIGS,
    key=['KT', 'KH', 'KW', 'W'],
    prune_configs_by={'early_config_prune': _prune_mfma_configs},
)
@triton.jit
def _na3d_mfma_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_bnh, stride_seq,
    T, H, W, SEQ,
    HD: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Flash-attention inner kernel. Assumes W >= BLOCK_Q (same (t,h) row per block)."""
    pid_q   = tl.program_id(0)
    pid_bnh = tl.program_id(1)
    HW      = H * W

    q_start = pid_q * BLOCK_Q
    q_offs  = tl.arange(0, BLOCK_Q)
    q_idx   = q_start + q_offs
    q_mask  = q_idx < SEQ

    q_t = q_idx // HW
    q_h = (q_idx % HW) // W
    q_w = q_idx % W

    q_t_ws = tl.minimum(tl.maximum(q_t - KT // 2, 0), T - KT)
    q_h_ws = tl.minimum(tl.maximum(q_h - KH // 2, 0), H - KH)
    q_w_ws = tl.minimum(tl.maximum(q_w - KW // 2, 0), W - KW)

    INF_I = 999999
    t_ws  = tl.min(tl.where(q_mask, q_t_ws, INF_I))
    h_ws  = tl.min(tl.where(q_mask, q_h_ws, INF_I))
    w_lo  = tl.min(tl.where(q_mask, q_w_ws, INF_I))

    hd_offs = tl.arange(0, HD)
    kv_offs = tl.arange(0, BLOCK_KV)
    # int64 cast: pid_bnh(int32) × stride_bnh can exceed 2^31 for large sequences.
    base    = pid_bnh.to(tl.int64) * stride_bnh
    kv_w    = w_lo + kv_offs
    kv_ok   = kv_w < W   # W-boundary guard, constant across the KT×KH loop

    # Q loaded once into registers for the entire 121-iteration loop.
    Q_tile = tl.load(
        Q_ptr + base + q_idx[:, None] * stride_seq + hd_offs[None, :],
        mask=q_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    m_i = tl.full((BLOCK_Q,), -3e38, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,),       dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HD),    dtype=tl.float32)

    # Flat KT×KH loop with static trip count — enables Triton's software pipeliner.
    # valid_w, kv_ok, and q_mask are loop-invariant; the Triton compiler's LICM
    # handles them without explicit pre-computation (which would raise VGPR pressure
    # from 71 to ~87/lane, dropping occupancy from 7 to 6 waves/SIMD).
    for kv_idx in range(KT * KH):
        dt   = kv_idx // KH
        dh   = kv_idx  % KH
        t_kv = t_ws + dt
        h_kv = h_ws + dh

        row_base = t_kv * HW + h_kv * W
        kv_flat  = row_base + kv_w

        K_T = tl.load(
            K_ptr + base + kv_flat[None, :] * stride_seq + hd_offs[:, None],
            mask=kv_ok[None, :], other=0.0,
        ).to(tl.bfloat16)

        V_tile = tl.load(
            V_ptr + base + kv_flat[:, None] * stride_seq + hd_offs[None, :],
            mask=kv_ok[:, None], other=0.0,
        ).to(tl.bfloat16)

        # QK in log2 space (fold log2e into scale to use exp2 throughout).
        scores = tl.dot(Q_tile, K_T, out_dtype=tl.float32) * 1.4426950408889634

        valid_w = (
            (kv_w[None, :] >= q_w_ws[:, None]) &
            (kv_w[None, :] <  q_w_ws[:, None] + KW)
        )
        scores = tl.where(valid_w & kv_ok[None, :] & q_mask[:, None],
                          scores, float('-inf'))

        m_new    = tl.maximum(m_i, tl.max(scores, axis=1))
        exp_s    = tl.exp2(scores  - m_new[:, None])
        exp_diff = tl.exp2(m_i     - m_new)
        l_i = l_i * exp_diff + tl.sum(exp_s, axis=1)
        acc = acc * exp_diff[:, None] + tl.dot(
            exp_s.to(tl.bfloat16), V_tile, out_dtype=tl.float32
        )
        m_i = m_new

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc    = acc / safe_l[:, None]

    tl.store(
        Out_ptr + base + q_idx[:, None] * stride_seq + hd_offs[None, :],
        acc.to(tl.bfloat16),
        mask=q_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------
def na3d_mfma_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """BF16-MFMA 3D neighbourhood flash attention with autotuned tile sizes.

    Args:
        q, k, v     : (B, T, H, W, NH, HD=64) bfloat16.
        kernel_size : (KT, KH, KW) neighbourhood window.
    Returns:
        Output (B, T, H, W, NH, HD) bfloat16.
    """
    B, T, H, W, NH, HD = q.shape
    assert HD == 64, "head_dim must be 64"
    KT, KH, KW = kernel_size
    SEQ = T * H * W

    # 2 GB buffer_load guard (only relevant when AMDGCN_USE_BUFFER_OPS=1).
    tensor_bytes = B * NH * SEQ * HD * 2
    if os.environ.get("AMDGCN_USE_BUFFER_OPS") == "1" and tensor_bytes >= 2**31:
        raise AssertionError(
            f"Tensor {tensor_bytes/1e9:.1f} GB exceeds 2 GB AMDGCN buffer_load limit. "
            "Unset AMDGCN_USE_BUFFER_OPS to use predicated flat loads with no size cap."
        )

    def _flat(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 4, 1, 2, 3, 5).reshape(B * NH, SEQ, HD).contiguous()

    q_f, k_f, v_f = _flat(q), _flat(k), _flat(v)
    out_f = torch.empty_like(q_f)

    # Grid uses a lambda so Triton passes the autotuned BLOCK_Q via meta.
    grid = lambda meta: (triton.cdiv(SEQ, meta['BLOCK_Q']), B * NH)

    _na3d_mfma_fwd[grid](
        q_f, k_f, v_f, out_f,
        SEQ * HD, HD,
        T, H, W, SEQ,
        HD=HD, KT=KT, KH=KH, KW=KW,
    )
    return out_f.reshape(B, NH, T, H, W, HD).permute(0, 2, 3, 4, 1, 5).contiguous()


# ---------------------------------------------------------------------------
# Attention processor
# ---------------------------------------------------------------------------
class LTX2VideoVaeMfmaAttnProcessor:
    """Triton flash-NA3D attention for the LTX-2.5 diffusion decoder.

    Drop-in replacement for LTX2VideoVaeEagerSdpaAttnProcessor. Works on AMD
    (MFMA) and NVIDIA (Tensor Cores) via Triton's JIT compilation.
    Improvements over the SDPA fallback:
      - Hardware matrix units via tl.dot (MFMA on ROCm, TC on CUDA).
      - Autotuned BLOCK_Q, num_stages, num_warps per kernel/shape combination.
      - Fused QKV GEMM: reads hidden_states once instead of 3×.
    """

    def __init__(self):
        self._fused_qkv: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _get_fused_qkv(self, attn) -> torch.nn.Linear:
        if attn not in self._fused_qkv:
            W_cat = torch.cat([attn.to_q.weight.data,
                               attn.to_k.weight.data,
                               attn.to_v.weight.data], dim=0)
            b_cat = torch.cat([attn.to_q.bias.data,
                               attn.to_k.bias.data,
                               attn.to_v.bias.data], dim=0)
            out_f, in_f = W_cat.shape
            fused = torch.nn.Linear(in_f, out_f, bias=True,
                                    device=W_cat.device, dtype=W_cat.dtype)
            fused.weight = torch.nn.Parameter(W_cat, requires_grad=False)
            fused.bias   = torch.nn.Parameter(b_cat, requires_grad=False)
            self._fused_qkv[attn] = fused
        return self._fused_qkv[attn]

    def _project_qkv_fused(self, attn, hidden_states: torch.Tensor):
        B, T, H, W, C = hidden_states.shape
        shape = (B, T, H, W, attn.heads, attn.head_dim)
        fused = self._get_fused_qkv(attn)
        qkv   = fused(hidden_states)
        q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
        query = attn.norm_q(q_raw.view(shape))
        key   = attn.norm_k(k_raw.view(shape))
        query = query * attn.scale
        return attn.rope(query), attn.rope(key), v_raw.view(shape)

    def __call__(self, attn, hidden_states, block_mask=None):
        B, T, H, W, _ = hidden_states.shape
        q, k, v = self._project_qkv_fused(attn, hidden_states)
        out = na3d_mfma_attn(q, k, v, kernel_size=tuple(attn.kernel_size))
        out = out.reshape(B, T, H, W, attn.heads * attn.head_dim)
        return attn.to_out[0](out)
