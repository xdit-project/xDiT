"""Hyperloom fused helpers for the Wan (Wan2.1 / Wan2.2) transformer.

Everything here is inert unless its own environment switch is exported, so with
no switch set the calling code takes the byte-for-byte original path.  The fast
path additionally guards on the exact shapes / dtypes / layouts it requires and
returns ``None`` otherwise, letting the caller fall through to the reference
implementation (degrade in speed, never in correctness).

Switches
--------
XFUSER_HL_WAN_FUSED_QK_NORM_ROPE
    Fuse ``RMSNorm(q) -> interleaved RoPE`` and ``RMSNorm(k) -> interleaved
    RoPE`` into a single Triton kernel (rewrite taxonomy (d)(ii): fuse a chain
    of adjacent bandwidth-bound ops).

    The reference path in ``xFuserWanAttnProcessor.__call__`` is four separate
    passes over a ``[1, S, H*D]`` bf16 tensor:

        query = attn.norm_q(query)            # RMSNorm over H*D, fp32 upcast
        key   = attn.norm_k(key)
        query = apply_rotary_emb(query, ...)  # out[...,0::2]= / out[...,1::2]=
        key   = apply_rotary_emb(key, ...)

    inductor cannot fuse the RoPE into the norm because the RMS reduction sits
    between them, and the two ``out[..., 0::2] = ...`` / ``out[..., 1::2] = ...``
    assignments are uncoalesced stride-2 scatters into a fresh ``empty_like``
    buffer.  Measured on the live Wan2.2-T2V shape (S=10752 per Ulysses rank,
    H=40, D=128, bf16) on MI355X: 987 us eager, 341 us under the run's live
    ``torch.compile(mode="default")``, 91 us fused.  The chain runs twice per
    self-attention layer x 40 layers x 28 steps x 2 CFG branches.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

_TRITON_OK = True
try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    _TRITON_OK = False
    triton = None  # type: ignore
    tl = None  # type: ignore


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def fused_qk_norm_rope_enabled() -> bool:
    return _TRITON_OK and _HAVE_OP and _flag("XFUSER_HL_WAN_FUSED_QK_NORM_ROPE")


# --------------------------------------------------------------------------
# Triton kernel
# --------------------------------------------------------------------------

if _TRITON_OK:

    @triton.jit
    def _hl_norm_rope_row(X, W, O, eps, cos, sin, offs, partner, is_odd, mask,
                          HD: tl.constexpr):
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        # torch.nn.RMSNorm(H*D) -- "rms_norm_across_heads": the reduction spans
        # the whole flattened row, so one program per token owns it entirely.
        var = tl.sum(x * x, axis=0) / HD
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        xn = x * rstd * w
        # Partner element of the interleaved RoPE pair; adjacent address, so it
        # is served from cache rather than a second HBM read.
        xp = tl.load(X + partner, mask=mask, other=0.0).to(tl.float32)
        wp = tl.load(W + partner, mask=mask, other=0.0).to(tl.float32)
        xpn = xp * rstd * wp
        # even slot: x_even*cos - x_odd*sin ; odd slot: x_even*sin + x_odd*cos
        out = tl.where(is_odd, xpn * sin + xn * cos, xn * cos - xpn * sin)
        tl.store(O + offs, out.to(O.dtype.element_ty), mask=mask)

    @triton.jit
    def _hl_qk_norm_rope_kernel(
        Q, K, WQ, WK, COS, SIN, OQ, OK,
        q_stride_s, k_stride_s, o_stride_s, f_stride_s,
        eps,
        HD: tl.constexpr,      # H * D, the RMSNorm width
        D: tl.constexpr,       # head dim
        BLOCK: tl.constexpr,   # >= HD, power of two
    ):
        t = tl.program_id(0)

        offs = tl.arange(0, BLOCK)
        mask = offs < HD

        # Interleaved RoPE: elements 2p and 2p+1 of every head form a pair, and
        # diffusers builds the tables with repeat_interleave(2), so the value the
        # reference reads as ``freqs_cos[..., 0::2]`` lives at head-slot 2p and
        # ``freqs_sin[..., 1::2]`` at head-slot 2p+1.  `offs` walks the flattened
        # H*D row, so (offs % D) is the position inside the head.
        in_head = offs % D
        is_odd = (in_head % 2) == 1
        partner = tl.where(is_odd, offs - 1, offs + 1)
        even_in_head = in_head - (in_head % 2)
        cos = tl.load(COS + t * f_stride_s + even_in_head, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(SIN + t * f_stride_s + even_in_head + 1, mask=mask, other=0.0).to(tl.float32)

        _hl_norm_rope_row(Q + t * q_stride_s, WQ, OQ + t * o_stride_s, eps,
                          cos, sin, offs, partner, is_odd, mask, HD=HD)
        _hl_norm_rope_row(K + t * k_stride_s, WK, OK + t * o_stride_s, eps,
                          cos, sin, offs, partner, is_odd, mask, HD=HD)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# --------------------------------------------------------------------------
# Opaque custom op so torch.compile does not have to trace the launch
# --------------------------------------------------------------------------

_HAVE_OP = False

if _TRITON_OK and hasattr(torch, "library") and hasattr(torch.library, "custom_op"):

    try:

        @torch.library.custom_op("xfuser_hl::wan_qk_norm_rope", mutates_args=())
        def _wan_qk_norm_rope(
            query: torch.Tensor,
            key: torch.Tensor,
            weight_q: torch.Tensor,
            weight_k: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            heads: int,
            eps: float,
        ) -> List[torch.Tensor]:
            B, S, HD = query.shape
            D = HD // heads
            BLOCK = _next_pow2(HD)
            oq = torch.empty((B, S, heads, D), dtype=query.dtype, device=query.device)
            ok = torch.empty((B, S, heads, D), dtype=key.dtype, device=key.device)
            # num_warps=1 gives one wavefront the whole contiguous H*D row:
            # perfectly coalesced, and S programs is ample parallelism.
            # Measured best of {1,2,4,8,16} on MI355X at S=10752/HD=5120.
            _hl_qk_norm_rope_kernel[(S,)](
                query, key, weight_q, weight_k, freqs_cos, freqs_sin, oq, ok,
                query.stride(1), key.stride(1), heads * D, freqs_cos.stride(1),
                eps,
                HD=HD, D=D, BLOCK=BLOCK,
                num_warps=1, num_stages=2,
            )
            return [oq, ok]

        @_wan_qk_norm_rope.register_fake
        def _(query, key, weight_q, weight_k, freqs_cos, freqs_sin, heads, eps):
            B, S, HD = query.shape
            D = HD // heads
            return [
                query.new_empty((B, S, heads, D)),
                key.new_empty((B, S, heads, D)),
            ]

        _HAVE_OP = True
    except Exception:  # pragma: no cover - registration must never be fatal
        _HAVE_OP = False


# --------------------------------------------------------------------------
# Guarded entry point
# --------------------------------------------------------------------------

def fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q: torch.nn.Module,
    norm_k: torch.nn.Module,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    heads: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(q, k)`` normed, head-split and rotated, or ``None`` to fall back.

    ``query`` / ``key`` are the raw projection outputs of shape ``[B, S, H*D]``.
    The returned tensors are ``[B, S, H, D]`` -- exactly what the reference path
    produces after ``unflatten(2, (heads, -1))`` followed by ``apply_rotary_emb``.
    """
    if not (_TRITON_OK and _HAVE_OP):
        return None
    # Inference-only fast path: the fused op has no autograd formula, so any
    # grad-enabled call falls back to the reference implementation.
    if torch.is_grad_enabled():
        return None
    if query.device.type != "cuda" or query.dim() != 3 or key.dim() != 3:
        return None
    if query.shape != key.shape or query.dtype != key.dtype:
        return None
    if query.shape[0] != 1:
        return None
    if not isinstance(norm_q, torch.nn.RMSNorm) or not isinstance(norm_k, torch.nn.RMSNorm):
        return None
    wq = getattr(norm_q, "weight", None)
    wk = getattr(norm_k, "weight", None)
    if wq is None or wk is None or wq.dtype != query.dtype or wk.dtype != key.dtype:
        return None
    if not wq.is_contiguous() or not wk.is_contiguous():
        return None

    _, S, HD = query.shape
    if heads <= 0 or HD % heads:
        return None
    D = HD // heads
    if (D % 2) or D > 512:
        return None
    if tuple(norm_q.normalized_shape) != (HD,) or tuple(norm_k.normalized_shape) != (HD,):
        return None
    eps_q, eps_k = norm_q.eps, norm_k.eps
    if eps_q is None or eps_k is None or eps_q != eps_k:
        return None

    # freqs: [1, S, 1, D], last dim contiguous (diffusers WanRotaryPosEmbed)
    if freqs_cos.dim() != 4 or freqs_sin.shape != freqs_cos.shape:
        return None
    if tuple(freqs_cos.shape) != (1, S, 1, D):
        return None
    if freqs_cos.dtype != freqs_sin.dtype or freqs_cos.dtype not in (torch.float32, torch.float64):
        return None
    if freqs_cos.stride(-1) != 1 or freqs_sin.stride(-1) != 1:
        return None
    if freqs_cos.stride(1) != freqs_sin.stride(1):
        return None
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return None
    if _next_pow2(HD) > 8192:
        return None

    oq, ok = torch.ops.xfuser_hl.wan_qk_norm_rope(
        query, key, wq, wk, freqs_cos, freqs_sin, heads, float(eps_q)
    )
    return oq, ok
