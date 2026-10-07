# Adapted from Tencent-Hunyuan/Prism (https://github.com/Tencent-Hunyuan/Prism),
# hymm/models/modules/block_sparse_attention/{bsa_interface,flash_attn_bsa_varlen_mask}.py.
# Modified for xDiT: forward pass of the uniform 3D-block path only;
# ROCm launch preset, fp32 casts and a torch.library wrapper added.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Prism's block-sparse attention over a 3D token grid, forward only.

Vendored from Tencent-Hunyuan/Prism ``hymm/models/modules/block_sparse_attention``
(``bsa_interface.py`` and ``flash_attn_bsa_varlen_mask.py``), keeping the uniform
block-shape path Prism samples with: tokens are grouped into ``t x h x w`` blocks,
every query block scores every key block by mean-pooled ``q . k``, keeps a top-k
(``sparsity``) and/or top-p (``cdf_threshold``) set, and attends to those blocks
only. Selection keeps upstream's ``torch.compile`` so its rounding, and with it
which blocks win, matches the reference. The Triton kernels are unchanged.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Upstream presets (autotuning off). The aligned kernel serves blocks of up to
# 128 tokens, which covers the 4x4x4 = 64 Prism samples with.
_FWD_PRESET = {
    "default": {"BLOCK_N": 64, "num_stages": 3, "num_warps": 8},
    "BLOCK_N_LG=64": {"BLOCK_N": 64, "num_stages": 3, "num_warps": 4},
}
_FWD_ALIGN_PRESET = {
    "default": {"num_stages": 3, "num_warps": 8},
    "BLOCK_N_LG=64": {"num_stages": 3, "num_warps": 4},
}
if torch.version.hip:
    # Swept on MI350X (gfx950) at Prism's Ulysses-8 shape, 5 heads x 82680 tokens x 128,
    # sparsity 0.75: 16x16 MFMA tiles over two stages run 64-token blocks in 8.8 ms
    # against 15.2 ms for upstream's preset, within 1 bf16 ulp of its output.
    _FWD_ALIGN_PRESET["BLOCK_N_LG=64"] = {
        "num_stages": 2,
        "num_warps": 4,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
    }


@triton.jit
def _attn_fwd_bsa_varlen(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    block_indices,  # [B, H, M_COMPRESS, S_MAX]
    block_indices_lens,  # [B, H, M_COMPRESS]
    kv_valid_mask,  # [N_CTX] bool, True=valid (shared across B,H)
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_lz,
    stride_lh,
    stride_lm,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # Inductor hands a captured kernel Python floats as fp64; keep the softmax state fp32.
    qk_scale = sm_scale.to(tl.float32)
    qk_scale *= 1.44269504  # 1/ln2
    q = tl.load(Q_block_ptr)
    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo, hi = block_id * BLOCK_N_LG, (block_id + 1) * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N)
        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        V_block_ptr_i = tl.advance(V_block_ptr, (lo, 0))
        mask_offset = lo

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            kT = tl.load(KT_block_ptr_i)
            qkT = tl.dot(q, kT)

            if HAS_KV_MASK:
                offs_n = mask_offset + tl.arange(0, BLOCK_N)
                k_valid = tl.load(kv_valid_mask + offs_n)
                qkT = tl.where(k_valid[None, :], qkT, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qkT, 1) * qk_scale)
            qkT = qkT * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qkT)

            if HAS_KV_MASK:
                p = tl.where(k_valid[None, :], p, 0.0)

            # A fully padded block processed first leaves m_i == m_ij == -inf; it
            # contributes nothing, so leave acc and l_i untouched rather than NaN.
            alpha = tl.where(m_ij == float("-inf"), 1.0, tl.math.exp2(m_i - m_ij))
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V_block_ptr_i)
            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            V_block_ptr_i = tl.advance(V_block_ptr_i, (BLOCK_N, 0))
            KT_block_ptr_i = tl.advance(KT_block_ptr_i, (0, BLOCK_N))
            mask_offset += BLOCK_N

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_bsa_varlen_align(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    block_indices,  # [B, H, M_COMPRESS, S_MAX]
    block_indices_lens,  # [B, H, M_COMPRESS]
    kv_valid_mask,  # [N_CTX] bool
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_lz,
    stride_lh,
    stride_lm,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N_LG, HEAD_DIM),
        order=(1, 0),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N_LG),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # Inductor hands a captured kernel Python floats as fp64; keep the softmax state fp32.
    qk_scale = sm_scale.to(tl.float32)
    qk_scale *= 1.44269504  # 1/ln2
    q = tl.load(Q_block_ptr)
    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo = block_id * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N_LG)
        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        V_block_ptr_i = tl.advance(V_block_ptr, (lo, 0))

        kT = tl.load(KT_block_ptr_i)
        qkT = tl.dot(q, kT)

        if HAS_KV_MASK:
            offs_n = lo + tl.arange(0, BLOCK_N_LG)
            k_valid = tl.load(kv_valid_mask + offs_n)
            qkT = tl.where(k_valid[None, :], qkT, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qkT, 1) * qk_scale)
        qkT = qkT * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qkT)

        if HAS_KV_MASK:
            p = tl.where(k_valid[None, :], p, 0.0)

        # See _attn_fwd_bsa_varlen: a fully padded first block must not produce NaN.
        alpha = tl.where(m_ij == float("-inf"), 1.0, tl.math.exp2(m_i - m_ij))
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr_i)
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@torch.compile
def _mean_pool(x: torch.Tensor, block_size: int, valid_mask: torch.Tensor) -> torch.Tensor:
    """Per-block mean over the tokens ``valid_mask`` marks real; ``x`` is block-ordered."""
    B, H, S, D = x.shape
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        pad_len = num_block * block_size - S
        x = F.pad(x, (0, 0, 0, pad_len))
        valid_mask = F.pad(valid_mask, (0, pad_len), value=False)
    x_blocks = x.view(B, H, num_block, block_size, D)
    mask_f = valid_mask.view(num_block, block_size).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    counts = mask_f.sum(dim=3, keepdim=True).clamp(min=1.0)
    return (x_blocks * mask_f).sum(dim=3) / counts.squeeze(3)


@torch.compile
def _mean_pool_unmasked(x: torch.Tensor, block_size: int) -> torch.Tensor:
    B, H, S = x.shape[:3]
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        x = F.pad(x, (0, 0, 0, num_block * block_size - S))
    return x.view(B, H, num_block, block_size, -1).mean(dim=3)


@torch.compile
def _block_scores(q, k):
    return torch.matmul(q, k.transpose(-1, -2))


@torch.compile
def _select_topk(score, sparsity):
    num_selected = int((1 - sparsity) * score.shape[-1])
    block_indices = torch.sort(torch.topk(score, num_selected)[1], dim=-1)[0]
    lens = torch.full(score.shape[:3], num_selected, dtype=torch.int32, device=score.device)
    return block_indices, lens


@torch.compile
def _select_cdf(score, cdf_threshold, sm_scale):
    weights = torch.softmax(score * sm_scale, dim=-1)
    Sk = weights.shape[-1]
    upper_bound = min(Sk, int(cdf_threshold * Sk) + 1)
    topk_vals, topk_idx = torch.topk(weights, k=upper_bound, dim=-1, largest=True, sorted=True)
    cdf = torch.cumsum(topk_vals, dim=-1)
    # Nucleus: the smallest set whose cumulative weight reaches the threshold.
    num_selected = ((cdf < cdf_threshold).to(torch.int32).sum(dim=-1, keepdim=True) + 1).clamp(min=1, max=Sk)
    pos = torch.arange(upper_bound, device=topk_idx.device)
    block_indices = torch.sort(topk_idx.contiguous().masked_fill(pos >= num_selected, Sk), dim=-1)[0]
    return block_indices, num_selected.squeeze(-1).to(torch.int32)


@torch.compile
def _select_cdf_topk(score, sparsity, cdf_threshold, sm_scale):
    weights = torch.softmax(score * sm_scale, dim=-1)
    Sk = weights.shape[-1]
    num_selected_topk = max(1, int((1 - sparsity) * Sk))
    upper_bound = min(Sk, max(num_selected_topk, int(cdf_threshold * Sk) + 1))
    topk_vals, topk_idx = torch.topk(weights, k=upper_bound, dim=-1, largest=True, sorted=True)
    cdf = torch.cumsum(topk_vals, dim=-1)
    # Top-p, but never fewer blocks than the top-k floor.
    num_selected = (cdf < cdf_threshold).to(torch.int32).sum(dim=-1, keepdim=True) + 1
    num_selected = num_selected.clamp(min=num_selected_topk, max=Sk)
    pos = torch.arange(upper_bound, device=topk_idx.device)
    block_indices = torch.sort(topk_idx.contiguous().masked_fill(pos >= num_selected, Sk), dim=-1)[0]
    return block_indices, num_selected.squeeze(-1).to(torch.int32)


def _to_blocks(x, grid, chunk):
    (Nt, Nh, Nw), (t, h, w) = grid, chunk
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, t, Nh, h, Nw, w, D).permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
    return x.contiguous().view(B, H, -1, D)


def _from_blocks(x, grid, chunk):
    (Nt, Nh, Nw), (t, h, w) = grid, chunk
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, Nh, Nw, t, h, w, D).permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
    return x.contiguous().view(B, H, -1, D)


def _mask_to_blocks(mask, grid, chunk):
    (Nt, Nh, Nw), (t, h, w) = grid, chunk
    return mask.view(Nt, t, Nh, h, Nw, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1)


def _sparse_attention(q, k, v, chunk_size, sparsity, cdf_threshold, kv_valid_mask):
    """Block selection and the sparse kernel over block-ordered ``[B, H, S, D]``."""
    head_dim = q.shape[-1]
    if head_dim not in (16, 32, 64, 128, 256):
        raise ValueError(f"Prism block-sparse attention needs a head dim of 16-256 (power of two), got {head_dim}.")
    sm_scale = 1 / head_dim**0.5

    if sparsity is None and cdf_threshold is None:
        raise ValueError("Prism block-sparse attention needs a sparsity, a cdf_threshold, or both.")

    if kv_valid_mask is not None:
        q_cmp = _mean_pool(q, chunk_size, kv_valid_mask)
        k_cmp = _mean_pool(k, chunk_size, kv_valid_mask)
    else:
        q_cmp = _mean_pool_unmasked(q, chunk_size)
        k_cmp = _mean_pool_unmasked(k, chunk_size)
    score = _block_scores(q_cmp, k_cmp)
    if cdf_threshold is None:
        block_indices, block_indices_lens = _select_topk(score, sparsity)
    elif sparsity is None:
        block_indices, block_indices_lens = _select_cdf(score, cdf_threshold, sm_scale)
    else:
        block_indices, block_indices_lens = _select_cdf_topk(score, sparsity, cdf_threshold, sm_scale)

    B, H, S, D = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    preset = "BLOCK_N_LG=64" if chunk_size == 64 else "default"
    if chunk_size > 128:
        kernel, config = _attn_fwd_bsa_varlen, _FWD_PRESET[preset]
    else:
        kernel, config = _attn_fwd_bsa_varlen_align, _FWD_ALIGN_PRESET[preset]
    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()
    mask = kv_valid_mask if kv_valid_mask is not None else torch.empty(0, dtype=torch.bool, device=q.device)
    kernel[(triton.cdiv(S, chunk_size), B * H, 1)](
        q,
        k,
        v,
        sm_scale,
        lse,
        out,
        block_indices,
        block_indices_lens,
        mask,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *block_indices.stride(),
        *block_indices_lens.stride(),
        H,
        S,
        D,
        BLOCK_M=chunk_size,
        BLOCK_N_LG=chunk_size,
        SPARSITY=sparsity,
        HAS_KV_MASK=int(kv_valid_mask is not None),
        **config,
    )
    return out


def _block_sparse_attention_3d(q, k, v, thw, chunk_thw, sparsity, cdf_threshold):
    """See block_sparse_attention_3d."""
    T, H, W = thw
    t, h, w = chunk_thw
    pad_t, pad_h, pad_w = (-T) % t, (-H) % h, (-W) % w
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, heads, _, D = q.shape
    valid_mask = None
    if pad_t or pad_h or pad_w:
        T_p, H_p, W_p = T + pad_t, H + pad_h, W + pad_w
        valid_mask = (
            (torch.arange(T_p, device=q.device)[:, None, None] < T)
            & (torch.arange(H_p, device=q.device)[None, :, None] < H)
            & (torch.arange(W_p, device=q.device)[None, None, :] < W)
        ).reshape(-1)

        def pad(x):
            x = F.pad(x.view(B, heads, T, H, W, D), (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
            return x.reshape(B, heads, -1, D).contiguous()

        q, k, v = pad(q), pad(k), pad(v)
    else:
        T_p, H_p, W_p = T, H, W

    grid = (T_p // t, H_p // h, W_p // w)
    block_mask = _mask_to_blocks(valid_mask, grid, chunk_thw) if valid_mask is not None else None
    out = _sparse_attention(
        _to_blocks(q, grid, chunk_thw),
        _to_blocks(k, grid, chunk_thw),
        _to_blocks(v, grid, chunk_thw),
        t * h * w,
        sparsity,
        cdf_threshold,
        block_mask,
    )
    out = _from_blocks(out, grid, chunk_thw)
    if valid_mask is not None:
        out = out.view(B, heads, T_p, H_p, W_p, D)[:, :, :T, :H, :W].reshape(B, heads, T * H * W, D)
    return out.contiguous()


# An opaque op to torch.compile: traced, the Triton launch is re-issued by Inductor
# without the ROCm launch options above, and ran 16.8 ms against 12.2 ms eager.
@torch.library.custom_op("xfuser::prism_block_sparse_attention_3d", mutates_args=())
def _block_sparse_attention_3d_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    thw: list[int],
    chunk_thw: list[int],
    sparsity: Optional[float],
    cdf_threshold: Optional[float],
) -> torch.Tensor:
    return _block_sparse_attention_3d(q, k, v, tuple(thw), tuple(chunk_thw), sparsity, cdf_threshold)


@_block_sparse_attention_3d_op.register_fake
def _(q, k, v, thw, chunk_thw, sparsity, cdf_threshold):
    return q.new_empty(q.shape)


def block_sparse_attention_3d(q, k, v, thw, chunk_thw=(4, 4, 4), sparsity=0.75, cdf_threshold=None):
    """Self-attention over a ``T x H x W`` token grid, ``[B, heads, T*H*W, D]`` in THW order.

    The grid is zero-padded up to whole blocks; padded keys are masked out of both
    the block scores and the attention, and padded queries are dropped from the
    output. ``sparsity`` is the fraction of key blocks each query block skips
    (top-k); ``cdf_threshold`` keeps the smallest set of blocks holding that much
    softmax weight (top-p). With both, top-p is floored at the top-k count.
    """
    return _block_sparse_attention_3d_op(q, k, v, list(thw), list(chunk_thw), sparsity, cdf_threshold)
