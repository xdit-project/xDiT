"""Fused MOSS-Image attention prologue: per-head RMSNorm + RoPE + Ulysses send layout."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    rope_ptr,
    q_out_ptr,
    k_out_ptr,
    seq_len,
    stride_xs,
    stride_rs,
    local_heads,
    stride_op,
    stride_ohl,
    stride_ot,
    eps,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    head = tl.program_id(1)
    which = tl.program_id(2)
    tokens = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.arange(0, HALF)
    mask = tokens[:, None] < seq_len
    if which == 0:
        x_ptr, w_ptr, o_ptr = q_ptr, qw_ptr, q_out_ptr
    else:
        x_ptr, w_ptr, o_ptr = k_ptr, kw_ptr, k_out_ptr
    x_off = tokens[:, None] * stride_xs + head * (2 * HALF) + dims[None, :]
    a = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + x_off + HALF, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / (2 * HALF) + eps)
    wa = tl.load(w_ptr + dims).to(tl.float32)
    wb = tl.load(w_ptr + HALF + dims).to(tl.float32)
    a = (a * rstd[:, None]).to(tl.bfloat16).to(tl.float32) * wa[None, :]
    b = (b * rstd[:, None]).to(tl.bfloat16).to(tl.float32) * wb[None, :]
    a = a.to(tl.bfloat16).to(tl.float32)
    b = b.to(tl.bfloat16).to(tl.float32)
    r_off = tokens[:, None] * stride_rs + dims[None, :]
    sin = tl.load(rope_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(rope_ptr + r_off + HALF, mask=mask, other=0.0).to(tl.float32)
    out_a = a * cos - b * sin
    out_b = b * cos + a * sin
    o_off = (head // local_heads) * stride_op + (head % local_heads) * stride_ohl + tokens[:, None] * stride_ot
    o_off = o_off + dims[None, :]
    tl.store(o_ptr + o_off, out_a.to(tl.bfloat16), mask=mask)
    tl.store(o_ptr + o_off + HALF, out_b.to(tl.bfloat16), mask=mask)


@triton_op("xfuser::moss_qk_norm_rope", mutates_args={})
def qk_norm_rope_send(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope: torch.Tensor,
    world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [1, S/P, H*D] projections (rows may be strided); rope: [1, S/P, D] (sin | cos).

    Returns RMSNorm'd, rotated q and k laid out as Ulysses send buffers
    [P, S/P, H/P, D] (token-major chunks per destination rank).
    """
    _, seq_len, width = q.shape
    head_dim = q_weight.shape[0]
    heads = width // head_dim
    local_heads = heads // world
    q_out = q.new_empty((world, seq_len, local_heads, head_dim))
    k_out = torch.empty_like(q_out)
    block_t = 32
    grid = (triton.cdiv(seq_len, block_t), heads, 2)
    wrap_triton(_qk_norm_rope_kernel)[grid](
        q,
        k,
        q_weight,
        k_weight,
        rope,
        q_out,
        k_out,
        seq_len,
        q.stride(1),
        rope.stride(1),
        local_heads,
        seq_len * local_heads * head_dim,
        head_dim,
        local_heads * head_dim,
        eps,
        HALF=head_dim // 2,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return q_out, k_out
