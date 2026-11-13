"""
AITER flash attention.
"""

import os
from typing import Tuple

import torch
from aiter import flash_attn_func, flash_attn_varlen_func
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import fmha_v3_fwd, fmha_v3_varlen_fwd
from torch.library import custom_op

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.spec import AttnCall

# AITER FAv3 on gfx942 supports three different rounding modes:
#   0 = RTNE (round to nearest even)
#   1 = RTNA (round to nearest away)
#   2 = RTZ  (round to zero)
_BF16_ROUNDING_MODE = 2
_NUM_CU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
_IS_GFX950 = get_gfx() == "gfx950"


# ---------------------------------------------------------------------------
# custom ops
#
# AITER annotates both entry points with `X | None`, which Dynamo cannot wrap
# sourcelessly -- it aborts with "SourcelessBuilder.create does not know how to
# wrap types.UnionType" as soon as it inlines the nested _validate_cu helper,
# so any compiled model reaching either one fails to build a graph. Wrapping
# each makes it a single opaque node.
#
# Registered at module import, which happens at backend selection, outside
# every compiled region -- the same reason Impl resolves there.
# ---------------------------------------------------------------------------


def _v3_eligible(query, key, value, dropout_p, is_causal):
    return (
        _IS_GFX950
        and query.dtype == torch.bfloat16
        and query.shape[-1] == 128
        and value.shape[-1] == 128
        and query.shape[2] == key.shape[2]
        and query.shape[1] > 128
        and dropout_p == 0.0
        and not is_causal
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    )


@custom_op("xfuser::aiter_attn", mutates_args=())
def _dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _v3_eligible(query, key, value, dropout_p, is_causal):
        # Same kernel flash_attn_func routes to, minus its per-call Python dispatch.
        out, lse, _, _ = fmha_v3_fwd(
            query,
            key,
            value,
            0.0,
            query.shape[-1] ** -0.5,
            False,
            -1,
            -1,
            True,
            False,
            _BF16_ROUNDING_MODE,
        )
        return out, lse
    return flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_lse=True,
        return_attn_probs=False,
        how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
    )


@_dense.register_fake
def _dense_fake(query, key, value, dropout_p, is_causal):
    batch, seq_len, heads, _ = query.shape
    return (
        torch.empty_like(query),
        query.new_empty((batch, heads, seq_len), dtype=torch.float32),
    )


@custom_op("xfuser::aiter_attn_varlen", mutates_args=())
def _varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=is_causal,
        return_lse=True,
        return_attn_probs=False,
        how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
    )


@_varlen.register_fake
def _varlen_fake(
    query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, dropout_p, is_causal
):
    total_q, heads, _ = query.shape
    return (
        torch.empty_like(query),
        query.new_empty((heads, total_q), dtype=torch.float32),
    )


@custom_op("xfuser::aiter_attn_splitkv", mutates_args=())
def _splitkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = fmha_v3_varlen_fwd(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0,
        0.0,
        query.shape[-1] ** -0.5,
        0.0,
        False,
        False,
        -1,
        -1,
        True,
        False,
        _BF16_ROUNDING_MODE,
    )
    return out, lse


@_splitkv.register_fake
def _splitkv_fake(query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    total_q, heads, _ = query.shape
    return torch.empty_like(query), query.new_empty((heads, total_q), dtype=torch.float32)


# Split K/V across extra workgroups when B*H*ceil(Sq/256) leaves most CUs idle
# (e.g. 3-6 heads per rank after Ulysses): the v3 kernel runs one 256-row Q tile
# per CU, so a 4K-token call on 6 heads occupies ~100 of 256 CUs.
_KV_SPLITS = os.environ.get("XDIT_AITER_KV_SPLITS", "1")
_SPLITKV_MIN_SEQ = 1024


def _kv_splits(batch, heads, seq_q, seq_k):
    if _KV_SPLITS == "auto":
        tiles = batch * heads * -(-seq_q // 256)
        splits = min(4, _NUM_CU // tiles)
    else:
        splits = int(_KV_SPLITS)
    if batch != 1 or seq_q < _SPLITKV_MIN_SEQ or seq_k < splits * 256:
        return 1
    return max(splits, 1)


def _splitkv_attention(q, k, v, splits):
    """q/k/v BSHD (k/v may be strided); returns (BSHD out, [B, H, Sq] lse)."""
    _, seq_q, heads, head_dim = q.shape
    seq_k = k.shape[1]
    q_rep = q.expand(splits, -1, -1, -1).reshape(splits * seq_q, heads, head_dim)
    # Offsets are built in the traced graph rather than cached: a tensor created
    # inside the op during CUDA-graph warmup would outlive the graph's pool.
    steps = torch.arange(splits + 1, dtype=torch.int32, device=q.device)
    cu_q = steps * seq_q
    cu_k = steps * seq_k // splits
    out, lse = _splitkv(q_rep, k[0], v[0], cu_q, cu_k, seq_q, -(-seq_k // splits))
    lse = lse.view(heads, splits, seq_q)
    lse_max = lse.amax(1, keepdim=True)
    weight = torch.exp(lse - lse_max)
    total = weight.sum(1, keepdim=True)
    weight = (weight / total).permute(1, 2, 0).unsqueeze(-1)
    out = (out.view(splits, seq_q, heads, head_dim).float() * weight).sum(0)
    lse = (lse_max + torch.log(total)).view(1, heads, seq_q)
    return out.to(q.dtype).unsqueeze(0), lse


def aiter_attention(query, key, value, call: AttnCall):
    """BSHD kernel; packs K/V when the model supplies varlen indices."""
    if call.varlen is None and not call.is_causal and call.dropout_p == 0.0:
        batch, heads, seq_q, _ = query.shape
        splits = _kv_splits(batch, heads, seq_q, key.shape[2])
        if splits > 1:
            q = to_bshd(query)
            k, v = to_bshd(key, value)
            out, lse = _splitkv_attention(q, k, v, splits)
            return from_bshd(out), lse

    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, lse = _dense(q, k, v, call.dropout_p, call.is_causal)
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, lse = _varlen(
            p.q,
            p.k,
            p.v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.head_dim**-0.5,
            call.dropout_p,
            call.is_causal,
        )
        out = p.unflatten(out)
        lse = lse.view(p.heads, p.batch, p.seq_len).permute(1, 0, 2).contiguous()

    return from_bshd(out), lse
