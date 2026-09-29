"""
AITER flash attention.
"""

from typing import Tuple

import torch
from aiter import flash_attn_func, flash_attn_varlen_func
from torch.library import custom_op

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.spec import AttnCall

# AITER FAv3 on gfx942 supports three different rounding modes:
#   0 = RTNE (round to nearest even)
#   1 = RTNA (round to nearest away)
#   2 = RTZ  (round to zero)
_BF16_ROUNDING_MODE = 2


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


@custom_op("xfuser::aiter_attn", mutates_args=())
def _dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def aiter_attention(query, key, value, call: AttnCall):
    """BSHD kernel; packs K/V when the model supplies varlen indices."""
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
