"""Dao-AILab FlashAttention, v2 through v4, plus the fp8/fp4 recipes.

Three independent upstream packages, grouped here because they share the
dense/varlen shape and move together in practice. A machine may have any
subset -- FAv2 on ROCm without v3, say -- so none can be a plain import:
that would make selecting FLASH require FAv3 and FAv4 as well. Each is
resolved once, here, at the import that backend selection triggers. Whichever
is None, `requires` has already refused the specs that need it.
"""

import functools

import torch

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.spec import AttnCall

from .spec import (
    _FA2,
    _FA2_VARLEN,
    _FA3,
    _FA3_VARLEN,
    _FA4,
    _FA4_VARLEN,
    _FP4_QUANT,
)

FA2 = resolve(_FA2)
FA2_VARLEN = resolve(_FA2_VARLEN)
FA3 = resolve(_FA3)
FA3_VARLEN = resolve(_FA3_VARLEN)
FA4 = resolve(_FA4)
FA4_VARLEN = resolve(_FA4_VARLEN)
QUANTIZE_QK_TO_FP4 = resolve(_FP4_QUANT)


@functools.lru_cache(maxsize=None)
def _dtype_max(dtype):
    try:
        return torch.finfo(dtype).max
    except TypeError:
        return torch.iinfo(dtype).max


def per_tensor_quant(x, quant_dtype=torch.float8_e4m3fn):
    """Per-tensor fp8 quantisation returning the descale broadcast per batch
    and head, which is the layout FAv3 expects."""
    x = x.to(torch.float32)
    scale = torch.abs(x).max() / _dtype_max(quant_dtype)
    return (x / scale).to(quant_dtype), scale.expand(*x.shape[:2]).to(torch.float32)


def flash_2(query, key, value, call: AttnCall):
    # Deliberately not contiguous: FAv2 accepts the permuted view, so this
    # path never forces a copy.
    q, k, v = to_bshd(query, key, value)

    if call.varlen is None:
        out, softmax_lse, _ = FA2(
            q, k, v, dropout_p=call.dropout_p, causal=call.is_causal,
            return_attn_probs=True,
        )
    else:
        p = pack_kv(q, k, v, call.varlen)
        out = FA2_VARLEN(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            dropout_p=call.dropout_p, softmax_scale=p.head_dim ** -0.5,
            causal=call.is_causal,
        )
        out, softmax_lse = p.unflatten(out), None

    return from_bshd(out), softmax_lse


def flash_3(query, key, value, call: AttnCall):
    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, softmax_lse = FA3(
            q, k, v, causal=call.is_causal, return_attn_probs=True
        )
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, softmax_lse = FA3_VARLEN(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            causal=call.is_causal, return_attn_probs=True,
        )
        out = p.unflatten(out)

    return from_bshd(out), softmax_lse


def flash_3_fp8(query, key, value, call: AttnCall):
    query, q_descale = per_tensor_quant(query)
    key, k_descale = per_tensor_quant(key)
    value, v_descale = per_tensor_quant(value)

    q, k, v = to_bshd(query, key, value, contiguous=True)
    out, softmax_lse = FA3(
        q, k, v, causal=call.is_causal, return_attn_probs=True,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    return from_bshd(out), softmax_lse


@torch.compiler.disable  # FAv4 is not traceable
def flash_4(query, key, value, call: AttnCall):
    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, softmax_lse = FA4(q, k, v, causal=call.is_causal)
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, softmax_lse = FA4_VARLEN(
            p.q, p.k, p.v,
            cu_seqlens_q=p.cu_seqlens_q, cu_seqlens_k=p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            causal=call.is_causal,
        )
        out = p.unflatten(out)

    return from_bshd(out), softmax_lse


@torch.compiler.disable
def flash_4_fp4(query, key, value, call: AttnCall):
    """Q and K quantised to NVFP4; V stays bf16."""
    q, k, v = to_bshd(query, key, value, contiguous=True)
    q_fp4, q_scale = QUANTIZE_QK_TO_FP4(q)
    k_fp4, k_scale = QUANTIZE_QK_TO_FP4(k)

    out, softmax_lse = FA4(
        q_fp4, k_fp4, v, causal=call.is_causal, mSFQ=q_scale, mSFK=k_scale
    )
    return from_bshd(out), softmax_lse


