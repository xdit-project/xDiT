"""Dao-AILab FlashAttention, v2 through v4, plus the fp8/fp4 recipes.

Four separate upstream packages with four separate import paths; grouped here
because they share the dense/varlen shape and move together in practice.
"""

import functools

import torch

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import CUDA_CAPABILITY, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

_FA2 = "flash_attn:flash_attn_func"
_FA3 = "flash_attn_interface:flash_attn_func"
_FA4 = "flash_attn.cute.interface:flash_attn_func"


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
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    # Deliberately not contiguous: FAv2 accepts the permuted view, and the
    # legacy backend never forced a copy here.
    q, k, v = to_bshd(query, key, value)

    if call.varlen is None:
        out, softmax_lse, _ = flash_attn_func(
            q, k, v, dropout_p=call.dropout_p, causal=call.is_causal,
            return_attn_probs=True,
        )
    else:
        p = pack_kv(q, k, v, call.varlen)
        out = flash_attn_varlen_func(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            dropout_p=call.dropout_p, softmax_scale=p.head_dim ** -0.5,
            causal=call.is_causal,
        )
        out, softmax_lse = p.unflatten(out), None

    return from_bshd(out), softmax_lse


def flash_3(query, key, value, call: AttnCall):
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, softmax_lse = flash_attn_func(
            q, k, v, causal=call.is_causal, return_attn_probs=True
        )
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, softmax_lse = flash_attn_varlen_func(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            causal=call.is_causal, return_attn_probs=True,
        )
        out = p.unflatten(out)

    return from_bshd(out), softmax_lse


def flash_3_fp8(query, key, value, call: AttnCall):
    from flash_attn_interface import flash_attn_func

    query, q_descale = per_tensor_quant(query)
    key, k_descale = per_tensor_quant(key)
    value, v_descale = per_tensor_quant(value)

    q, k, v = to_bshd(query, key, value, contiguous=True)
    out, softmax_lse = flash_attn_func(
        q, k, v, causal=call.is_causal, return_attn_probs=True,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    return from_bshd(out), softmax_lse


@torch.compiler.disable  # FAv4 is not traceable
def flash_4(query, key, value, call: AttnCall):
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, softmax_lse = flash_attn_func(q, k, v, causal=call.is_causal)
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, softmax_lse = flash_attn_varlen_func(
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
    from flash_attn.cute.interface import flash_attn_func

    from xfuser.core.distributed.fp4_quantize import quantize_qk_to_fp4

    q, k, v = to_bshd(query, key, value, contiguous=True)
    q_fp4, q_scale = quantize_qk_to_fp4(q)
    k_fp4, k_scale = quantize_qk_to_fp4(k)

    out, softmax_lse = flash_attn_func(
        q_fp4, k_fp4, v, causal=call.is_causal, mSFQ=q_scale, mSFK=k_scale
    )
    return from_bshd(out), softmax_lse


# FAv2 is NOT CUDA-only: envs.check_flash_attn admits ROCm, and
# _select_attention_backend picks FLASH on HIP when aiter is absent. Gating it
# on cuda would take that path away from AMD users, so it stays symbol-gated.
# FAv3 (Hopper), FAv4 (CUTE DSL) and SageAttention have no ROCm build; the
# platform check turns "not importable" into a message that says why.
SPECS = [
    Spec(AttentionBackendType.FLASH, impl=flash_2, returns_lse=True,
         requires=SYMBOL(_FA2) & SYMBOL("flash_attn:flash_attn_varlen_func")),

    Spec(AttentionBackendType.FLASH_3, impl=flash_3, returns_lse=True,
         requires=PLATFORM("cuda") & SYMBOL(_FA3)
                & SYMBOL("flash_attn_interface:flash_attn_varlen_func")),

    Spec(AttentionBackendType.FLASH_3_FP8, impl=flash_3_fp8, returns_lse=True,
         low_precision=True, requires=PLATFORM("cuda") & SYMBOL(_FA3)),

    # FAv4 produces an LSE but the legacy ring blocklist excludes it, so it
    # does not participate in ring attention.
    Spec(AttentionBackendType.FLASH_4, impl=flash_4,
         requires=PLATFORM("cuda") & SYMBOL(_FA4)
                & SYMBOL("flash_attn.cute.interface:flash_attn_varlen_func")),

    Spec(AttentionBackendType.FLASH_4_FP4, impl=flash_4_fp4, low_precision=True,
         requires=PLATFORM("cuda") & CUDA_CAPABILITY((10, 0)) & SYMBOL(_FA4)
                & SYMBOL("xfuser.core.distributed.fp4_quantize:quantize_qk_to_fp4")),
]
