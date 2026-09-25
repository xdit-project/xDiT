"""AITER per-tensor FP8 attention.

Three paths, picked by what the call carries:

  pre-quantised   Q/K/V already fp8 from fp8 comms, descales in the kwargs
  MHA v4          dense, head_dim 128, non-causal -- the kernel owns rotation
                  and quantisation
  rotate here     everything else: rotate Q/K, quantise, then dense or varlen

"""

from typing import Optional

import aiter
import torch
from torch.library import custom_op, register_fake

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import ARCH, SYMBOL, resolve
from xfuser.core.attention.spec import AttnCall
from xfuser.envs import environment_variables


def _static_scale() -> Optional[float]:
    """AITER_FP8_STATIC_SCALE_WITH_DESCALE, when set above 1."""
    try:
        value = float(environment_variables["AITER_FP8_STATIC_SCALE_WITH_DESCALE"]())
    except (TypeError, ValueError):
        return None
    return value if value > 1 else None


def _quantize(query, key, value):
    # flash_attn_fp8_pertensor_func takes descale vectors, so they are always
    # passed; there is no no-descale branch.
    quant_dtype = aiter.dtypes.fp8
    static = _static_scale()
    scale = (
        None if static is None
        else torch.tensor(static, dtype=torch.float32, device=query.device)
    )
    out = []
    for tensor in (query, key, value):
        quantized, descale = aiter.per_tensor_quant(
            tensor, scale=scale, quant_dtype=quant_dtype,
            dtypeMax=torch.finfo(quant_dtype).max,
        )
        out.append((quantized.to(quant_dtype), descale))
    return out


# Both halves matter: the symbol is arch-independent, so importability alone
# would take this path on an arch the kernel does not run on, and the arch alone
# would take it on a build that predates MHA v4, where the import raises.
#
# Decided at import, not per call: evaluating a Requirement inside a compiled
# region is a fullgraph failure, which
# test_requirement_is_not_traceable_under_fullgraph pins.
_USE_MHA_V4 = (
    ARCH("gfx950", "gfx942")
    & SYMBOL("aiter.ops.mha_v4:mha_v4")
    & SYMBOL("aiter.ops.mha_v4:mha_v4_packed")
    & SYMBOL("aiter.ops.mha_v4:native_fp8_format")
    & SYMBOL("aiter.ops.mha_v4:AttentionScaleMode")
).satisfied()

# The spec's requires deliberately omits mha_v4: the rotate-here path below
# serves builds without it, and requiring it would refuse AITER_FP8 outright on
# an AITER that has no MHA v4. Hence a conditional import rather than a plain
# one at the top of the module.
if _USE_MHA_V4:
    from aiter.ops.mha_v4 import (
        AttentionScaleMode,
        mha_v4,
        mha_v4_packed,
        native_fp8_format,
    )


def _mha_v4_eligible(query, call: AttnCall) -> bool:
    return (
        _USE_MHA_V4
        and call.varlen is None
        and query.is_cuda
        and query.shape[-1] == 128
        and not call.is_causal
    )


def _pre_quantized(query, key, value, call: AttnCall):
    kwargs = call.attention_kwargs
    if call.varlen is not None:
        raise NotImplementedError(
            "fp8 comms pre-quantized attention does not support varlen packing; "
            "the indices_k mask would be silently dropped and dense attention "
            "would run over padded keys."
        )
    q, k, v = to_bshd(query, key, value, contiguous=True)
    softmax_scale = q.shape[-1] ** -0.5

    if _mha_v4_eligible(query, call):
        # Already fp8 with per-tensor descales, which is exactly what the
        # packed entry point takes -- no quantisation step in between.
        fp8 = native_fp8_format()
        per_tensor = AttentionScaleMode.F32_PER_TENSOR
        out = mha_v4_packed(
            q, k, v,
            kwargs["q_descale"], kwargs["k_descale"], kwargs["v_descale"],
            fp8, fp8, fp8,
            per_tensor, per_tensor, per_tensor,
            softmax_scale=softmax_scale,
        )
    else:
        out = aiter.flash_attn_fp8_pertensor_func(
            q, k, v,
            causal=call.is_causal,
            softmax_scale=softmax_scale,
            q_descale=kwargs["q_descale"],
            k_descale=kwargs["k_descale"],
            v_descale=kwargs["v_descale"],
        )
    return from_bshd(out), None


def _mha_v4(query, key, value, call: AttnCall):
    """The raw MHA v4 API owns canonical Q/K rotation and quantisation."""
    q, k, v = to_bshd(query, key, value, contiguous=True)
    fp8 = native_fp8_format()
    return from_bshd(mha_v4(q, k, v, fp8, fp8, fp8)), None


# Quantisation and the kernel sit behind a custom op so Dynamo steps over them
# rather than tracing the per-tensor max and the dtype casts. Only these two
# paths are wrapped: pre-quantised takes fp8 straight from fp8 comms, and MHA
# v4 quantises inside AITER. Rotation stays outside, as it is a plain matmul
# Dynamo traces happily and fp8 comms may have applied it already.
_VARLEN = resolve("aiter:flash_attn_varlen_fp8_pertensor_func")


@custom_op("xfuser::aiter_fp8_dense", mutates_args=())
def _dense_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    (q, q_descale), (k, k_descale), (v, v_descale) = _quantize(query, key, value)
    return aiter.flash_attn_fp8_pertensor_func(
        q, k, v,
        causal=is_causal, softmax_scale=softmax_scale,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )


@register_fake("xfuser::aiter_fp8_dense")
def _dense_op_fake(query, key, value, softmax_scale, is_causal):
    return torch.empty_like(query)


@custom_op("xfuser::aiter_fp8_varlen", mutates_args=())
def _varlen_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    if _VARLEN is None:
        raise RuntimeError(
            "aiter.flash_attn_varlen_fp8_pertensor_func is not available"
        )
    (q, q_descale), (k, k_descale), (v, v_descale) = _quantize(query, key, value)
    return _VARLEN(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=is_causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )


@register_fake("xfuser::aiter_fp8_varlen")
def _varlen_op_fake(
    query, key, value, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, softmax_scale, is_causal,
):
    return torch.empty_like(query)


def _rotate_and_quantize(query, key, value, call: AttnCall):
    """Rotate Q/K here, quantise, then dense or varlen. Both kernels expect
    pre-rotated Q/K, unlike MHA v4 which does its own."""
    q, k, v = to_bshd(query, key, value, contiguous=True)
    q, k = hadamard.rotate_qk(q, k)

    if call.varlen is None:
        out = _dense_op(q, k, v, q.shape[-1] ** -0.5, call.is_causal)
    else:
        p = pack_kv(q, k, v, call.varlen)
        out = p.unflatten(_varlen_op(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            p.max_seqlen_q, p.max_seqlen_k, p.head_dim ** -0.5, call.is_causal,
        ))
    return from_bshd(out), None


def aiter_fp8(query, key, value, call: AttnCall):
    if call.attention_kwargs.get("pre_quantized", False):
        return _pre_quantized(query, key, value, call)
    if _mha_v4_eligible(query, call):
        return _mha_v4(query, key, value, call)
    return _rotate_and_quantize(query, key, value, call)


