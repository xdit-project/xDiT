"""AITER per-tensor FP8 attention.

Three paths, picked by what the call carries:

  pre-quantised   Q/K/V already fp8 from fp8 comms, descales in the kwargs
  MHA v4          dense, head_dim 128, non-causal -- the kernel owns rotation
                  and quantisation
  legacy          everything else: rotate Q/K here, quantise, dense or varlen

Only the first two are reachable on a modern build for a typical model; the
legacy path still serves varlen packing, which MHA v4 has no mask for.
"""

from typing import Optional

import torch

from xfuser.core.attention import hadamard
from xfuser.core.attention.constraints import NO_DROPOUT
from xfuser.core.attention.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec
from xfuser.envs import environment_variables


def _static_scale() -> Optional[float]:
    """AITER_FP8_STATIC_SCALE_WITH_DESCALE, when set above 1."""
    try:
        value = float(environment_variables["AITER_FP8_STATIC_SCALE_WITH_DESCALE"]())
    except (TypeError, ValueError):
        return None
    return value if value > 1 else None


def _rotation(query) -> torch.Tensor:
    """128-blocked for head_dim that is a multiple of 128 (every current
    model); full-head for smaller power-of-two dims such as LTX-2.5 audio."""
    head_dim = query.shape[-1]
    block_r = 128 if head_dim % 128 == 0 else head_dim
    return hadamard.matrix(block_r, str(query.device))


def _quantize(query, key, value):
    # aiter-shim cut 2026-09: AITER_FP8_HAS_DESCALE (added 2026-03-09) probed
    # whether flash_attn_fp8_pertensor_func took descale vectors. It does, so
    # the no-descale branch and its static scale of 1.0 are gone.
    import aiter

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


def _pre_quantized(query, key, value, call: AttnCall):
    import aiter

    kwargs = call.attention_kwargs
    if call.varlen is not None:
        raise NotImplementedError(
            "fp8 comms pre-quantized attention does not support varlen packing; "
            "the indices_k mask would be silently dropped and dense attention "
            "would run over padded keys."
        )
    q, k, v = to_bshd(query, key, value, contiguous=True)
    out = aiter.flash_attn_fp8_pertensor_func(
        q, k, v,
        causal=call.is_causal,
        softmax_scale=q.shape[-1] ** -0.5,
        q_descale=kwargs["q_descale"],
        k_descale=kwargs["k_descale"],
        v_descale=kwargs["v_descale"],
    )
    return from_bshd(out), None


def _mha_v4_eligible(query, call: AttnCall) -> bool:
    try:
        from aiter.ops.mha_v4 import mha_v4  # noqa: F401  - presence is the test
    except ImportError:
        return False

    return (
        call.varlen is None
        and query.is_cuda
        and query.shape[-1] == 128
        and not call.is_causal
    )


def _mha_v4(query, key, value, call: AttnCall):
    """The raw MHA v4 API owns canonical Q/K rotation and quantisation."""
    from aiter.ops.mha_v4 import mha_v4, native_fp8_format

    q, k, v = to_bshd(query, key, value, contiguous=True)
    fp8 = native_fp8_format()
    return from_bshd(mha_v4(q, k, v, fp8, fp8, fp8)), None


def _legacy_dense(q, k, v, call: AttnCall):
    import aiter

    (qq, q_descale), (kk, k_descale), (vv, v_descale) = _quantize(q, k, v)
    return aiter.flash_attn_fp8_pertensor_func(
        qq, kk, vv,
        causal=call.is_causal, softmax_scale=q.shape[-1] ** -0.5,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )


def _legacy_varlen(q, k, v, call: AttnCall):
    import aiter

    varlen_func = getattr(aiter, "flash_attn_varlen_fp8_pertensor_func", None)
    if varlen_func is None:
        raise RuntimeError(
            "aiter.flash_attn_varlen_fp8_pertensor_func is not available"
        )

    p = pack_kv(q, k, v, call.varlen)
    (qq, q_descale), (kk, k_descale), (vv, v_descale) = _quantize(p.q, p.k, p.v)
    return p.unflatten(varlen_func(
        qq, kk, vv,
        cu_seqlens_q=p.cu_seqlens_q, cu_seqlens_k=p.cu_seqlens_k,
        max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
        softmax_scale=p.head_dim ** -0.5, causal=call.is_causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    ))


def _legacy(query, key, value, call: AttnCall):
    """Rotate Q/K here, quantise, then dense or varlen. Both kernels expect
    pre-rotated Q/K, unlike MHA v4 which does its own."""
    q, k, v = to_bshd(query, key, value, contiguous=True)
    r = _rotation(q)
    q = hadamard.rotate(q, r).contiguous()
    k = hadamard.rotate(k, r).contiguous()

    launch = _legacy_dense if call.varlen is None else _legacy_varlen
    return from_bshd(launch(q, k, v, call)), None


def aiter_fp8(query, key, value, call: AttnCall):
    if call.attention_kwargs.get("pre_quantized", False):
        return _pre_quantized(query, key, value, call)
    if _mha_v4_eligible(query, call):
        return _mha_v4(query, key, value, call)
    return _legacy(query, key, value, call)


SPECS = [
    Spec(
        AttentionBackendType.AITER_FP8,
        impl=aiter_fp8,
        low_precision=True,
        accepts=NO_DROPOUT,
        requires=SYMBOL("aiter:flash_attn_fp8_pertensor_func")
               & SYMBOL("aiter:per_tensor_quant"),
    ),
]
