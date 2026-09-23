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

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import ARCH
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


# Legacy gates this on architecture (_AITER_MHA_V4.enabled is
# "gfx942" or "gfx950" in the arch name), not on whether mha_v4 imports. The
# symbol is arch-independent, so an import test would take the MHA v4 path on
# RDNA4, where legacy takes the legacy path.
_MHA_V4_ARCH = ARCH("gfx950", "gfx942")


def _mha_v4_eligible(query, call: AttnCall) -> bool:
    return (
        _MHA_V4_ARCH.satisfied()
        and call.varlen is None
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
    q, k = hadamard.rotate_qk(q, k)

    launch = _legacy_dense if call.varlen is None else _legacy_varlen
    return from_bshd(launch(q, k, v, call)), None


def aiter_fp8(query, key, value, call: AttnCall):
    if call.attention_kwargs.get("pre_quantized", False):
        return _pre_quantized(query, key, value, call)
    if _mha_v4_eligible(query, call):
        return _mha_v4(query, key, value, call)
    return _legacy(query, key, value, call)


