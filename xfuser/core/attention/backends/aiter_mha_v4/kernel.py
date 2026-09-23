"""MHA v4 launchers. Imported when one of the family is selected, so AITER is
present and its enums can be translated once here rather than per call."""

import inspect

import torch
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    mha_v4_packed,
    mha_v4_q_multiplier,
    native_fp8_format,
    quantize_fp8,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
)

from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttnCall

from .spec import Fmt, MhaV4Format, Scale

# Resolved once at import. MXFP6 and MXFP4 are AITER *aliases* for FP6_E2M3 and
# FP4_E2M1 and do not appear when iterating AttentionFormat; the resolution
# test in tests/attention/test_equivalence.py fails if one is dropped upstream.
FORMAT = {f: getattr(AttentionFormat, f.name) for f in Fmt if f is not Fmt.NATIVE_FP8}
FORMAT[Fmt.NATIVE_FP8] = native_fp8_format()
SCALE = {s: getattr(AttentionScaleMode, s.name) for s in Scale}

HAS_SCALE_MODES = "q_scale_mode" in inspect.signature(mha_v4).parameters


def _read_kv_tile() -> int:
    """Sparge's KV tile must match the kernel's sparse geometry."""
    # Not hoisted: optional, with an arch-derived fallback below.
    try:
        from aiter.ops.mha_v4 import mha_v4_kv_tile

        return int(mha_v4_kv_tile())
    except ImportError:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        return 64 if "gfx942" in arch else 128


KV_TILE = _read_kv_tile()

def _launch(q, k, v, fmt: MhaV4Format, block_mask=None):
    """One MHA v4 launch. Tensors are BSHD."""
    if fmt.is_mxfp8 and not HAS_SCALE_MODES:
        return _launch_mxfp8_legacy(q, k, v, block_mask)

    kwargs = {}
    if fmt.qk_scale is not None:
        kwargs = {
            "q_scale_mode": SCALE[fmt.qk_scale],
            "k_scale_mode": SCALE[fmt.qk_scale],
            "v_scale_mode": SCALE[fmt.v_scale],
        }
    qk = FORMAT[fmt.qk]
    return mha_v4(q, k, v, qk, qk, FORMAT[fmt.v],
                  block_mask=block_mask, **kwargs)


def _launch_mxfp8_legacy(q, k, v, block_mask):
    # aiter-shim: added 2026-08-24, drop once the floor passes the build that
    # gained scale modes. AITER deprecated mha_v4_mxfp8, and its
    # DeprecationWarning is untraceable by Dynamo, which breaks fullgraph=True.
    # Not hoisted: the deprecated entry point, absent on newer AITER.
    from aiter.ops.mha_v4 import mha_v4_mxfp8

    if block_mask is None:
        return mha_v4_mxfp8(q, k, v)
    return mha_v4_mxfp8(q, k, v, block_mask=block_mask)


def _launch_mxfp8_packed(q, k, v, block_mask):
    """Hand-packed MXFP8 sparse path, used when mha_v4 takes no block_mask for
    the deprecated MXFP8 entry point."""
    # Not hoisted: only the sparge specs require this symbol; the dense
    # specs in this family must load without it.
    from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

    lut = block_attn_mask_to_ragged_lut(block_mask, return_none_if_dense=False)
    if lut is None:
        raise RuntimeError("block_attn_mask_to_ragged_lut returned None")
    kv_block_indices, lut_start, lut_count = lut

    softmax_scale = q.shape[-1] ** -0.5
    q_q, q_scale = quantize_mxfp8_q(q, mha_v4_q_multiplier(softmax_scale))
    k_q, k_scale = quantize_mxfp8_k(k)
    v_q, v_scale = quantize_fp8(v)
    fp8 = native_fp8_format()
    return mha_v4_packed(
        q_q, k_q, v_q, q_scale, k_scale, v_scale,
        fp8, fp8, fp8,
        SCALE[Scale.E8M0_PER_1X32],
        SCALE[Scale.E8M0_PER_1X32],
        SCALE[Scale.F32_PER_TENSOR],
        softmax_scale=softmax_scale,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
    )


def mha_v4_dense(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    q, k, v = to_bshd(query, key, value, contiguous=True)
    return from_bshd(_launch(q, k, v, fmt)), None


def mha_v4_sparge(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=256, block_n=KV_TILE,
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=True,
    )
    q, k, v = to_bshd(q, k, v, contiguous=True)

    if fmt.is_mxfp8 and HAS_SCALE_MODES:
        # The deprecated MXFP8 entry point is the one that takes a block_mask;
        # with scale modes present we pack the sparse call by hand instead.
        output = _launch_mxfp8_packed(q, k, v, block_mask)
    else:
        output = _launch(q, k, v, fmt, block_mask=block_mask)

    return restore_sparge_output(from_bshd(output), state), None


