"""AITER MHA v4: a quantisation-format table crossed with dense/sparge.

Eighteen enum members are one kernel, one sparsity strategy, and a ten-row
table of (Q/K format, V format). Adding a format is a row plus two enum
members; everything else is derived.

AITER_FP8's *dense* path is not here -- it predates MHA v4 and still carries a
Hadamard-rotation and varlen fallback, so it migrates with those helpers. Its
sparge variant is pure table and is generated below.
"""

import functools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import torch

from xfuser.core.attention.layout import from_bshd, to_bshd
from xfuser.core.attention.requirements import ARCH, PARAM, SYMBOL, Requirement
from xfuser.core.attention.constraints import (
    HEAD_DIM,
    MHA_ONLY,
    NON_CAUSAL,
    NO_DROPOUT,
    NO_VARLEN,
    SELF_ATTENTION,
)
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

_MHA_V4 = "aiter.ops.mha_v4:mha_v4"

GFX950 = ARCH("gfx950")
GFX950_OR_GFX942 = ARCH("gfx950", "gfx942")


# ---------------------------------------------------------------------------
# formats
#
# Local, so the table is plain data that can be defined on a machine without
# AITER. Translated to AITER's own enums inside the launcher, which only runs
# once availability has been established.
# ---------------------------------------------------------------------------

class Fmt(Enum):
    """Names are resolved against AITER's AttentionFormat by name, so a member
    here must match one of its members or aliases. Note MXFP6 and MXFP4 are
    aliases (for FP6_E2M3 and FP4_E2M1) and do not appear when iterating that
    enum; if AITER drops them the resolution test below fails.

    NATIVE_FP8 is the exception: AITER exposes it as a function rather than a
    member, because the concrete type is architecture-dependent (FP8_E4M3 vs
    FP8_E4M3_FNUZ).
    """

    BF16 = auto()
    INT8 = auto()
    NATIVE_FP8 = auto()
    MXFP6 = auto()
    MXFP4 = auto()


class Scale(Enum):
    """Names must match AITER's AttentionScaleMode members."""

    E8M0_PER_1X32 = auto()
    F32_PER_TENSOR = auto()


@functools.lru_cache(maxsize=None)
def _aiter_format(fmt: Fmt):
    from aiter.ops.mha_v4 import AttentionFormat, native_fp8_format

    if fmt is Fmt.NATIVE_FP8:
        return native_fp8_format()
    return getattr(AttentionFormat, fmt.name)


@functools.lru_cache(maxsize=None)
def _aiter_scale(scale: Scale):
    from aiter.ops.mha_v4 import AttentionScaleMode

    return getattr(AttentionScaleMode, scale.name)


@functools.lru_cache(maxsize=1)
def _kv_tile() -> int:
    """Sparge's KV tile must match the kernel's sparse geometry."""
    from aiter.ops.mha_v4 import mha_v4

    try:
        from aiter.ops.mha_v4 import mha_v4_kv_tile

        return int(mha_v4_kv_tile())
    except ImportError:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        return 64 if "gfx942" in arch else 128


@functools.lru_cache(maxsize=1)
def _has_scale_modes() -> bool:
    import inspect

    from aiter.ops.mha_v4 import mha_v4

    return "q_scale_mode" in inspect.signature(mha_v4).parameters


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MhaV4Format:
    name: str
    qk: Fmt
    v: Fmt
    sparge_on: Optional[Requirement] = None   # None = no sparge variant
    qk_scale: Optional[Scale] = None
    v_scale: Optional[Scale] = None
    dense: bool = True                        # False = sparge variant only

    @property
    def is_mxfp8(self) -> bool:
        """MXFP8 is the one row needing block-scaled Q/K, which AITER exposes
        through scale modes rather than a format."""
        return self.qk_scale is Scale.E8M0_PER_1X32


FORMATS = [
    #            name        Q/K              V                sparge on
    MhaV4Format("BF16",      Fmt.BF16,        Fmt.BF16),
    MhaV4Format("BF16FP8",   Fmt.BF16,        Fmt.NATIVE_FP8),
    MhaV4Format("I8FP8",     Fmt.INT8,        Fmt.NATIVE_FP8,  GFX950_OR_GFX942),
    MhaV4Format("F8F6",      Fmt.NATIVE_FP8,  Fmt.MXFP6,       GFX950),
    MhaV4Format("MXFP6",     Fmt.MXFP6,       Fmt.NATIVE_FP8,  GFX950),
    MhaV4Format("F6F4",      Fmt.MXFP6,       Fmt.MXFP4,       GFX950),
    MhaV4Format("MXFP4",     Fmt.MXFP4,       Fmt.NATIVE_FP8,  GFX950),
    MhaV4Format("F4F4",      Fmt.MXFP4,       Fmt.MXFP4,       GFX950),
    MhaV4Format("MXFP8",     Fmt.NATIVE_FP8,  Fmt.NATIVE_FP8,  GFX950,
                qk_scale=Scale.E8M0_PER_1X32, v_scale=Scale.F32_PER_TENSOR),
    # Dense AITER_FP8 lives in the legacy module until its fp8 helpers migrate;
    # the sparge variant needs nothing but mha_v4 and is generated here.
    MhaV4Format("FP8",       Fmt.NATIVE_FP8,  Fmt.NATIVE_FP8,  GFX950_OR_GFX942,
                dense=False),
]


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

DENSE_CALLS = NO_DROPOUT & NON_CAUSAL & NO_VARLEN & HEAD_DIM(128)

# SELF_ATTENTION is stricter than the legacy validator, which checks head
# counts but never that Q and K/V are the same length. Sparge reorders both
# against one spatial layout, so a cross-attention call indexes K/V out of
# bounds -- observed as GPU memory corruption and a cored process, not an
# exception. Declared rather than left latent.
SPARGE_CALLS = DENSE_CALLS & MHA_ONLY & SELF_ATTENTION


def _launch(q, k, v, fmt: MhaV4Format, block_mask=None):
    """One MHA v4 launch. Tensors are BSHD."""
    from aiter.ops.mha_v4 import mha_v4

    if fmt.is_mxfp8 and not _has_scale_modes():
        return _launch_mxfp8_legacy(q, k, v, block_mask)

    kwargs = {}
    if fmt.qk_scale is not None:
        kwargs = {
            "q_scale_mode": _aiter_scale(fmt.qk_scale),
            "k_scale_mode": _aiter_scale(fmt.qk_scale),
            "v_scale_mode": _aiter_scale(fmt.v_scale),
        }
    qk = _aiter_format(fmt.qk)
    return mha_v4(q, k, v, qk, qk, _aiter_format(fmt.v),
                  block_mask=block_mask, **kwargs)


def _launch_mxfp8_legacy(q, k, v, block_mask):
    # aiter-shim: added 2026-08-24, drop once the floor passes the build that
    # gained scale modes. AITER deprecated mha_v4_mxfp8, and its
    # DeprecationWarning is untraceable by Dynamo, which breaks fullgraph=True.
    from aiter.ops.mha_v4 import mha_v4_mxfp8

    if block_mask is None:
        return mha_v4_mxfp8(q, k, v)
    return mha_v4_mxfp8(q, k, v, block_mask=block_mask)


def _launch_mxfp8_packed(q, k, v, block_mask):
    """Hand-packed MXFP8 sparse path, used when mha_v4 takes no block_mask for
    the deprecated MXFP8 entry point."""
    from aiter.ops.mha_v4 import (
        mha_v4_packed,
        mha_v4_q_multiplier,
        native_fp8_format,
        quantize_fp8,
        quantize_mxfp8_k,
        quantize_mxfp8_q,
    )
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
        _aiter_scale(Scale.E8M0_PER_1X32),
        _aiter_scale(Scale.E8M0_PER_1X32),
        _aiter_scale(Scale.F32_PER_TENSOR),
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
        block_m=256, block_n=_kv_tile(),
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=True,
    )
    q, k, v = to_bshd(q, k, v, contiguous=True)

    if fmt.is_mxfp8 and _has_scale_modes():
        # The deprecated MXFP8 entry point is the one that takes a block_mask;
        # with scale modes present we pack the sparse call by hand instead.
        output = _launch_mxfp8_packed(q, k, v, block_mask)
    else:
        output = _launch(q, k, v, fmt, block_mask=block_mask)

    return restore_sparge_output(from_bshd(output), state), None


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

def _dense_spec(fmt: MhaV4Format) -> Spec:
    return Spec(
        AttentionBackendType[f"AITER_{fmt.name}"],
        impl=functools.partial(mha_v4_dense, fmt=fmt),
        low_precision=fmt.qk is not Fmt.BF16,
        accepts=DENSE_CALLS,
        requires=SYMBOL(_MHA_V4) & GFX950_OR_GFX942,
    )


def _sparge_spec(fmt: MhaV4Format) -> Spec:
    return Spec(
        AttentionBackendType[f"AITER_{fmt.name}_SPARGE"],
        impl=functools.partial(mha_v4_sparge, fmt=fmt),
        is_sparse=True,
        head_balanced=True,
        low_precision=True,
        accepts=SPARGE_CALLS,
        requires=SYMBOL(_MHA_V4) & PARAM(_MHA_V4, "block_mask") & fmt.sparge_on,
    )


SPECS = (
    [_dense_spec(f) for f in FORMATS if f.dense]
    + [_sparge_spec(f) for f in FORMATS if f.sparge_on is not None]
)
