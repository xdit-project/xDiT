"""AITER MHA v4: a quantisation-format table crossed with dense/sparge.

Eighteen enum members are one kernel, one sparsity strategy, and a ten-row
table of (Q/K format, V format). Adding a format is a row plus two enum
members; everything else is derived.

AITER_FP8's *dense* path is not here -- it predates MHA v4 and still carries a
Hadamard-rotation and varlen fallback, so it lives with those helpers. Its
sparge variant is pure table and is generated below.
"""

import functools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from xfuser.core.attention.requirements import ALWAYS, ARCH, PARAM, SYMBOL, Requirement
from xfuser.core.attention.constraints import (
    HEAD_DIM,
    MHA_ONLY,
    NON_CAUSAL,
    NO_DROPOUT,
    NO_VARLEN,
    SELF_ATTENTION,
    TRAILING_PAD_ONLY,
)
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

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
    MhaV4Format("FP8",       Fmt.NATIVE_FP8,  Fmt.NATIVE_FP8,  GFX950_OR_GFX942,
                dense=False),
]


# Dense serves a declared trailing pad by shortening K/V; sparge cannot, since
# its sorted-sparse launch needs the key length padded to its KV tile, which is
# the alignment such a slice removes.
DENSE_CALLS = NO_DROPOUT & NON_CAUSAL & TRAILING_PAD_ONLY & HEAD_DIM(128)
SPARGE_CALLS = (
    NO_DROPOUT & NON_CAUSAL & NO_VARLEN & HEAD_DIM(128) & MHA_ONLY & SELF_ATTENTION
)


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

def _scale_modes(fmt: MhaV4Format) -> Requirement:
    if fmt.qk_scale is None:
        return ALWAYS
    return PARAM(_MHA_V4, "q_scale_mode")


def _dense_spec(fmt: MhaV4Format) -> Spec:
    return Spec(
        AttentionBackendType[f"AITER_{fmt.name}"],
        impl=Impl("kernel:mha_v4_dense", {"fmt": fmt}),
        low_precision=fmt.qk is not Fmt.BF16,
        accepts=DENSE_CALLS,
        requires=SYMBOL(_MHA_V4) & GFX950_OR_GFX942 & _scale_modes(fmt),
    )


def _sparge_spec(fmt: MhaV4Format) -> Spec:
    return Spec(
        AttentionBackendType[f"AITER_{fmt.name}_SPARGE"],
        impl=Impl("kernel:mha_v4_sparge", {"fmt": fmt}),
        sparsity="sparge",
        head_balanced=True,
        low_precision=True,
        accepts=SPARGE_CALLS,
        requires=SYMBOL(_MHA_V4) & PARAM(_MHA_V4, "block_mask")
               & _scale_modes(fmt) & fmt.sparge_on,
    )


SPECS = (
    [_dense_spec(f) for f in FORMATS if f.dense]
    + [_sparge_spec(f) for f in FORMATS if f.sparge_on is not None]
)
