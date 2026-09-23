"""AITER FlyDSL: a gfx1201 MHA kernel, with SDPA for shapes it cannot take."""

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN
from xfuser.core.attention.requirements import ARCH, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_FLYDSL = "aiter.ops.flydsl:flydsl_flash_attn_func"

SPECS = [
    Spec(AttentionBackendType.AITER_FLYDSL, impl=Impl("kernel:flydsl"),
         accepts=NO_VARLEN,
         requires=SYMBOL(_FLYDSL) & ARCH("gfx1201")),

    Spec(AttentionBackendType.AITER_FLYDSL_FP8, impl=Impl("kernel:flydsl_fp8"),
         low_precision=True,
         accepts=NO_DROPOUT & NO_VARLEN,
         accepts_prequantized=True,
         prequant_rotate=hadamard.rotate_qk,
         requires=SYMBOL(_FLYDSL) & SYMBOL("aiter.ops.flydsl:flydsl_fp8_quant")
                & ARCH("gfx1201")),
]
