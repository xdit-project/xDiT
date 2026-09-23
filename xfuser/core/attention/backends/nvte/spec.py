"""Transformer Engine FP8 attention."""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(AttentionBackendType.NVTE_FP8, impl=Impl("kernel:nvte_fp8"),
         low_precision=True, accepts=NO_VARLEN,
         # Platform first: All reports the first failure, and "requires cuda,
         # found rocm" is more use to an AMD user than "not importable".
         requires=PLATFORM("cuda")
                & SYMBOL("transformer_engine.pytorch:DotProductAttention")
                & SYMBOL("transformer_engine.common:recipe")),
]
