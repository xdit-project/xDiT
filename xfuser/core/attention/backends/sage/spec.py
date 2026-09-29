"""SageAttention (the standalone package, not AITER's port)."""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import ALWAYS, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(
        AttentionBackendType.SAGE,
        impl=Impl("kernel:sage_attention"),
        ring=ALWAYS,
        low_precision=True,
        accepts=NO_VARLEN,
        requires=PLATFORM("cuda") & SYMBOL("sageattention:sageattn"),
    ),
]
