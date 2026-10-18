"""Prism block-sparse attention: top-k / top-p over 3D token blocks, in Triton."""

from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN, NON_CAUSAL
from xfuser.core.attention.requirements import NEVER, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Sparsity, Spec

SPECS = [
    # The kernel returns no LSE, so it cannot join a ring.
    Spec(
        AttentionBackendType.TRITON_BSA,
        impl=Impl("kernel:triton_bsa"),
        ring=NEVER,
        sparsity=Sparsity.BSA,
        accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        requires=SYMBOL("triton:jit"),
    ),
]
