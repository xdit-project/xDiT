"""AITER VSA: Jenga block-sparse self-attention over a spatial layout."""

from xfuser.core.attention.backends.aiter.spec import AITER_ARCH
from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN, NON_CAUSAL
from xfuser.core.attention.requirements import NEVER, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Sparsity, Spec

_VSA = "aiter.ops.jenga_sparse_attention:vsa_sparse_attention"

SPECS = [
    Spec(
        AttentionBackendType.AITER_VSA,
        impl=Impl("kernel:vsa_attention"),
        ring=NEVER,
        sparsity=Sparsity.VSA,
        low_precision=True,
        accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        requires=SYMBOL(_VSA) & AITER_ARCH,
    ),
]
