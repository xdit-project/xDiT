"""AITER VSA: Jenga block-sparse self-attention over a spatial layout."""

from xfuser.core.attention.backends.aiter.spec import AITER_ARCH
from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN, NON_CAUSAL
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_VSA = "aiter.ops.jenga_sparse_attention:vsa_sparse_attention"

SPECS = [
    Spec(
        AttentionBackendType.AITER_VSA,
        impl=Impl("kernel:vsa_attention"),
        sparsity="vsa",
        low_precision=True,
        # Causal and dropout are refused outright; the dense-routing cases are
        # handled inside the kernel because they depend on the metadata, not
        # the tensors.
        accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        requires=SYMBOL(_VSA) & AITER_ARCH,
    ),
]
