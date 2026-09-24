"""FastH3 VSA-H3: 64-token tiles through torch's own FlexAttention.

"""

from xfuser.core.attention.constraints import NO_DROPOUT, NON_CAUSAL, NO_VARLEN
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_H3 = "xfuser.core.attention.backends.vsa_h3.attention"

SPECS = [
    Spec(AttentionBackendType.FLEX_VSA_H3, impl=Impl("kernel:flex_vsa_h3"),
         sparsity="h3", accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
         requires=SYMBOL(f"{_H3}:flex_h3_vsa_attention")),
]
