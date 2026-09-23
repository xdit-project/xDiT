"""FlexAttention-backed sparse kernels: block masks and VSA-H3 tiling."""

from xfuser.core.attention.constraints import (
    NO_DROPOUT,
    NO_VARLEN,
    NON_CAUSAL,
    SELF_ATTENTION,
)
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_FLEX = "flex_block_attn:flex_block_attn_func"

SPECS = [
    Spec(AttentionBackendType.FLEX_BLOCK_ATTN, impl=Impl("kernel:flex_block"),
         sparsity="ssta", accepts=SELF_ATTENTION & NO_VARLEN,
         requires=SYMBOL(_FLEX)),

    Spec(AttentionBackendType.FLEX_BLOCK_SPARGE, impl=Impl("kernel:flex_sparge"),
         sparsity="sparge", head_balanced=True,
         accepts=SELF_ATTENTION & NO_VARLEN, requires=SYMBOL(_FLEX)),

    # Flex-attention based but not flex_block_attn: this one goes through
    # torch's own FlexAttention, so it has no third-party requirement.
    Spec(AttentionBackendType.FLEX_VSA_H3, impl=Impl("kernel:flex_vsa_h3"),
         sparsity="h3", accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
         requires=SYMBOL("xfuser.core.vsa_h3_attention:flex_h3_vsa_attention")),
]
