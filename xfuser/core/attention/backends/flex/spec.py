"""flex_block_attn: SSTA and Sparge block masks through FlexAttention."""

from xfuser.core.attention.constraints import NO_VARLEN, SELF_ATTENTION
from xfuser.core.attention.requirements import NEVER, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Sparsity, Spec

_FLEX_BLOCK_ATTN = "flex_block_attn:flex_block_attn_func"

SPECS = [
    Spec(
        AttentionBackendType.FLEX_BLOCK_ATTN,
        impl=Impl("kernel:flex_block"),
        ring=NEVER,
        sparsity=Sparsity.SSTA,
        accepts=SELF_ATTENTION & NO_VARLEN,
        requires=SYMBOL(_FLEX_BLOCK_ATTN),
    ),
    Spec(
        AttentionBackendType.FLEX_BLOCK_SPARGE,
        impl=Impl("kernel:flex_sparge"),
        ring=NEVER,
        sparsity=Sparsity.SPARGE,
        head_balanced=True,
        accepts=SELF_ATTENTION & NO_VARLEN,
        requires=SYMBOL(_FLEX_BLOCK_ATTN),
    ),
]
