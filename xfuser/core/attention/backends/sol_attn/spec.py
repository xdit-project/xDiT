"""Sol-Attn, the optional Sana on-the-fly sparse attention kernel."""

from xfuser.core.attention.constraints import (
    BF16,
    HEAD_DIM,
    NO_DROPOUT,
    NO_VARLEN,
    NON_CAUSAL,
    SAME_QKV,
)
from xfuser.core.attention.requirements import CUDA_CAPABILITY, NEVER, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_SOL_ATTN = "sol_attn:sol_attn"

# Calls this kernel cannot serve, including cross-attention, other dtypes, and
# other head widths, fall through to SDPA. The kernel returns no log-sumexp,
# so ring attention cannot merge it.
SPECS = [
    Spec(
        AttentionBackendType.SOL_ATTN,
        impl=Impl("kernel:sol_attention"),
        ring=NEVER,
        accepts=SAME_QKV & HEAD_DIM(128) & BF16 & NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        fallback=AttentionBackendType.SDPA,
        requires=PLATFORM("cuda") & CUDA_CAPABILITY((8, 0)) & SYMBOL(_SOL_ATTN),
    ),
]
