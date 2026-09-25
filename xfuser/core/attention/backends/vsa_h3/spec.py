"""FastH3 VSA-H3: 64-token tiles, on either of two kernels.

Both select the same key tiles and differ only in how they read them, so they
share everything around the call. The Triton kernel reads packed rows through
the tile map; FlexAttention needs the padded tile buffers built for it.
"""

from xfuser.core.attention.constraints import NO_DROPOUT, NON_CAUSAL, NO_VARLEN
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_H3 = "xfuser.core.attention.backends.vsa_h3.attention"

# Neither produces an LSE, so neither can join a ring.
_CALLS = NON_CAUSAL & NO_DROPOUT & NO_VARLEN

SPECS = [
    Spec(AttentionBackendType.FLEX_VSA_H3, impl=Impl("kernel:flex_vsa_h3"),
         sparsity="h3", accepts=_CALLS,
         requires=SYMBOL(f"{_H3}:h3_vsa_attention")),

    # Triton is not in `requires`: where the kernel cannot run this falls back
    # to FlexAttention rather than refusing, so the backend stays selectable
    # and says so once at the first call.
    Spec(AttentionBackendType.TRITON_VSA_H3, impl=Impl("kernel:triton_vsa_h3"),
         sparsity="h3", accepts=_CALLS,
         requires=SYMBOL(f"{_H3}:h3_vsa_attention")),
]
