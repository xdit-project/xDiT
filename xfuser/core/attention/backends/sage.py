"""SageAttention (the standalone package, not AITER's port)."""

from xfuser.core.attention.requirements import PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec


def sage_attention(query, key, value, call: AttnCall):
    from sageattention import sageattn

    return sageattn(query, key, value, is_causal=call.is_causal, return_lse=True)


SPECS = [
    Spec(AttentionBackendType.SAGE, impl=sage_attention, returns_lse=True,
         low_precision=True, requires=PLATFORM("cuda") & SYMBOL("sageattention:sageattn")),
]
