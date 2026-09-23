"""SageAttention (the standalone package, not AITER's port)."""

from sageattention import sageattn

from xfuser.core.attention.spec import AttnCall


def sage_attention(query, key, value, call: AttnCall):
    return sageattn(query, key, value, is_causal=call.is_causal, return_lse=True)


