"""SageAttention (the standalone package, not AITER's port)."""

from xfuser.core.attention.spec import AttnCall


def sage_attention(query, key, value, call: AttnCall):
    from sageattention import sageattn

    return sageattn(query, key, value, is_causal=call.is_causal, return_lse=True)


