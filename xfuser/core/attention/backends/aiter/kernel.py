"""AITER flash attention.

Imported when the backend is selected, so AITER is present by the time this
module loads and its imports can sit at the top where imports belong.

Also the dense path several sparse backends route to when their metadata is
absent; they import `aiter_attention` from here rather than duplicating it.
"""

from aiter import flash_attn_func, flash_attn_varlen_func

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.spec import AttnCall

# aiter-shim cut 2026-09: the AITER_HAS_ROUND_MODE probe (added 2026-03-09)
# guarded whether flash_attn_func accepted how_v3_bf16_cvt at all.
# AITER FAv3 on gfx942 supports three different rounding modes:
#   0 = RTNE (round to nearest even)
#   1 = RTNA (round to nearest away)
#   2 = RTZ  (round to zero)
_BF16_ROUNDING_MODE = 2


def aiter_attention(query, key, value, call: AttnCall):
    """BSHD kernel; packs K/V when the model supplies varlen indices."""
    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        out, lse = flash_attn_func(
            q, k, v,
            dropout_p=call.dropout_p, causal=call.is_causal,
            return_lse=True, return_attn_probs=False,
            how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
        )
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, lse = flash_attn_varlen_func(
            p.q, p.k, p.v, p.cu_seqlens_q, p.cu_seqlens_k,
            max_seqlen_q=p.max_seqlen_q, max_seqlen_k=p.max_seqlen_k,
            softmax_scale=p.head_dim ** -0.5,
            dropout_p=call.dropout_p, causal=call.is_causal,
            return_lse=True, return_attn_probs=False,
            how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
        )
        out = p.unflatten(out)

    return from_bshd(out), lse
