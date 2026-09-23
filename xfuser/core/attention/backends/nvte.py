"""Transformer Engine FP8 attention."""

import functools

import torch

from xfuser.core.attention.layout import from_bshd, to_bshd
from xfuser.core.attention.requirements import PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec


@functools.lru_cache(maxsize=32)
def _dot_product_attention(num_heads, head_dim, attn_mask_type, device_index):
    from transformer_engine.pytorch import DotProductAttention

    return (
        DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            qkv_format="bshd",
            attn_mask_type=attn_mask_type,
            attention_dropout=0.0,
        )
        .to(torch.device("cuda", device_index))
        .eval()
    )


@functools.lru_cache(maxsize=1)
def _fp8_recipe():
    from transformer_engine.common import recipe

    return recipe.DelayedScaling(fp8_dpa=True)


def nvte_fp8(query, key, value, call: AttnCall):
    from transformer_engine.pytorch import fp8_autocast

    q, k, v = to_bshd(query, key, value, contiguous=True)
    batch, seq_len, num_heads, head_dim = q.shape
    attn_mask_type = "causal" if call.is_causal else "no_mask"

    dpa = _dot_product_attention(
        num_heads, head_dim, attn_mask_type, q.device.index or 0
    )
    with fp8_autocast(enabled=True, fp8_recipe=_fp8_recipe()):
        out = dpa(q, k, v, attn_mask_type=attn_mask_type)

    return from_bshd(out.view(batch, seq_len, num_heads, head_dim)), None


SPECS = [
    Spec(AttentionBackendType.NVTE_FP8, impl=nvte_fp8, low_precision=True,
         # Platform first: All reports the first failure, and "requires cuda,
         # found rocm" is more use to an AMD user than "not importable".
         requires=PLATFORM("cuda")
                & SYMBOL("transformer_engine.pytorch:DotProductAttention")
                & SYMBOL("transformer_engine.common:recipe")),
]
