"""Transformer Engine FP8 attention."""

import functools

from transformer_engine.common import recipe
from transformer_engine.pytorch import DotProductAttention, fp8_autocast

import torch

from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.spec import AttnCall


@functools.lru_cache(maxsize=32)
def _dot_product_attention(num_heads, head_dim, attn_mask_type, device_index):
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
    return recipe.DelayedScaling(fp8_dpa=True)


def nvte_fp8(query, key, value, call: AttnCall):
    q, k, v = to_bshd(query, key, value, contiguous=True)
    batch, seq_len, num_heads, head_dim = q.shape
    attn_mask_type = "causal" if call.is_causal else "no_mask"

    dpa = _dot_product_attention(
        num_heads, head_dim, attn_mask_type, q.device.index or 0
    )
    with fp8_autocast(enabled=True, fp8_recipe=_fp8_recipe()):
        out = dpa(q, k, v, attn_mask_type=attn_mask_type)

    return from_bshd(out.view(batch, seq_len, num_heads, head_dim)), None


