"""The DiTFastAttn helpers that set and reset the attention layer state."""

import torch.nn as nn
from diffusers.models.attention_processor import Attention

from xfuser.model_executor.fast_attention_calibration import reset_fast_attn_layers, set_fast_attn_step
from xfuser.model_executor.layers.attention_processor import xFuserAttentionBaseWrapper
from xfuser.model_executor.layers.base_layer import xFuserLayerBaseWrapper


class _AttentionWrapper(xFuserAttentionBaseWrapper):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _OtherWrapper(xFuserLayerBaseWrapper):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _transformer():
    attention = _AttentionWrapper(Attention(query_dim=8, heads=1, dim_head=8))
    attention.stepi = 5
    attention.cached_residual = "residual"
    attention.cached_output = "output"
    # The wrapper forwards attribute lookups, so hasattr(other, "stepi") is True.
    wrapped = nn.Linear(8, 8)
    wrapped.stepi = 7
    other = _OtherWrapper(wrapped)

    block = nn.Module()
    block.attn1 = attention
    block.ff = other
    transformer = nn.Module()
    transformer.transformer_blocks = nn.ModuleList([block])
    return transformer, attention, other


def test_set_fast_attn_step_keeps_the_caches():
    transformer, attention, other = _transformer()

    set_fast_attn_step(transformer, 3)

    assert attention.stepi == 3
    assert attention.cached_residual == "residual"
    assert attention.cached_output == "output"
    assert other.stepi == 7


def test_reset_fast_attn_layers_removes_the_caches():
    transformer, attention, other = _transformer()

    reset_fast_attn_layers(transformer)

    assert attention.stepi == 0
    assert attention.cached_residual is None
    assert attention.cached_output is None
    assert other.stepi == 7
    assert "cached_residual" not in vars(other)
