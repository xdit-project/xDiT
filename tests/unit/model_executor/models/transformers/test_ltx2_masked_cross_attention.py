"""LTX-2 text cross attention hands the attention backend a key-padding mask
together with the varlen packing derived from it. The SDPA backends serve
that through the mask; they used to refuse it outright."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta
from xfuser.model_executor.models.transformers.transformer_ltx2 import (
    xFuserLTX2VideoTransformer3DWrapper,
)


@pytest.fixture
def text_cross_attention():
    torch.manual_seed(0)
    model = xFuserLTX2VideoTransformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        num_layers=1,
        caption_channels=16,
    ).eval()
    return model.transformer_blocks[0].attn2


@pytest.mark.parametrize("backend", [AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH])
@torch.no_grad()
def test_text_cross_attention_on_sdpa_attends_only_to_valid_text(text_cross_attention, backend):
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 5, 16)
    text = torch.randn(2, 6, 16)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])

    runtime_state = SimpleNamespace(attention_backend=backend)
    with mock.patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime_state):
        out = text_cross_attention(
            hidden_states, encoder_hidden_states=text, attention_mask=make_attn_mask_with_meta(valid)
        )
        for b in range(2):
            alone = text_cross_attention(
                hidden_states[b : b + 1], encoder_hidden_states=text[b : b + 1, valid[b].bool()]
            )
            torch.testing.assert_close(out[b : b + 1], alone)
