"""LTX-2 text cross attention runs on cuDNN, the backend xDiT picks by default
on NVIDIA when FlashAttention is not installed.

The text keys are padded, so the processor hands the backend a key-padding
mask together with the varlen packing derived from it; cuDNN refused that
with "CUDNN does not support varlen packed keys" on the first step.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta
from xfuser.model_executor.models.transformers.transformer_ltx2 import (
    xFuserLTX2VideoTransformer3DWrapper,
)

pytestmark = pytest.mark.nvidia


@pytest.fixture
def text_cross_attention():
    torch.manual_seed(0)
    model = xFuserLTX2VideoTransformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=64,
        cross_attention_dim=128,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=128,
        num_layers=1,
        caption_channels=128,
    )
    return model.transformer_blocks[0].attn2.to("cuda", torch.bfloat16).eval()


@torch.no_grad()
def test_text_cross_attention_on_cudnn_attends_only_to_valid_text(text_cross_attention):
    generator = torch.Generator(device="cuda").manual_seed(1)
    hidden_states = torch.randn(2, 40, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    text = torch.randn(2, 24, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    valid = torch.zeros(2, 24, dtype=torch.long, device="cuda")
    valid[0, :9] = 1
    valid[1, :21] = 1

    runtime_state = SimpleNamespace(attention_backend=AttentionBackendType.CUDNN)
    with mock.patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime_state):
        out = text_cross_attention(
            hidden_states, encoder_hidden_states=text, attention_mask=make_attn_mask_with_meta(valid)
        )
        for b in range(2):
            alone = text_cross_attention(
                hidden_states[b : b + 1], encoder_hidden_states=text[b : b + 1, valid[b].bool()]
            )
            torch.testing.assert_close(out[b : b + 1], alone, atol=2e-2, rtol=2e-2)
