"""Lumina2 GQA keeps padded keys masked through USP backend selection."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention_processor import Attention

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import usp
from xfuser.model_executor.models.transformers.transformer_lumina2 import xFuserLumina2AttnProcessor2_0


@pytest.mark.parametrize(
    "backend", [AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH, AttentionBackendType.AITER_FLYDSL_FP8]
)
@torch.no_grad()
def test_gqa_mask_matches_removing_padded_keys(monkeypatch, backend):
    state = SimpleNamespace(
        attention_backend=backend, fp8_comms=None, runtime_config=SimpleNamespace(use_sparseattn_head_balance=False)
    )
    monkeypatch.setattr(usp, "get_runtime_state", lambda: state)
    for name in ("get_ulysses_parallel_world_size", "get_ring_parallel_world_size", "get_sequence_parallel_world_size"):
        monkeypatch.setattr(usp, name, lambda: 1)
    torch.manual_seed(4)
    attn = Attention(
        query_dim=16, cross_attention_dim=16, heads=4, kv_heads=2, dim_head=4, processor=xFuserLumina2AttnProcessor2_0()
    ).eval()
    hidden_states = torch.randn(1, 3, 16)
    text = torch.randn(1, 6, 16)
    actual = attn(hidden_states, encoder_hidden_states=text, attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]))
    state.attention_backend = AttentionBackendType.SDPA
    expected = attn(hidden_states, encoder_hidden_states=text[:, :4])
    torch.testing.assert_close(actual, expected)
