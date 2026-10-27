from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.transformers import transformer_ideogram4
from xfuser.model_executor.models.transformers.transformer_ideogram4 import _refuse_padding_under_low_precision_backend


@pytest.mark.parametrize("backend", [AttentionBackendType.AITER_FP8, AttentionBackendType.SAGE])
def test_padded_shard_is_refused_under_a_quantising_backend(backend):
    with pytest.raises(NotImplementedError, match=backend.name):
        _refuse_padding_under_low_precision_backend(1, backend)


@pytest.mark.parametrize(
    ("pad_amount", "backend"),
    [(1, AttentionBackendType.SDPA_EFFICIENT), (0, AttentionBackendType.AITER_FP8)],
    ids=["padded-full-precision", "unpadded-quantising"],
)
def test_other_combinations_are_not_refused(pad_amount, backend):
    _refuse_padding_under_low_precision_backend(pad_amount, backend)


_IMAGE_TOKENS = 9


@pytest.mark.parametrize(("text_tokens", "valid_kv_len"), [(4, 13), (5, None)], ids=["padded", "divisible"])
def test_padding_is_published_to_the_usp_calls_through_the_shared_dict(monkeypatch, text_tokens, valid_kv_len):
    """The wrapper pads, publishes the real length, and every USP call receives the shared dict."""
    pytest.importorskip("diffusers.models.transformers.transformer_ideogram4")
    from diffusers.models.transformers.transformer_ideogram4 import (
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        Ideogram4Transformer2DModel,
    )

    calls = []

    def usp_recorder(query, key, value, **kwargs):
        shared = kwargs.get("attention_kwargs")
        calls.append((shared, dict(shared or {}), key.shape[2]))
        return query

    runtime = SimpleNamespace(attention_backend=AttentionBackendType.SDPA_EFFICIENT)
    monkeypatch.setattr(transformer_ideogram4, "USP", usp_recorder)
    monkeypatch.setattr(transformer_ideogram4, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.transformers_utils.get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim=dim)),
    )
    config = dict(
        in_channels=8, num_layers=2, attention_head_dim=8, num_attention_heads=4, intermediate_size=64,
        adaln_dim=16, llm_features_dim=24, rope_theta=10_000, mrope_section=(2, 1, 1),
    )  # fmt: skip
    wrapper = transformer_ideogram4.get_ideogram4_transformer_wrapper_class().from_config(
        Ideogram4Transformer2DModel(**config).config
    )
    wrapper._sp_rank, wrapper._sp_size = 1, 2

    length = text_tokens + _IMAGE_TOKENS
    indicator = torch.full((1, length), OUTPUT_IMAGE_INDICATOR)
    indicator[:, :text_tokens] = LLM_TOKEN_INDICATOR
    wrapper(
        hidden_states=torch.randn(1, length, 8),
        timestep=torch.tensor([0.3]),
        encoder_hidden_states=torch.randn(1, length, 24),
        position_ids=torch.zeros(1, length, 3, dtype=torch.long),
        segment_ids=torch.ones(1, length, dtype=torch.long),
        indicator=indicator,
        return_dict=False,
    )

    assert len(calls) == config["num_layers"]
    for shared, published, local_length in calls:
        assert shared is wrapper._usp_attention_kwargs
        assert published == {"valid_kv_len": valid_kv_len}
        assert local_length == (length + 1) // 2
