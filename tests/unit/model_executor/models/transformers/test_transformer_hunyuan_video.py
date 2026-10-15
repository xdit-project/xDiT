from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.transformers import transformer_hunyuan_video as hunyuan
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import (
    _refuse_padding_under_low_precision_backend,
)


@pytest.mark.parametrize(
    "backend", [AttentionBackendType.AITER_VSA, AttentionBackendType.AITER_FP8, AttentionBackendType.SAGE]
)
def test_padded_shard_is_refused_under_vsa_and_quantising_backends(backend):
    with pytest.raises(NotImplementedError, match=backend.name):
        _refuse_padding_under_low_precision_backend(2, backend)


@pytest.mark.parametrize(
    ("pad_amount", "backend"),
    [
        (2, AttentionBackendType.SDPA_EFFICIENT),
        (0, AttentionBackendType.AITER_VSA),
        (0, AttentionBackendType.AITER_FP8),
    ],
    ids=["padded-dense", "unpadded-vsa", "unpadded-quantising"],
)
def test_other_combinations_are_not_refused(pad_amount, backend):
    _refuse_padding_under_low_precision_backend(pad_amount, backend)


@pytest.mark.parametrize(
    ("frames", "valid_kv_len", "split_text"), [(3, 27, False), (2, None, False)], ids=["padded", "divisible"]
)
def test_padding_is_published_to_the_usp_calls_through_the_shared_dict(monkeypatch, frames, valid_kv_len, split_text):
    """The wrapper pads the video tokens, publishes their real count and hands USP the shared dict."""
    calls = []

    def usp_recorder(query, key, value, **kwargs):
        shared = kwargs.get("attention_kwargs")
        calls.append((shared, dict(shared or {})))
        joint_query = kwargs.get("joint_query")
        return query if joint_query is None else torch.cat([query, joint_query], dim=2)

    runtime = SimpleNamespace(
        attention_backend=AttentionBackendType.SDPA_EFFICIENT,
        increment_step_counter=lambda: None,
        split_text_embed_in_sp=False,
        max_condition_sequence_length=5,
    )
    monkeypatch.setattr(hunyuan, "USP", usp_recorder)
    monkeypatch.setattr(hunyuan, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_rank", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(hunyuan, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda x, dim: x))
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.transformers_utils.get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim=dim)),
    )
    config = dict(
        in_channels=4, out_channels=4, num_attention_heads=6, attention_head_dim=8, num_layers=1,
        num_single_layers=1, num_refiner_layers=1, patch_size=1, patch_size_t=1, text_embed_dim=16,
        pooled_projection_dim=8, rope_axes_dim=(2, 2, 4),
    )  # fmt: skip
    wrapper = hunyuan.xFuserHunyuanVideoTransformer3DWrapper(**config).install_xdit_attention_processors()

    mask = torch.ones(1, 7, dtype=torch.int64)
    mask[:, 5:] = 0  # five text tokens remain, so the text never divides two ranks
    wrapper(
        hidden_states=torch.randn(1, 4, frames, 3, 3),
        timestep=torch.tensor([700]),
        encoder_hidden_states=torch.randn(1, 7, 16),
        encoder_attention_mask=mask,
        pooled_projections=torch.randn(1, 8),
        guidance=torch.tensor([6000.0]),
        return_dict=False,
    )

    assert len(calls) == config["num_layers"] + config["num_single_layers"]
    assert runtime.split_text_embed_in_sp is split_text
    for shared, published in calls:
        assert shared is wrapper._usp_attention_kwargs
        assert published == {"valid_kv_len": valid_kv_len}
