"""Qwen-Image refuses FP8 communication where the prompt would have to join attention as joint tokens."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.transformers import transformer_qwen as qwen


def _model():
    return qwen.xFuserQwenImageTransformerWrapper(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )


def _forward(model, monkeypatch, *, sp_size, image_tokens, text_tokens, fp8_comms):
    monkeypatch.setattr(qwen, "get_sequence_parallel_world_size", lambda: sp_size)
    monkeypatch.setattr(qwen, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(qwen, "get_runtime_state", lambda: SimpleNamespace(fp8_comms=fp8_comms))
    model(
        torch.ones(1, image_tokens, 16),
        encoder_hidden_states=torch.ones(1, text_tokens, 16),
        timestep=torch.tensor([0.5]),
        img_shapes=[[(1, 1, image_tokens)]],
    )


def test_an_indivisible_prompt_with_fp8_comms_names_the_prompt_length_and_the_degree(monkeypatch):
    with pytest.raises(ValueError, match=r"--use_fp8_comms needs the prompt length \(7 tokens\).*\(4\)"):
        _forward(_model(), monkeypatch, sp_size=4, image_tokens=8, text_tokens=7, fp8_comms=object())


def test_image_padding_spanning_several_shards_with_fp8_comms_is_refused(monkeypatch):
    with pytest.raises(ValueError, match=r"--use_fp8_comms needs an image whose padding fits in one shard \(got 5"):
        _forward(_model(), monkeypatch, sp_size=8, image_tokens=5, text_tokens=8, fp8_comms=object())
