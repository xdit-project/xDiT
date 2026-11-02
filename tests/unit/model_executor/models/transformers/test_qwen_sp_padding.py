"""Qwen-Image declares a replicated prompt, and refuses FP8 communication where the prompt would join attention as joint tokens."""

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


def _declared_kwargs(model, monkeypatch, *, sp_size, image_tokens, text_tokens):
    seen = {}

    def block(*args, **kwargs):
        seen.update(model._self_attn_kwargs)
        return kwargs["encoder_hidden_states"], kwargs["hidden_states"]

    model.transformer_blocks[0].forward = block
    monkeypatch.setattr(qwen, "gather_and_unpad", lambda x, pad_amount, dim: x)
    _forward(model, monkeypatch, sp_size=sp_size, image_tokens=image_tokens, text_tokens=text_tokens, fp8_comms=None)
    return seen


def test_a_prompt_that_does_not_split_evenly_is_declared_replicated_with_its_length(monkeypatch):
    seen = _declared_kwargs(_model(), monkeypatch, sp_size=4, image_tokens=8, text_tokens=7)
    assert seen["replicated_prompt_tokens"] == 7


def test_a_sharded_prompt_is_declared_with_no_replicated_tokens(monkeypatch):
    seen = _declared_kwargs(_model(), monkeypatch, sp_size=4, image_tokens=8, text_tokens=8)
    assert seen["replicated_prompt_tokens"] == 0
