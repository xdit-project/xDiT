"""Qwen-Image keeps sequence-parallel padding out of attention."""

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
        _forward(_model(), monkeypatch, sp_size=4, image_tokens=1, text_tokens=7, fp8_comms=object())


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


def test_an_indivisible_prompt_is_sharded_with_trailing_padding(monkeypatch):
    seen = _declared_kwargs(_model(), monkeypatch, sp_size=4, image_tokens=8, text_tokens=7)
    assert seen["replicated_prompt_tokens"] == 0
    assert seen["sharded_padding"] is True
    assert seen["valid_kv_len"] == 15


def test_a_sharded_prompt_is_declared_with_no_replicated_tokens(monkeypatch):
    seen = _declared_kwargs(_model(), monkeypatch, sp_size=4, image_tokens=8, text_tokens=8)
    assert seen["replicated_prompt_tokens"] == 0
    assert seen["sharded_padding"] is False


def test_padded_streams_are_reordered_behind_real_tokens_without_replication(monkeypatch):
    seen = {}

    def usp(query, key, value, **kwargs):
        seen.update(query=query, key=key, value=value, kwargs=kwargs)
        return query

    monkeypatch.setattr(qwen, "USP", usp)
    attn = SimpleNamespace(
        heads=2,
        to_q=torch.nn.Identity(),
        to_k=torch.nn.Identity(),
        to_v=torch.nn.Identity(),
        add_q_proj=torch.nn.Identity(),
        add_k_proj=torch.nn.Identity(),
        add_v_proj=torch.nn.Identity(),
        norm_q=None,
        norm_k=None,
        norm_added_q=None,
        norm_added_k=None,
        to_out=torch.nn.ModuleList([torch.nn.Identity()]),
        to_add_out=torch.nn.Identity(),
    )
    processor = qwen.xFuserQwenDoubleStreamAttnProcessor(
        {
            "valid_kv_len": 6,
            "replicated_prompt_tokens": 0,
            "sharded_padding": True,
            "local_image_pad_rows": 1,
            "local_text_pad_rows": 1,
        }
    )

    input_image = torch.randn(1, 4, 8)
    input_text = torch.randn(1, 4, 8)
    image, text = processor(
        attn,
        hidden_states=input_image,
        encoder_hidden_states=input_text,
    )

    assert seen["query"].shape[2] == 8
    assert seen["key"].shape[2] == 8
    assert seen["value"].shape[2] == 8
    assert seen["kwargs"].get("joint_query") is None
    assert seen["kwargs"].get("joint_key") is None
    torch.testing.assert_close(image, input_image)
    torch.testing.assert_close(text, input_text)
