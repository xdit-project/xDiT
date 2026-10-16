"""Hunyuan's two streams retain their outputs when SP padding is removed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.models.transformers import transformer_hunyuan_video15 as hunyuan
from xfuser.model_executor.models.transformers import transformers_utils
from xfuser.core.attention.spec import AttentionBackendType


def _attention():
    # Identity projections expose the token layout at the collective boundary.
    identity = torch.nn.Identity()
    return SimpleNamespace(
        heads=2,
        to_q=identity,
        to_k=identity,
        to_v=identity,
        norm_q=identity,
        norm_k=identity,
        add_q_proj=identity,
        add_k_proj=identity,
        add_v_proj=identity,
        norm_added_q=identity,
        norm_added_k=identity,
        to_out=[identity, identity],
        to_add_out=identity,
    )


def _heads(x):
    return x.unflatten(-1, (2, -1)).transpose(1, 2)


def _tiny_model(**kwargs):
    return hunyuan.xFuserHunyuanVideo15Transformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=2,
        num_refiner_layers=1,
        text_embed_dim=8,
        text_embed_2_dim=8,
        image_embed_dim=8,
        rope_axes_dim=(2, 2, 4),
        **kwargs,
    ).eval()


@pytest.mark.parametrize("split_text", [True, False], ids=["sharded-prompt", "replicated-prompt"])
def test_dense_attention_excludes_image_padding_and_restores_the_two_streams(monkeypatch, split_text):
    generator = torch.Generator().manual_seed(13)
    image = torch.randn(1, 2, 8, generator=generator)
    text = torch.randn(1, 2, 8, generator=generator)
    # A block may write nonzero pad queries; none may contribute as keys.
    image_pad = torch.full((1, 1, 8), 100.0)
    padded_image = torch.cat([image, image_pad], dim=1)
    shared_kwargs = {"valid_kv_len": 4 if split_text else 2}
    calls = []

    def collective_attention(query, key, value, **kwargs):
        calls.append((query, kwargs))
        assert kwargs["attention_kwargs"] is shared_kwargs
        end = shared_kwargs["valid_kv_len"]
        key, value = key[:, :, :end], value[:, :, :end]
        if not split_text:
            query = torch.cat([query, kwargs["joint_query"]], dim=2)
            key = torch.cat([key, kwargs["joint_key"]], dim=2)
            value = torch.cat([value, kwargs["joint_value"]], dim=2)
        return F.scaled_dot_product_attention(query, key, value)

    monkeypatch.setattr(hunyuan, "USP", collective_attention)
    monkeypatch.setattr(hunyuan, "get_runtime_state", lambda: SimpleNamespace(split_text_embed_in_sp=split_text))
    processor = hunyuan.xFuserHunyuanVideo15AttnProcessor(shared_kwargs)
    actual_image, actual_text = processor(_attention(), padded_image, text)

    real = _heads(torch.cat([image, text], dim=1))
    expected = F.scaled_dot_product_attention(real, real, real).transpose(1, 2).flatten(2)
    torch.testing.assert_close(actual_image[:, : image.shape[1]], expected[:, : image.shape[1]])
    torch.testing.assert_close(actual_text, expected[:, image.shape[1] :])
    query, kwargs = calls[0]
    if split_text:
        torch.testing.assert_close(query[:, :, : text.shape[1]], _heads(text))
        torch.testing.assert_close(query[:, :, -padded_image.shape[1] :], _heads(padded_image))
        assert "joint_query" not in kwargs
    else:
        torch.testing.assert_close(query, _heads(padded_image))
        torch.testing.assert_close(kwargs["joint_query"], _heads(text))


@pytest.mark.parametrize(
    "backend,text_first",
    [(AttentionBackendType.FLEX_BLOCK_ATTN, False), (AttentionBackendType.SDPA, True)],
    ids=["sparse-step", "dense-hybrid-step"],
)
def test_sparse_checkpoint_uses_the_layout_of_the_current_step(monkeypatch, backend, text_first):
    image, text = torch.randn(1, 3, 8), torch.randn(1, 2, 8)

    def collective_attention(query, key, value, **kwargs):
        streams = [text, image] if text_first else [image, text]
        torch.testing.assert_close(query, _heads(torch.cat(streams, dim=1)))
        assert "joint_query" not in kwargs
        return value

    monkeypatch.setattr(hunyuan, "USP", collective_attention)
    monkeypatch.setattr(
        hunyuan, "get_runtime_state", lambda: SimpleNamespace(split_text_embed_in_sp=True, attention_backend=backend)
    )
    processor = hunyuan.xFuserHunyuanVideo15AttnProcessor({"attn_sparse_type": "ssta"})
    actual_image, actual_text = processor(_attention(), image, text)
    torch.testing.assert_close(actual_image, image)
    torch.testing.assert_close(actual_text, text)


@pytest.mark.parametrize("image_rows,text_rows", [(2, 1), (0, 2), (5, 0), (0, 0)])
def test_every_block_zeroes_its_own_stream_padding(monkeypatch, image_rows, text_rows):
    monkeypatch.setattr(hunyuan, "get_runtime_state", lambda: SimpleNamespace(split_text_embed_in_sp=True))
    monkeypatch.setattr(
        hunyuan, "USP", lambda query, key, value, **kwargs: F.scaled_dot_product_attention(query, key, value)
    )
    model = _tiny_model()
    model._sp_pad_image["local_rows"] = image_rows
    model._sp_pad_text["local_rows"] = text_rows
    image, text, temb = torch.randn(1, 5, 16), torch.randn(1, 4, 16), torch.randn(1, 16)

    with torch.no_grad():
        for block in model.transformer_blocks:
            # Direct forward computes the same block without invoking its hook.
            reference = block.forward(image, text, temb)
            actual = block(image, text, temb)
            for output, expected, rows in zip(actual, reference, (image_rows, text_rows)):
                end = output.shape[1] - rows
                torch.testing.assert_close(output[:, :end], expected[:, :end])
                if rows:
                    assert output[:, end:].count_nonzero() == 0


def test_dense_hybrid_steps_remove_sparse_text_padding_and_use_joint_prompt(monkeypatch):
    state = SimpleNamespace(split_text_embed_in_sp=True, attention_backend=AttentionBackendType.FLEX_BLOCK_ATTN)
    monkeypatch.setattr(hunyuan, "get_runtime_state", lambda: state)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_rank", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim=dim)),
    )
    calls = []

    def collective_attention(query, key, value, **kwargs):
        calls.append((query.shape[2], kwargs.get("joint_query"), dict(kwargs["attention_kwargs"])))
        if kwargs.get("joint_query") is not None:
            return torch.cat([value, kwargs["joint_value"]], dim=2)
        return value

    monkeypatch.setattr(hunyuan, "USP", collective_attention)
    model = _tiny_model(attention_kwargs={"attn_sparse_type": "ssta"})
    inputs = dict(
        hidden_states=torch.randn(1, 4, 1, 1, 5),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 3, 8),
        encoder_attention_mask=torch.ones(1, 3),
        encoder_hidden_states_2=torch.randn(1, 1, 8),
        encoder_attention_mask_2=torch.zeros(1, 1),
        image_embeds=torch.zeros(1, 1, 8),
    )
    with torch.no_grad():
        for backend in (
            AttentionBackendType.FLEX_BLOCK_ATTN,
            AttentionBackendType.SDPA,
            AttentionBackendType.FLEX_BLOCK_ATTN,
        ):
            state.attention_backend = backend
            calls.clear()
            output = model(**inputs).sample
            assert output.shape == inputs["hidden_states"].shape
            for length, joint, kwargs in calls:
                # SSTA applies the per-sample text mask itself; a dense step has no use for it.
                assert (kwargs["text_mask"] is not None) == (backend == AttentionBackendType.FLEX_BLOCK_ATTN)
                if backend == AttentionBackendType.FLEX_BLOCK_ATTN:
                    assert length == 5  # local image length 3 + padded local text length 2
                    assert joint is None
                    assert kwargs["encoder_sp_padding"] == 1
                else:
                    assert length == 3
                    assert joint.shape[2] == 3
                    assert kwargs["encoder_sp_padding"] == 0
                    assert model._sp_pad_text["local_rows"] == 0
