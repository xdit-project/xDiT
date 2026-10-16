from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("diffusers.models.transformers.transformer_qwenimage")

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel  # noqa: E402

from xfuser.model_executor.models.transformers import transformer_qwen  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    return QwenImageTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
    ).eval()


def _inputs():
    generator = torch.Generator().manual_seed(1)
    return {
        "hidden_states": torch.randn(1, 6, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, 3, 12, generator=generator),
        "encoder_hidden_states_mask": torch.tensor([[True, True, False]]),
        "timestep": torch.tensor([0.4]),
        "img_shapes": [(1, 2, 3)],
    }


def _install_pipefusion_processors(model, monkeypatch):
    cache = SimpleNamespace(update_and_get_kv_cache=lambda new_kv, **_kwargs: new_kv)
    monkeypatch.setattr(
        transformer_qwen,
        "get_runtime_state",
        lambda: SimpleNamespace(num_pipeline_patch=2),
    )
    monkeypatch.setattr(transformer_qwen, "get_cache_manager", lambda: cache)
    monkeypatch.setattr(transformer_qwen, "_HAS_FLYDSL", False)

    def local_attention(query, key, value, attention_kwargs=None, **_kwargs):
        mask = attention_kwargs.get("attn_mask") if attention_kwargs else None
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    monkeypatch.setattr(transformer_qwen, "USP", local_attention)
    for block in model.transformer_blocks:
        block.attn.processor = transformer_qwen.xFuserQwenDoubleStreamAttnProcessor()


def test_single_stage_pipefusion_forward_matches_diffusers(monkeypatch):
    monkeypatch.setattr(transformer_qwen, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_qwen, "is_pipeline_last_stage", lambda: True)
    model = _tiny_model()
    inputs = _inputs()

    with torch.no_grad():
        expected = model(**inputs, return_dict=False)[0]
        _install_pipefusion_processors(model, monkeypatch)
        ((actual, encoder_out),) = transformer_qwen.xFuserQwenImageTransformer2DWrapper.forward(
            model, **inputs, return_dict=False
        )

    assert encoder_out is None
    torch.testing.assert_close(actual, expected)


def test_pipefusion_forward_accepts_patch_rotary_embeddings(monkeypatch):
    monkeypatch.setattr(transformer_qwen, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_qwen, "is_pipeline_last_stage", lambda: True)
    model = _tiny_model()
    inputs = _inputs()
    rotary = model.pos_embed(
        inputs["img_shapes"],
        max_txt_seq_len=inputs["encoder_hidden_states"].shape[1],
        device=inputs["hidden_states"].device,
    )
    monkeypatch.setattr(
        model.pos_embed,
        "forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recomputed rotary embeddings")),
    )
    _install_pipefusion_processors(model, monkeypatch)

    with torch.no_grad():
        ((output, encoder_out),) = transformer_qwen.xFuserQwenImageTransformer2DWrapper.forward(
            model,
            **inputs,
            image_rotary_emb=rotary,
            return_dict=False,
        )

    assert output.shape == inputs["hidden_states"].shape
    assert encoder_out is None


def test_compile_capture_propagates_qwen_stage_shapes(monkeypatch):
    model = _tiny_model()
    inputs = _inputs()
    monkeypatch.setattr(transformer_qwen, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_qwen, "is_pipeline_last_stage", lambda: False)

    ((image_state, text_state),) = (
        transformer_qwen.xFuserQwenImageTransformer2DWrapper.pipefusion_compile_capture_forward(
            model, **inputs, return_dict=False
        )
    )

    assert image_state.shape == (1, 6, 16)
    assert text_state.shape == (1, 3, 16)
    assert torch.count_nonzero(image_state) == 0
    assert torch.count_nonzero(text_state) == 0

    monkeypatch.setattr(transformer_qwen, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(transformer_qwen, "is_pipeline_last_stage", lambda: True)
    ((output, encoder_out),) = transformer_qwen.xFuserQwenImageTransformer2DWrapper.pipefusion_compile_capture_forward(
        model, image_state, text_state, return_dict=False
    )

    assert output.shape == (1, 6, 8)
    assert encoder_out is None


class _IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.to_q = nn.Identity()
        self.to_k = nn.Identity()
        self.to_v = nn.Identity()
        self.add_q_proj = nn.Identity()
        self.add_k_proj = nn.Identity()
        self.add_v_proj = nn.Identity()
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.norm_added_q = nn.Identity()
        self.norm_added_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.to_add_out = nn.Identity()


def test_pipefusion_attention_caches_only_image_kv_and_keeps_text_fresh(monkeypatch):
    attn = _IdentityAttention()
    processor = transformer_qwen.xFuserQwenDoubleStreamAttnProcessor()
    processor.use_long_ctx_attn_kvcache = False
    image = torch.tensor([[[1.0, 2.0]]])
    text = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
    text_mask = torch.tensor([[True, False]])
    calls = {}

    class Cache:
        def update_and_get_kv_cache(self, new_kv, **kwargs):
            calls["cached"] = (new_kv, kwargs)
            prefix = torch.full_like(new_kv[0], 9)
            return [torch.cat([prefix, item], dim=1) for item in new_kv]

    def fake_usp(query, key, value, **kwargs):
        calls["usp"] = (query, key, value, kwargs)
        return query

    monkeypatch.setattr(transformer_qwen, "get_runtime_state", lambda: SimpleNamespace(num_pipeline_patch=2))
    monkeypatch.setattr(transformer_qwen, "get_cache_manager", Cache)
    monkeypatch.setattr(transformer_qwen, "USP", fake_usp)

    image_out, text_out = processor(
        attn,
        image,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        latte_temporal_attention=False,
    )

    cached, cache_kwargs = calls["cached"]
    torch.testing.assert_close(cached[0].flatten(2), image)
    torch.testing.assert_close(cached[1].flatten(2), image)
    assert cache_kwargs["layer"] is attn

    _, joint_key, _, usp_kwargs = calls["usp"]
    torch.testing.assert_close(joint_key[:, :, :2].transpose(1, 2).flatten(2), text)
    assert joint_key.shape[2] == 4
    assert usp_kwargs["attn_layer"] is None
    assert usp_kwargs["head_balance_layer"] is attn
    assert usp_kwargs["attention_kwargs"]["attn_mask"].shape == (1, 1, 1, 4)
    torch.testing.assert_close(image_out, image)
    torch.testing.assert_close(text_out, text)
