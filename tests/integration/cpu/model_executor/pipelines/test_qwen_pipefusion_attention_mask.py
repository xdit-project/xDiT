from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("diffusers.models.transformers.transformer_qwenimage")

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel  # noqa: E402

from xfuser.core.cache_manager import cache_manager  # noqa: E402
from xfuser.core.attention import AttnCall, AttentionBackendType, registry  # noqa: E402
from xfuser.core.distributed import runtime_state  # noqa: E402
from xfuser.model_executor.layers import usp  # noqa: E402
from xfuser.model_executor.models.transformers import transformer_qwen  # noqa: E402
from xfuser.model_executor.pipelines import base_pipeline, pipeline_qwen_image  # noqa: E402
from xfuser.model_executor.pipelines.pipeline_qwen_image import QwenPipeFusionMixin  # noqa: E402


class _Pipeline(QwenPipeFusionMixin):
    _pipefusion_async_enabled = base_pipeline.xFuserPipelineBaseWrapper._pipefusion_async_enabled


@pytest.mark.parametrize(
    "backend",
    [AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH, AttentionBackendType.AITER_FLYDSL_FP8],
    ids=["sdpa", "sdpa-math", "mask-unsupported"],
)
@pytest.mark.parametrize(
    ("masks", "true_cfg", "cfg_size", "unmasked_branches"),
    [
        (None, False, 1, 1),
        ([[True, True, True]], False, 1, 1),
        ([[True, True, False]], False, 1, 0),
        ([[True, True, True], [True, False, False]], False, 1, 0),
        ([[True, True, True], [True, True, True]], True, 1, 2),
        ([[True, False, False], [True, True, True]], True, 1, 1),
        ([[True, True, True]], True, 2, 1),
        ([[True, True, False]], True, 2, 0),
    ],
    ids=[
        "no-mask",
        "all-valid",
        "padding",
        "padded-batch",
        "pure-cfg-valid",
        "pure-cfg-padding",
        "cfg2-valid",
        "cfg2-padding",
    ],
)
def test_denoising_omits_only_redundant_masks(monkeypatch, masks, true_cfg, cfg_size, unmasked_branches, backend):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=0),
        num_pipeline_patch=2,
        patch_mode=False,
        set_patched_mode=lambda _value: None,
        attention_backend=AttentionBackendType.AITER_FLYDSL_FP8,
    )
    manager = cache_manager.CacheManager()
    monkeypatch.setattr(cache_manager, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(runtime_state, "get_runtime_state", lambda: state)
    for module in (pipeline_qwen_image, base_pipeline):
        monkeypatch.setattr(module, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_qwen_image, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_qwen_image, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(pipeline_qwen_image, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_world_size", lambda: cfg_size)
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_rank", lambda: 1)
    monkeypatch.setattr(
        pipeline_qwen_image,
        "get_cfg_group",
        lambda: SimpleNamespace(all_gather=lambda prediction, **_kwargs: (prediction * 0.9, prediction)),
    )
    monkeypatch.setattr(transformer_qwen, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_qwen, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(transformer_qwen, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(transformer_qwen, "get_runtime_state", lambda: state)
    monkeypatch.setattr(transformer_qwen, "get_cache_manager", lambda: manager)
    monkeypatch.setattr(pipeline_qwen_image, "get_cache_manager", lambda: manager)
    monkeypatch.setattr(transformer_qwen, "_HAS_FLYDSL", False)
    monkeypatch.setattr(usp, "get_runtime_state", lambda: state)
    attention_calls = []

    def local_attention(query, key, value, attention_kwargs=None, backend=None, **_kwargs):
        selected = backend or state.attention_backend
        attention_calls.append(selected)
        if selected != AttentionBackendType.AITER_FLYDSL_FP8:
            return registry.get(selected).run(query, key, value, AttnCall(attention_kwargs=attention_kwargs or {}))[0]
        return F.scaled_dot_product_attention(query, key, value)

    monkeypatch.setattr(transformer_qwen, "USP", local_attention)
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    for block in model.transformer_blocks:
        block.attn.processor = transformer_qwen.xFuserQwenDoubleStreamAttnProcessor()
        manager.register_cache_entry(block.attn, "attn")
    pipe = _Pipeline()
    pipe.transformer = partial(transformer_qwen.xFuserQwenImageTransformer2DWrapper.forward, model)
    pipe.transformer.transformer_blocks = model.transformer_blocks
    pipe._qwen_branch_kv = {"uncond": {}, "cond": {}} if true_cfg and cfg_size == 1 else None
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    pipe._qwen_do_true_cfg = true_cfg
    pipe._qwen_true_cfg_scale = 4.0
    pipe._attention_kwargs = {}
    condition_mask = torch.tensor(masks, dtype=torch.bool) if masks is not None else None
    batch = len(masks) if masks is not None else 1
    latent_batch = batch // 2 if true_cfg and cfg_size == 1 else batch
    inputs = dict(
        latents=torch.randn(latent_batch, 6, 8),
        image_latents=None,
        condition=torch.randn(batch, 3, 12),
        condition_mask=condition_mask,
        guidance=None,
        img_shapes=[(1, 2, 3)],
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
    )
    with torch.no_grad():
        pipe.scheduler.set_timesteps(2)
        expected = pipe._sync_qwen_pipeline(**inputs, generated_tokens=6, timesteps=pipe.scheduler.timesteps)
        attention_calls.clear()
        state.attention_backend = backend
        pipe.scheduler.set_timesteps(2)
        actual = pipe._denoise_qwen(**inputs, timesteps=pipe.scheduler.timesteps)

    torch.testing.assert_close(actual, expected)
    branches = 2 if true_cfg and cfg_size == 1 else 1
    assert len(attention_calls) == 2 * branches
    if backend == AttentionBackendType.AITER_FLYDSL_FP8:
        assert attention_calls.count(backend) == 2 * unmasked_branches
        assert attention_calls.count(AttentionBackendType.SDPA) == 2 * (branches - unmasked_branches)
    else:
        assert attention_calls == [backend] * (2 * branches)
