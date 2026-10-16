from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xfuser.model_executor.pipefusion import PipeFusionPatchLayout
from xfuser.model_executor.pipelines import pipeline_qwen_image
from xfuser.model_executor.pipelines.pipeline_qwen_image import QwenPipeFusionMixin
from xfuser.model_executor.pipelines.pipeline_qwen_image_edit import _image_size


def test_qwen_pipefusion_rejects_step_end_callbacks():
    with pytest.raises(NotImplementedError, match="cannot be applied consistently across stages"):
        QwenPipeFusionMixin._validate_qwen_callback(lambda *_args: None)


def test_qwen_sync_prefix_uses_explicit_patch_mode_handoff(monkeypatch):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=4),
        num_pipeline_patch=4,
        pp_patches_token_num=(2, 2, 2, 2),
    )
    monkeypatch.setattr(pipeline_qwen_image, "get_runtime_state", lambda: state)
    monkeypatch.setattr(
        pipeline_qwen_image.PipeFusionPatchLayout,
        "from_runtime_state",
        lambda *_args, **_kwargs: SimpleNamespace(install=lambda _state: nullcontext()),
    )
    calls = {}
    synchronized = torch.tensor([1.0])
    handed_off = torch.tensor([2.0])

    def sync_pipeline(**kwargs):
        calls["sync_only"] = kwargs["sync_only"]
        calls["sync_timesteps"] = tuple(kwargs["timesteps"])
        return synchronized

    def handoff(latents, **kwargs):
        calls["handoff"] = (latents, kwargs["name"])
        return handed_off

    def async_pipeline(**kwargs):
        calls["async_latents"] = kwargs["latents"]
        calls["sync_handoff_complete"] = kwargs["sync_handoff_complete"]
        calls["async_timesteps"] = tuple(kwargs["timesteps"])
        return "done"

    pipe = SimpleNamespace(
        _pipefusion_async_enabled=lambda **_kwargs: True,
        _sync_qwen_pipeline=sync_pipeline,
        _pipefusion_handoff_sync_latents=handoff,
        _async_qwen_pipeline=async_pipeline,
    )

    result = QwenPipeFusionMixin._denoise_qwen(
        pipe,
        latents=torch.zeros(1, 8, 2),
        image_latents=None,
        condition=torch.zeros(1, 3, 2),
        condition_mask=torch.ones(1, 3, dtype=torch.bool),
        guidance=None,
        img_shapes=[[(1, 2, 4)]],
        timesteps=tuple(range(6)),
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
    )

    assert result == "done"
    assert calls["sync_only"] is True
    assert calls["sync_timesteps"] == (0, 1, 2, 3)
    assert calls["handoff"] == (synchronized, "pipefusion_sync_latents")
    assert calls["async_latents"] is handed_off
    assert calls["sync_handoff_complete"] is True
    assert calls["async_timesteps"] == (4, 5)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (torch.empty(3, 12, 20), (20, 12)),
        (np.empty((12, 20, 3)), (20, 12)),
        ([SimpleNamespace(size=(20, 12))], (20, 12)),
    ],
)
def test_edit_accepts_supported_image_size_interfaces(image, expected):
    assert _image_size(image) == expected


def test_edit_layout_slices_global_rope_across_generated_and_fixed_tokens():
    layout = PipeFusionPatchLayout.with_appended_reference_tokens(
        generated_tokens=6,
        reference_tokens=2,
        num_patches=4,
        name="qwen-edit",
    )
    image_freqs = torch.arange(8)
    text_freqs = torch.arange(3)
    pipe = SimpleNamespace(
        transformer=SimpleNamespace(
            pos_embed=lambda *_args, **_kwargs: (image_freqs, text_freqs),
        )
    )

    rotary = QwenPipeFusionMixin._patch_rotary_embeddings(
        pipe,
        layout,
        [[(1, 2, 3), (1, 1, 2)]],
        text_tokens=3,
        device=torch.device("cpu"),
    )

    assert [item[0].tolist() for item in rotary] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert all(item[1] is text_freqs for item in rotary)
    assert layout.token_counts == (2, 2, 2, 0)
    assert layout.reference_token_counts == (0, 0, 0, 2)


def test_pure_pipefusion_cfg_keeps_patch_kv_per_branch():
    class Layer:
        _xdit_kv_cache = None

    layer = Layer()
    events = []

    @contextmanager
    def cache_context(name):
        events.append(name)
        yield

    pipe = SimpleNamespace(
        transformer=SimpleNamespace(
            transformer_blocks=[SimpleNamespace(attn=layer)],
            cache_context=cache_context,
        ),
        _qwen_branch_kv={"uncond": {}, "cond": {}},
    )

    with QwenPipeFusionMixin._cache_context(pipe, "uncond"):
        layer._xdit_kv_cache = torch.tensor([1])
    with QwenPipeFusionMixin._cache_context(pipe, "cond"):
        assert layer._xdit_kv_cache is None
        layer._xdit_kv_cache = torch.tensor([2])
    with QwenPipeFusionMixin._cache_context(pipe, "uncond"):
        torch.testing.assert_close(layer._xdit_kv_cache, torch.tensor([1]))

    assert events == ["uncond", "cond", "uncond"]


def test_sync_edit_reappends_fixed_tokens_but_only_schedules_generated(monkeypatch):
    monkeypatch.setattr(pipeline_qwen_image, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_qwen_image, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(pipeline_qwen_image, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(
        pipeline_qwen_image,
        "get_runtime_state",
        lambda: SimpleNamespace(set_patched_mode=lambda _value: None),
    )
    seen_model_lengths = []
    seen_scheduler_lengths = []
    pipe = SimpleNamespace(
        interrupt=False,
        scheduler=SimpleNamespace(order=1),
        _qwen_do_true_cfg=False,
    )

    def backbone(latents, encoder_hidden_states, timestep, **kwargs):
        seen_model_lengths.append(latents.shape[1])
        return torch.zeros_like(latents), encoder_hidden_states

    def scheduler_step(noise_pred, previous, timestep):
        seen_scheduler_lengths.append((noise_pred.shape[1], previous.shape[1]))
        return previous

    pipe._backbone_forward = backbone
    pipe._cfg_prediction = lambda value: value
    pipe._scheduler_step = scheduler_step
    progress = SimpleNamespace(update=lambda: None)
    generated = torch.randn(1, 6, 2)
    fixed = torch.randn(1, 2, 2)

    result = QwenPipeFusionMixin._sync_qwen_pipeline(
        pipe,
        latents=generated,
        image_latents=fixed,
        condition=torch.randn(1, 3, 2),
        condition_mask=torch.ones(1, 3, dtype=torch.bool),
        guidance=None,
        img_shapes=[[(1, 2, 3), (1, 1, 2)]],
        generated_tokens=6,
        timesteps=[torch.tensor(2), torch.tensor(1)],
        num_warmup_steps=0,
        progress_bar=progress,
        sync_only=True,
    )

    assert seen_model_lengths == [8, 8]
    assert seen_scheduler_lengths == [(6, 6), (6, 6)]
    torch.testing.assert_close(result, generated)


def test_true_cfg_normalization_matches_qwen_formula(monkeypatch):
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_world_size", lambda: 1)
    pipe = SimpleNamespace(_qwen_do_true_cfg=True, _qwen_true_cfg_scale=2.0)
    negative = torch.tensor([[[1.0, 0.0]]])
    positive = torch.tensor([[[0.0, 2.0]]])

    actual = QwenPipeFusionMixin._cfg_prediction(pipe, torch.cat((negative, positive)))
    expected = torch.tensor([[[-0.48507124, 1.940285]]])

    torch.testing.assert_close(actual, expected)


def test_pure_cfg_preserves_the_available_prompt_mask(monkeypatch):
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_world_size", lambda: 1)
    pipe = SimpleNamespace(_qwen_do_true_cfg=True)
    cond = torch.zeros(1, 3, 2)
    uncond = torch.zeros_like(cond)
    cond_mask = torch.tensor([[True, False, False]])

    _, mask = QwenPipeFusionMixin._select_cfg_condition(
        pipe,
        cond,
        uncond,
        cond_mask,
        None,
    )

    torch.testing.assert_close(mask, torch.tensor([[True, True, True], [True, False, False]]))


def test_qwen_backbone_accepts_missing_prompt_mask(monkeypatch):
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_world_size", lambda: 1)
    calls = []

    def transformer(**kwargs):
        calls.append(kwargs)
        return ((kwargs["hidden_states"], kwargs["encoder_hidden_states"]),)

    @contextmanager
    def cache_context(_branch):
        yield

    pipe = SimpleNamespace(
        _qwen_do_true_cfg=False,
        transformer=transformer,
        attention_kwargs={},
        _cache_context=cache_context,
    )
    image = torch.randn(1, 2, 3)
    text = torch.randn(1, 4, 3)

    actual_image, actual_text = QwenPipeFusionMixin._backbone_forward(
        pipe,
        image,
        text,
        torch.tensor(1),
        encoder_hidden_states_mask=None,
        guidance=None,
        img_shapes=[[(1, 1, 2)]],
    )

    assert calls[0]["encoder_hidden_states_mask"] is None
    torch.testing.assert_close(actual_image, image)
    torch.testing.assert_close(actual_text, text)


def test_reference_only_patch_skips_scheduler():
    pipe = SimpleNamespace(
        _pipefusion_update_patch=lambda *_args: (_ for _ in ()).throw(AssertionError("scheduler called")),
    )
    previous = torch.empty(1, 0, 2)

    actual = QwenPipeFusionMixin._pipefusion_update_qwen_patch(
        pipe,
        SimpleNamespace(),
        torch.tensor(1),
        torch.empty_like(previous),
        previous,
    )

    assert actual is previous
