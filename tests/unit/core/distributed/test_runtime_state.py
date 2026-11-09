"""Per-request input handling of the DiT runtime state."""

from types import SimpleNamespace

import pytest

from xfuser.config.config import InputConfig, RuntimeConfig
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.runtime_state import DiTRuntimeState


@pytest.fixture
def state(monkeypatch):
    # The pipeline-parallel group is a process-group boundary; a request at an
    # unchanged size only asks it to drop its cached transfer shapes.
    pp_group = SimpleNamespace(reset_buffer=lambda: None, set_config=lambda dtype: None)
    monkeypatch.setattr(runtime_state_module, "get_pp_group", lambda: pp_group)

    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.runtime_config = RuntimeConfig(warmup_steps=4)
    state.input_config = InputConfig(height=512, width=512, num_frames=9, batch_size=1)
    state.ready = True
    state.split_latents_by_rows = True
    return state


def _image_request(state, num_inference_steps):
    state.set_input_parameters(height=512, width=512, batch_size=1, num_inference_steps=num_inference_steps)


def _video_request(state, num_inference_steps):
    state.set_video_input_parameters(
        height=512, width=512, num_frames=9, batch_size=1, num_inference_steps=num_inference_steps
    )


@pytest.mark.parametrize("request_fn", [_image_request, _video_request], ids=["image", "video"])
def test_short_request_does_not_shrink_warmup_of_later_requests(state, request_fn):
    # Pipelines read runtime_config.warmup_steps for every request and fall
    # back to synchronous steps when a request has no more steps than that.
    warmup_seen = []
    for num_inference_steps in (28, 2, 28):
        request_fn(state, num_inference_steps)
        warmup_seen.append(state.runtime_config.warmup_steps)

    assert warmup_seen == [4, 4, 4]


def test_wan_pipeline_uses_spatial_patch_size_and_tracks_temporal_patch_size():
    pipeline_type = type("xFuserWanTI2VPipeFusionPipeline", (), {})
    pipeline = pipeline_type()
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.transformer = SimpleNamespace(
        config=SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=48,
            num_attention_heads=24,
            attention_head_dim=128,
        )
    )
    state = DiTRuntimeState.__new__(DiTRuntimeState)

    state._check_pipeline_class_name(pipeline, config=None)

    assert state.wan_video is True
    assert state.backbone_patch_size == 2
    assert state.wan_temporal_patch_size == 1


def test_wan_patch_tokens_include_the_full_temporal_extent(monkeypatch):
    monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_rank", lambda: 0)
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.input_config = InputConfig(height=1280, width=736, num_frames=121, batch_size=1)
    state.parallel_config = SimpleNamespace(pp_config=SimpleNamespace(num_pipeline_patch=2))
    state.backbone_patch_size = 2
    state.vae_scale_factor_spatial = 8
    state.vae_scale_factor_temporal = 4
    state.wan_temporal_patch_size = 1

    state._calc_wan_patches_metadata()

    assert state.pp_patches_height == [80, 80]
    assert state.pp_patches_token_num == [57040, 57040]
    assert state.pp_patches_token_start_idx_local == [0, 57040, 114080]
    assert state.pp_patches_token_start_end_idx_global == [[0, 57040], [57040, 114080]]
