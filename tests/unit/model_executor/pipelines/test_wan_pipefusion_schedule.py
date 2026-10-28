from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.unit.model_executor.pipefusion._helpers import Group
from xfuser.model_executor.pipefusion import PipeFusionPatchLayout
from xfuser.model_executor.pipelines import pipeline_wan_pipefusion
from xfuser.model_executor.pipelines.pipeline_wan_pipefusion import (
    xFuserWanTI2VPipeFusionPipeline,
)


def test_wan_downstream_stage_uses_full_temporal_mask_for_geometry(monkeypatch):
    class Transformer:
        dtype = torch.bfloat16

        def cache_context(self, _key):
            return nullcontext()

        def __call__(self, **kwargs):
            self.hidden_states = kwargs["hidden_states"]
            return (kwargs["pipeline_hidden_states"],)

    transformer = Transformer()
    pipeline = SimpleNamespace(
        transformer=transformer,
        _timestep=lambda _timestep, _mask, batch_size: torch.zeros(batch_size, 1),
    )
    condition = torch.zeros(1, 4, 1, 3, 2)
    full_temporal_mask = torch.ones(1, 4, 3, 3, 2)
    received = torch.zeros(1, 54, 8)
    monkeypatch.setattr(pipeline_wan_pipefusion, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_pipeline_last_stage", lambda: False)

    result = xFuserWanTI2VPipeFusionPipeline._backbone_forward(
        pipeline,
        latents=None,
        condition=condition,
        first_frame_mask=full_temporal_mask,
        prompt_embeds=torch.zeros(1, 2, 8),
        timestep=torch.tensor(1.0),
        do_cfg=False,
        guidance_scale=1.0,
        pipeline_hidden_states=received,
    )

    assert transformer.hidden_states.shape == full_temporal_mask.shape
    assert transformer.hidden_states.dtype == transformer.dtype
    assert result is received


@pytest.mark.parametrize(
    ("cfg_size", "transformer_output", "expected_batch"),
    [
        (1, torch.tensor([1.0, 3.0]), 2),
        (2, torch.tensor([1.0]), 1),
    ],
)
def test_wan_last_stage_applies_cfg_for_local_and_distributed_branches(
    monkeypatch,
    cfg_size,
    transformer_output,
    expected_batch,
):
    class Transformer:
        dtype = torch.float32

        def cache_context(self, key):
            assert key == ("uncond" if cfg_size == 2 else "cond")
            return nullcontext()

        def __call__(self, **kwargs):
            assert kwargs["hidden_states"].shape[0] == expected_batch
            shape = (transformer_output.shape[0], 1, 1, 1, 1)
            return (transformer_output.reshape(shape),)

    monkeypatch.setattr(pipeline_wan_pipefusion, "get_classifier_free_guidance_world_size", lambda: cfg_size)
    monkeypatch.setattr(pipeline_wan_pipefusion, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(
        pipeline_wan_pipefusion,
        "get_cfg_group",
        lambda: SimpleNamespace(
            all_gather=lambda _output, separate_tensors: (
                torch.ones(1, 1, 1, 1, 1),
                torch.full((1, 1, 1, 1, 1), 3.0),
            )
        ),
    )
    pipeline = SimpleNamespace(
        transformer=Transformer(),
        _model_input=lambda latents, _condition, _mask: latents,
        _timestep=lambda _timestep, _mask, batch_size: torch.zeros(batch_size, 1),
    )

    result = xFuserWanTI2VPipeFusionPipeline._backbone_forward(
        pipeline,
        latents=torch.zeros(1, 1, 1, 1, 1),
        condition=torch.zeros(1, 1, 1, 1, 1),
        first_frame_mask=torch.ones(1, 1, 1, 1, 1),
        prompt_embeds=torch.zeros(expected_batch, 1, 1),
        timestep=torch.tensor(1.0),
        do_cfg=True,
        guidance_scale=2.0,
    )

    assert result.item() == 5.0


def test_wan_async_schedule_returns_each_patch_after_patch_local_scheduler(monkeypatch):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=1),
        pp_patches_start_end_idx_global=[[0, 1], [1, 2]],
        increment_step_counter=lambda: None,
        next_patch=lambda: None,
    )
    group = Group(
        [
            torch.full((1, 1, 1, 1, 1), 10.0),
            torch.full((1, 1, 1, 1, 1), 20.0),
            torch.full((1, 1, 1, 1, 1), 30.0),
            torch.full((1, 1, 1, 1, 1), 40.0),
        ]
    )
    monkeypatch.setattr(pipeline_wan_pipefusion, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_wan_pipefusion, "get_pp_group", lambda: group)
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_pipeline_last_stage", lambda: True)

    patch_latents = [
        torch.full((1, 1, 1, 1, 1), 1.0),
        torch.full((1, 1, 1, 1, 1), 2.0),
    ]

    class Scheduler:
        model_outputs = [None, None]
        last_sample = None

        def step(self, noise, _timestep, previous, return_dict):
            assert return_dict is False
            return (noise + previous,)

    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=1, attention_head_dim=1),
        ),
        scheduler=Scheduler(),
        _init_async_pipeline=lambda *_args, **_kwargs: patch_latents,
        _backbone_forward=lambda *_args, pipeline_hidden_states, **_kwargs: pipeline_hidden_states,
        _pipefusion_end_step=lambda *_args, **_kwargs: None,
    )
    layout = PipeFusionPatchLayout(
        split_dim=3,
        split_sizes=(1, 1),
        token_counts=(1, 1),
        name="wan-test",
    )

    result = xFuserWanTI2VPipeFusionPipeline._async_pipeline(
        pipeline,
        latents=torch.cat(patch_latents, dim=3),
        condition=torch.zeros(1, 1, 1, 2, 1),
        first_frame_mask=torch.ones(1, 1, 1, 2, 1),
        prompt_embeds=torch.zeros(1, 1, 1),
        timesteps=(torch.tensor(5.0), torch.tensor(4.0)),
        do_cfg=False,
        guidance_scale=1.0,
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
        layout=layout,
    )

    assert [sent[2].item() for sent in group.sent] == [11, 22]
    assert result.flatten().tolist() == [41, 62]


@pytest.mark.parametrize("output_owner", [True, False])
def test_wan_parallel_vae_decode_returns_on_the_wrapper_output_rank(monkeypatch, output_owner):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(use_parallel_vae=True),
        parallel_config=SimpleNamespace(vae_parallel_size=0),
    )
    monkeypatch.setattr(pipeline_wan_pipefusion, "get_runtime_state", lambda: state)
    # Not the last pipeline stage, which parallel VAE no longer makes the output rank.
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_pipeline_last_stage", lambda: False)
    monkeypatch.setattr(pipeline_wan_pipefusion, "is_dp_last_group", lambda: False)
    decoded = torch.zeros(1, 3, 2, 4, 4)
    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            dtype=torch.float32,
            config=SimpleNamespace(z_dim=2, latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0]),
            decode=lambda _latents, return_dict: (decoded,),
        ),
        gather_broadcast_latents=lambda value: value,
        is_dp_last_group=lambda: output_owner,
        video_processor=SimpleNamespace(postprocess_video=lambda video, output_type: ("video", video.shape)),
        maybe_free_model_hooks=lambda: None,
    )
    latents = torch.zeros(1, 2, 1, 2, 2)

    result = xFuserWanTI2VPipeFusionPipeline._decode_latents(
        pipe, latents, torch.ones_like(latents), torch.zeros_like(latents), output_type="np", return_dict=False
    )

    assert result == ((("video", (1, 3, 2, 4, 4)),) if output_owner else None)
