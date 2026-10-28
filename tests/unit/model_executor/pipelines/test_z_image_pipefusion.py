from types import SimpleNamespace

import pytest
import torch

from tests.unit.model_executor.pipefusion._helpers import Group
from xfuser.model_executor.pipefusion import PipeFusionPatchLayout
from xfuser.model_executor.pipelines import pipeline_z_image_pipefusion
from xfuser.model_executor.pipelines.pipeline_z_image_pipefusion import (
    xFuserZImagePipeFusionPipeline,
)


def test_z_image_pipefusion_rejects_step_end_callbacks():
    with pytest.raises(NotImplementedError, match="callback_on_step_end"):
        xFuserZImagePipeFusionPipeline._validate_callback(lambda *_args: None)


@pytest.mark.parametrize(
    ("cfg_size", "output"),
    [
        (1, torch.tensor([3.0, 1.0]).reshape(2, 1, 1, 1)),
        (2, torch.tensor([3.0]).reshape(1, 1, 1, 1)),
    ],
)
def test_z_image_cfg_matches_positive_first_diffusers_formula(monkeypatch, cfg_size, output):
    monkeypatch.setattr(
        pipeline_z_image_pipefusion,
        "get_classifier_free_guidance_world_size",
        lambda: cfg_size,
    )
    monkeypatch.setattr(
        pipeline_z_image_pipefusion,
        "get_cfg_group",
        lambda: SimpleNamespace(
            all_gather=lambda _output, separate_tensors: (
                torch.full((1, 1, 1, 1), 3.0),
                torch.ones(1, 1, 1, 1),
            )
        ),
    )
    pipeline = SimpleNamespace(
        do_classifier_free_guidance=True,
        _cfg_normalization=False,
    )

    result = xFuserZImagePipeFusionPipeline._cfg_prediction(
        pipeline,
        output,
        guidance_scale=2.0,
    )

    assert result.item() == 7.0


def test_z_image_scheduler_negates_model_prediction():
    seen = {}

    def step(noise_pred, timestep, latents, return_dict):
        seen["arguments"] = (noise_pred, timestep, latents, return_dict)
        return (latents + noise_pred,)

    pipeline = SimpleNamespace(scheduler=SimpleNamespace(step=step))
    prediction = torch.full((1, 1, 1, 1), 2.0)
    latents = torch.ones(1, 1, 1, 1)
    timestep = torch.tensor(5.0)

    result = xFuserZImagePipeFusionPipeline._scheduler_step(
        pipeline,
        prediction,
        timestep,
        latents,
    )

    noise_pred, seen_timestep, seen_latents, return_dict = seen["arguments"]
    assert noise_pred.item() == -2.0
    assert seen_timestep is timestep
    assert seen_latents is latents
    assert return_dict is False
    assert result.item() == -1.0


def test_z_image_sync_pipeline_broadcasts_feedback_instead_of_cyclic_p2p(monkeypatch):
    sent = []
    handoffs = []
    inputs = []
    state = SimpleNamespace(set_patched_mode=lambda value: None)
    group = SimpleNamespace(
        pipeline_send=lambda value: sent.append(value.clone()),
        pipeline_recv=lambda: pytest.fail("first stage must not receive cyclic feedback"),
    )
    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(order=1),
        _backbone_forward=lambda latents, *_args, **_kwargs: inputs.append(latents.clone()) or latents + 1,
        _pipefusion_handoff_sync_latents=lambda value, **_kwargs: (
            handoffs.append(value.clone()) or torch.full_like(value, 10 * len(handoffs))
        ),
    )
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pp_group", lambda: group)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_last_stage", lambda: False)

    result = xFuserZImagePipeFusionPipeline._sync_pipeline(
        pipeline,
        latents=torch.zeros(1),
        prompt_embeds=[],
        timesteps=torch.tensor([2.0, 1.0]),
        guidance_scales=[4.0, 4.0],
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
    )

    assert [value.item() for value in inputs] == [0.0, 10.0]
    assert [value.item() for value in sent] == [1.0, 11.0]
    assert len(handoffs) == 2
    assert result.item() == 20.0


def test_z_image_sync_pipeline_last_stage_does_not_send_feedback(monkeypatch):
    received = iter((torch.ones(1), torch.full((1,), 2.0)))
    group = SimpleNamespace(
        pipeline_send=lambda _value: pytest.fail("last stage feedback must use the collective handoff"),
        pipeline_recv=lambda: next(received),
    )
    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(order=1),
        _backbone_forward=lambda _latents, *_args, pipeline_hidden_states, **_kwargs: pipeline_hidden_states,
        _scheduler_step=lambda output, _timestep, previous: previous + output,
        _pipefusion_handoff_sync_latents=lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        pipeline_z_image_pipefusion,
        "get_runtime_state",
        lambda: SimpleNamespace(set_patched_mode=lambda value: None),
    )
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pp_group", lambda: group)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_last_stage", lambda: True)

    result = xFuserZImagePipeFusionPipeline._sync_pipeline(
        pipeline,
        latents=torch.full((1,), 5.0),
        prompt_embeds=[],
        timesteps=torch.tensor([2.0, 1.0]),
        guidance_scales=[4.0, 4.0],
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
    )

    assert result.item() == 8.0


def test_z_image_async_intermediate_stage_retains_patch_geometry(monkeypatch):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=0),
        pp_patches_start_end_idx_global=((0, 1), (1, 2)),
        next_patch=lambda: None,
        increment_step_counter=lambda: None,
    )
    group = Group(
        (
            torch.full((1, 33, 4), 10.0),
            torch.full((1, 33, 4), 20.0),
            torch.full((1, 33, 4), 30.0),
            torch.full((1, 33, 4), 40.0),
        )
    )
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pp_group", lambda: group)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_last_stage", lambda: False)
    monkeypatch.setattr(
        pipeline_z_image_pipefusion,
        "get_classifier_free_guidance_world_size",
        lambda: 1,
    )

    geometry = []

    def backbone_forward(latents, *_args, pipeline_hidden_states, **_kwargs):
        geometry.append(latents.clone())
        return pipeline_hidden_states

    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(dim=4)),
        do_classifier_free_guidance=False,
        _init_async_pipeline=lambda *_args, **_kwargs: [None, None],
        _backbone_forward=backbone_forward,
        _pipefusion_end_step=lambda *_args, **_kwargs: None,
    )
    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(1, 1),
        token_counts=(1, 1),
        name="z-image-test",
    )
    latents = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)

    result = xFuserZImagePipeFusionPipeline._async_pipeline(
        pipeline,
        latents=latents,
        prompt_embeds=[torch.zeros(1, 4)],
        timesteps=(torch.tensor(2.0), torch.tensor(1.0)),
        guidance_scales=(4.0, 4.0),
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
        layout=layout,
        sync_handoff_complete=True,
    )

    assert result is None
    assert [patch.item() for patch in geometry] == [1.0, 2.0, 1.0, 2.0]
    assert [sent[2][0, 0, 0].item() for sent in group.sent] == [10.0, 20.0, 30.0, 40.0]


def test_z_image_async_last_stage_returns_each_scheduled_patch_immediately(monkeypatch):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=0),
        pp_patches_start_end_idx_global=((0, 1), (1, 2)),
        next_patch=lambda: None,
    )
    group = Group(
        (
            torch.full((1, 1, 1, 1), 10.0),
            torch.full((1, 1, 1, 1), 20.0),
            torch.full((1, 1, 1, 1), 30.0),
            torch.full((1, 1, 1, 1), 40.0),
        )
    )
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_pp_group", lambda: group)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(
        pipeline_z_image_pipefusion,
        "get_classifier_free_guidance_world_size",
        lambda: 1,
    )

    scheduler_inputs = []

    def scheduler_step(output, _timestep, previous):
        scheduler_inputs.append((output.item(), previous.item()))
        return previous + output

    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(1, 1),
        token_counts=(1, 1),
        name="z-image-test",
    )
    latents = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(dim=4)),
        do_classifier_free_guidance=False,
        _init_async_pipeline=lambda *_args, **_kwargs: layout.split(latents),
        _backbone_forward=lambda _latents, *_args, pipeline_hidden_states, **_kwargs: pipeline_hidden_states,
        _scheduler_step=scheduler_step,
        _pipefusion_end_step=lambda *_args, **_kwargs: None,
    )

    result = xFuserZImagePipeFusionPipeline._async_pipeline(
        pipeline,
        latents=latents,
        prompt_embeds=[torch.zeros(1, 4)],
        timesteps=(torch.tensor(2.0), torch.tensor(1.0)),
        guidance_scales=(4.0, 4.0),
        num_warmup_steps=0,
        progress_bar=SimpleNamespace(update=lambda: None),
        layout=layout,
        sync_handoff_complete=True,
    )

    assert scheduler_inputs == [(10.0, 1.0), (20.0, 2.0), (30.0, 11.0), (40.0, 22.0)]
    assert [sent[2].item() for sent in group.sent] == [11.0, 22.0]
    assert result.flatten().tolist() == [41.0, 62.0]


@pytest.mark.parametrize("output_owner", [True, False])
def test_z_image_parallel_vae_decode_returns_on_the_wrapper_output_rank(monkeypatch, output_owner):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(use_parallel_vae=True),
        parallel_config=SimpleNamespace(vae_parallel_size=0),
    )
    monkeypatch.setattr(pipeline_z_image_pipefusion, "get_runtime_state", lambda: state)
    # Not the last pipeline stage, which parallel VAE no longer makes the output rank.
    monkeypatch.setattr(pipeline_z_image_pipefusion, "is_dp_last_group", lambda: False)
    decoded = torch.zeros(1, 3, 4, 4)
    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            dtype=torch.float32,
            config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
            decode=lambda _latents, return_dict: (decoded,),
        ),
        gather_broadcast_latents=lambda value: value,
        is_dp_last_group=lambda: output_owner,
        image_processor=SimpleNamespace(postprocess=lambda image, output_type: ("image", image.shape)),
        maybe_free_model_hooks=lambda: None,
    )

    result = xFuserZImagePipeFusionPipeline._decode_latents(
        pipe, torch.zeros(1, 4, 2, 2), output_type="pil", return_dict=False
    )

    assert result == ((("image", (1, 3, 4, 4)),) if output_owner else None)
