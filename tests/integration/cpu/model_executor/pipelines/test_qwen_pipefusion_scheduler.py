from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from xfuser.model_executor.pipefusion import PipeFusionImagePatchSchedule, PipeFusionPatchLayout
from xfuser.model_executor.pipelines import pipeline_qwen_image
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.pipeline_qwen_image import QwenPipeFusionMixin
from xfuser.model_executor.schedulers import base_scheduler
from xfuser.model_executor.schedulers import scheduling_flow_match_euler_discrete as flow_match


class _QwenPipeline(QwenPipeFusionMixin):
    _pipefusion_update_patch = xFuserPipelineBaseWrapper._pipefusion_update_patch


@pytest.mark.parametrize("true_cfg", [False, True], ids=["without-cfg", "pure-cfg"])
@pytest.mark.parametrize(
    ("generated", "reference", "num_patches"),
    [(4096, 4096, 2), (4096, 4096, 4), (16384, 4096, 2)],
    ids=["reference-only-pp2", "reference-only-pp4", "mixed-final-patch"],
)
def test_edit_patch_schedule_matches_full_frame_scheduler(monkeypatch, generated, reference, num_patches, true_cfg):
    state = SimpleNamespace(patch_mode=False, pipeline_patch_idx=0, num_pipeline_patch=num_patches)
    monkeypatch.setattr(flow_match, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_scheduler, "get_pipeline_parallel_world_size", lambda: num_patches)
    monkeypatch.setattr(base_scheduler, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_qwen_image, "get_classifier_free_guidance_world_size", lambda: 1)
    pipe = _QwenPipeline()
    pipe._qwen_do_true_cfg = true_cfg
    pipe._qwen_true_cfg_scale = 4.0
    pipe.scheduler = flow_match.xFuserFlowMatchEulerDiscreteSchedulerWrapper(FlowMatchEulerDiscreteScheduler())
    baseline = FlowMatchEulerDiscreteScheduler()
    for scheduler in (pipe.scheduler, baseline):
        scheduler.set_timesteps(sigmas=[1.0, 0.8, 0.5, 0.1])
    initial = torch.ones(1, generated, 2)
    prediction = torch.full_like(initial, 0.25)
    actual = pipe._scheduler_step(prediction, initial, pipe.scheduler.timesteps[0])
    expected = baseline.step(prediction, baseline.timesteps[0], initial, return_dict=False)[0]
    for timestep in baseline.timesteps[1:]:
        expected = baseline.step(prediction, timestep, expected, return_dict=False)[0]

    layout = PipeFusionPatchLayout.with_appended_reference_tokens(
        generated_tokens=generated, reference_tokens=reference, num_patches=num_patches, name="qwen-edit"
    )
    fixed = torch.full((1, reference, 2), 7.0)
    patches = layout.split(torch.cat((actual, fixed), dim=1))
    indices = []
    state.patch_mode = True

    def advance_patch():
        state.pipeline_patch_idx = (state.pipeline_patch_idx + 1) % num_patches

    def forward_patch(work, timestep, image, condition):
        prediction = torch.full_like(image, 0.25)
        if true_cfg:
            prediction = torch.cat((prediction, prediction), dim=0)
        return prediction, condition

    # Replace process transport and transformer output; keep the real async
    # driver, patch commits, Qwen update and wrapped scheduler.
    transport = SimpleNamespace(
        num_steps=3,
        num_patches=num_patches,
        first_stage=True,
        queue_receives=lambda: None,
        receive=lambda work: None,
        prefetch=lambda work: None,
        send=lambda tensor, patch_index: None,
        wait_sends=lambda: None,
    )
    result = PipeFusionImagePatchSchedule(
        patch_latents=patches,
        layout=layout,
        initial_condition=torch.zeros(1, 2, 2),
        first_stage=True,
        last_stage=True,
        condition_reuse=True,
        forward_patch_fn=forward_patch,
        update_last_patch_fn=pipe._pipefusion_update_qwen_patch,
        end_step_fn=lambda index, timestep: indices.append(pipe.scheduler.step_index),
        finalize_fn=lambda: torch.cat(patches, dim=1),
    ).run(timesteps=pipe.scheduler.timesteps[1:], transport=transport, advance_patch=advance_patch)

    assert indices == [2, 3, 4]
    torch.testing.assert_close(result[:, :generated], expected)
    torch.testing.assert_close(result[:, generated:], fixed)
