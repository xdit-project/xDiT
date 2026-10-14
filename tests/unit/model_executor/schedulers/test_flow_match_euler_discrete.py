from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from xfuser.model_executor.schedulers import (
    xFuserFlowMatchEulerDiscreteSchedulerWrapper,
)
from xfuser.model_executor.schedulers import (
    scheduling_flow_match_euler_discrete as flow_match,
)


def test_default_flow_match_step_matches_diffusers(monkeypatch):
    monkeypatch.setattr(
        flow_match,
        "get_runtime_state",
        lambda: SimpleNamespace(
            patch_mode=False,
            pipeline_patch_idx=0,
            num_pipeline_patch=1,
        ),
    )
    reference = FlowMatchEulerDiscreteScheduler()
    wrapped = xFuserFlowMatchEulerDiscreteSchedulerWrapper(FlowMatchEulerDiscreteScheduler())
    reference.set_timesteps(4)
    wrapped.set_timesteps(4)
    sample = torch.full((1, 4, 2, 2), 0.5)
    model_output = torch.full_like(sample, 0.25)
    reference_generator = torch.Generator().manual_seed(0)
    wrapped_generator = torch.Generator().manual_seed(0)

    expected = reference.step(
        model_output,
        reference.timesteps[0],
        sample,
        s_churn=0.0,
        generator=reference_generator,
    ).prev_sample
    actual = wrapped.step.__wrapped__(
        wrapped,
        model_output,
        wrapped.timesteps[0],
        sample,
        s_churn=0.0,
        generator=wrapped_generator,
    ).prev_sample

    torch.testing.assert_close(actual, expected)


def test_nonzero_churn_honors_sigma_bounds(monkeypatch):
    monkeypatch.setattr(
        flow_match,
        "get_runtime_state",
        lambda: SimpleNamespace(
            patch_mode=False,
            pipeline_patch_idx=0,
            num_pipeline_patch=1,
        ),
    )
    sample = torch.full((1, 4, 2, 2), 0.5)
    model_output = torch.full_like(sample, 0.25)

    def step_with_bounds(s_tmin: float) -> torch.Tensor:
        wrapped = xFuserFlowMatchEulerDiscreteSchedulerWrapper(FlowMatchEulerDiscreteScheduler())
        wrapped.set_timesteps(4)
        return wrapped.step.__wrapped__(
            wrapped,
            model_output,
            wrapped.timesteps[0],
            sample,
            s_churn=0.5,
            s_tmin=s_tmin,
            generator=torch.Generator().manual_seed(0),
        ).prev_sample

    in_bounds = step_with_bounds(0.0)
    out_of_bounds = step_with_bounds(float("inf"))

    assert not torch.allclose(in_bounds, out_of_bounds)


def test_patch_steps_match_full_flow_match_step(monkeypatch):
    state = SimpleNamespace(
        patch_mode=True,
        pipeline_patch_idx=0,
        num_pipeline_patch=2,
    )
    monkeypatch.setattr(flow_match, "get_runtime_state", lambda: state)
    reference = FlowMatchEulerDiscreteScheduler()
    wrapped = xFuserFlowMatchEulerDiscreteSchedulerWrapper(FlowMatchEulerDiscreteScheduler())
    reference.set_timesteps(4)
    wrapped.set_timesteps(4)
    generator = torch.Generator().manual_seed(7)
    full_sample = torch.randn((1, 4, 4, 2), generator=generator)
    patch_samples = list(full_sample.chunk(2, dim=2))

    for timestep in reference.timesteps:
        model_output = torch.randn(full_sample.shape, generator=generator)
        expected = reference.step(
            model_output,
            timestep,
            full_sample,
        ).prev_sample
        patch_outputs = model_output.chunk(2, dim=2)
        updated_patches = []
        for patch_index, (patch_output, patch_sample) in enumerate(zip(patch_outputs, patch_samples)):
            state.pipeline_patch_idx = patch_index
            updated_patches.append(
                wrapped.step.__wrapped__(
                    wrapped,
                    patch_output,
                    timestep,
                    patch_sample,
                ).prev_sample
            )
        full_sample = expected
        patch_samples = updated_patches
        torch.testing.assert_close(torch.cat(patch_samples, dim=2), full_sample)
        assert wrapped.step_index == reference.step_index
