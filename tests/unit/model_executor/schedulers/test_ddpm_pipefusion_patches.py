"""PipeFusion's patch-wise DDPM steps must add the noise a single device adds.

DDPM adds fresh generator noise at every step. PipeFusion steps the latent one
patch at a time, and each rank of a sequence-parallel group holds only its rows of
each patch. Drawing noise per patch gave the patches unrelated draws instead of the
slices of the one full-size draw diffusers makes, and gave every sequence-parallel
rank the same noise. Stepping every patch on every simulated rank must reproduce
diffusers' full-latent step exactly.
"""

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

from xfuser.model_executor.schedulers import base_scheduler, scheduling_ddpm
from xfuser.model_executor.schedulers.scheduling_ddpm import xFuserDDPMSchedulerWrapper

HEIGHT = WIDTH = 8
STEPS = 3


def _runtime_state(rows_per_patch):
    return SimpleNamespace(
        patch_mode=True,
        pipeline_patch_idx=0,
        input_config=SimpleNamespace(height=HEIGHT, width=WIDTH),
        vae_scale_factor=1,
        pp_patches_start_end_idx_global=rows_per_patch,
    )


@pytest.fixture
def pipefusion(monkeypatch):
    # Two pipeline stages, so the wrapper's own step runs instead of diffusers'.
    monkeypatch.setattr(base_scheduler, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_scheduler, "get_sequence_parallel_world_size", lambda: 1)

    def use(state):
        monkeypatch.setattr(scheduling_ddpm, "get_runtime_state", lambda: state)

    return use


def _inputs():
    generator = torch.Generator().manual_seed(0)
    model_outputs = [torch.randn(1, 4, HEIGHT, WIDTH, generator=generator) for _ in range(STEPS)]
    sample = torch.randn(1, 4, HEIGHT, WIDTH, generator=generator)
    return model_outputs, sample


def _expected():
    scheduler = DDPMScheduler()
    scheduler.set_timesteps(STEPS)
    model_outputs, sample = _inputs()
    generator = torch.Generator().manual_seed(1)
    for timestep, model_output in zip(scheduler.timesteps, model_outputs):
        sample = scheduler.step(model_output, timestep, sample, generator=generator).prev_sample
    return sample


def _patchwise(pipefusion, rows_per_patch):
    """Step each patch of one rank, as PipeFusion's last stage does."""
    state = _runtime_state(rows_per_patch)
    pipefusion(state)
    scheduler = DDPMScheduler()
    scheduler.set_timesteps(STEPS)
    wrapper = xFuserDDPMSchedulerWrapper(scheduler)
    model_outputs, sample = _inputs()
    patches = [sample[..., start:end, :] for start, end in rows_per_patch]
    generator = torch.Generator().manual_seed(1)
    for timestep, model_output in zip(scheduler.timesteps, model_outputs):
        for patch, (start, end) in enumerate(rows_per_patch):
            state.pipeline_patch_idx = patch
            patches[patch] = wrapper.step(
                model_output[..., start:end, :], timestep, patches[patch], generator=generator
            ).prev_sample
    return {rows: patch for rows, patch in zip(map(tuple, rows_per_patch), patches)}


def test_patches_get_slices_of_the_single_device_noise(pipefusion):
    expected = _expected()

    patches = _patchwise(pipefusion, [[0, 4], [4, 8]])

    for (start, end), patch in patches.items():
        torch.testing.assert_close(patch, expected[..., start:end, :])


def test_sequence_parallel_ranks_get_their_own_rows_of_the_noise(pipefusion):
    expected = _expected()

    # Two patches of four rows, each split between two sequence-parallel ranks.
    for rows_per_patch in ([[0, 2], [4, 6]], [[2, 4], [6, 8]]):
        for (start, end), patch in _patchwise(pipefusion, rows_per_patch).items():
            torch.testing.assert_close(patch, expected[..., start:end, :])
