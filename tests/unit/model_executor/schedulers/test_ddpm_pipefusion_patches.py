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

torch = pytest.importorskip("torch")
DDPMScheduler = pytest.importorskip("diffusers").DDPMScheduler

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


def test_reentrant_scheduler_keeps_its_own_noise_and_generator(pipefusion, monkeypatch):
    """A third-party scheduler callback must not borrow PipeFusion's noise."""
    state = _runtime_state([[0, 4], [4, 8]])
    pipefusion(state)
    scheduler = DDPMScheduler()
    scheduler.set_timesteps(STEPS)
    wrapper = xFuserDDPMSchedulerWrapper(scheduler)
    timestep = scheduler.timesteps[0]
    model_outputs, sample = _inputs()

    unrelated = DDPMScheduler.from_config(scheduler.config)
    unrelated.set_timesteps(STEPS)
    inputs_generator = torch.Generator().manual_seed(42)
    unrelated_sample = torch.randn(1, 4, 4, WIDTH, generator=inputs_generator)
    unrelated_model_output = torch.randn(1, 4, 4, WIDTH, generator=inputs_generator)
    expected_generator = torch.Generator().manual_seed(123)
    actual_generator = torch.Generator().manual_seed(123)
    expected = unrelated.step(unrelated_model_output, timestep, unrelated_sample, generator=expected_generator)
    reentrant_outputs = []
    get_variance = scheduler._get_variance

    def get_variance_with_unrelated_step(*args, **kwargs):
        reentrant_outputs.append(
            unrelated.step(unrelated_model_output, timestep, unrelated_sample, generator=actual_generator)
        )
        return get_variance(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_get_variance", get_variance_with_unrelated_step)
    wrapper.step(
        model_outputs[0][..., :4, :],
        timestep,
        sample[..., :4, :],
        generator=torch.Generator().manual_seed(1),
    )

    (actual,) = reentrant_outputs
    torch.testing.assert_close(actual.prev_sample, expected.prev_sample, rtol=0, atol=0)
    torch.testing.assert_close(actual.pred_original_sample, expected.pred_original_sample, rtol=0, atol=0)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


@pytest.mark.parametrize(
    "variance_type,prediction_type,clip_sample,return_dict,use_generator_list",
    [
        ("fixed_small", "epsilon", True, True, False),
        ("fixed_small_log", "v_prediction", False, False, True),
        ("fixed_large", "sample", True, True, True),
        ("learned", "epsilon", True, False, True),
        ("learned_range", "v_prediction", False, True, False),
    ],
    ids=["fixed-small", "fixed-small-log", "fixed-large", "learned", "learned-range"],
)
def test_patch_outputs_and_rng_match_full_latent_step(
    pipefusion, variance_type, prediction_type, clip_sample, return_dict, use_generator_list
):
    """Cover DDPM branches against Diffusers, including every timestep's RNG state."""
    rows_per_patch = [[0, 3], [3, 8]]
    state = _runtime_state(rows_per_patch)
    pipefusion(state)
    expected_scheduler = DDPMScheduler(
        variance_type=variance_type,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
        clip_sample_range=0.5,
    )
    scheduler = DDPMScheduler.from_config(expected_scheduler.config)
    # Nonuniform custom timesteps exercise previous_timestep as well as t == 0.
    expected_scheduler.set_timesteps(timesteps=[999, 527, 0])
    scheduler.set_timesteps(timesteps=[999, 527, 0])
    wrapper = xFuserDDPMSchedulerWrapper(scheduler)
    batch_size = 2 if use_generator_list else 1
    inputs_generator = torch.Generator().manual_seed(0)
    sample = torch.randn(batch_size, 4, HEIGHT, WIDTH, generator=inputs_generator)
    patches = [sample[..., start:end, :] for start, end in rows_per_patch]
    expected_generators = [torch.Generator().manual_seed(10 + index) for index in range(batch_size)]
    actual_generators = [torch.Generator().manual_seed(10 + index) for index in range(batch_size)]
    expected_generator = expected_generators if use_generator_list else expected_generators[0]
    actual_generator = actual_generators if use_generator_list else actual_generators[0]

    for timestep in scheduler.timesteps:
        model_output = torch.randn(sample.shape, generator=inputs_generator)
        if variance_type == "learned":
            predicted_variance = torch.rand(sample.shape, generator=inputs_generator) + 0.1
            model_output = torch.cat([model_output, predicted_variance], dim=1)
        elif variance_type == "learned_range":
            predicted_variance = torch.rand(sample.shape, generator=inputs_generator) * 2 - 1
            model_output = torch.cat([model_output, predicted_variance], dim=1)

        expected = expected_scheduler.step(
            model_output, timestep, sample, generator=expected_generator, return_dict=True
        )
        for patch, (start, end) in enumerate(rows_per_patch):
            state.pipeline_patch_idx = patch
            actual = wrapper.step(
                model_output[..., start:end, :],
                timestep,
                patches[patch],
                generator=actual_generator,
                return_dict=return_dict,
            )
            if return_dict:
                prev_sample, pred_original_sample = actual.prev_sample, actual.pred_original_sample
            else:
                assert isinstance(actual, tuple)
                prev_sample, pred_original_sample = actual
            torch.testing.assert_close(prev_sample, expected.prev_sample[..., start:end, :], rtol=0, atol=0)
            torch.testing.assert_close(
                pred_original_sample, expected.pred_original_sample[..., start:end, :], rtol=0, atol=0
            )
            # Only the first patch draws; later patches and timestep zero draw none.
            for actual_rng, expected_rng in zip(actual_generators, expected_generators):
                assert torch.equal(actual_rng.get_state(), expected_rng.get_state())
            patches[patch] = prev_sample
        sample = expected.prev_sample
