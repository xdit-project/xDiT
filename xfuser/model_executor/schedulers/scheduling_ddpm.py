from typing import Optional, Tuple, Union

import torch

from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler,
    DDPMSchedulerOutput,
)
from diffusers.utils.torch_utils import randn_tensor

from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from .register import xFuserSchedulerWrappersRegister
from .base_scheduler import xFuserSchedulerBaseWrapper


def _gather_sequence_parallel_rows(local: torch.Tensor) -> torch.Tensor:
    """Rebuild the whole latent from the rows each sequence-parallel rank holds.

    A rank holds, for every pipeline patch, its share of that patch's rows,
    concatenated along dim -2 (see DiTRuntimeState._calc_patches_metadata).
    """
    state = get_runtime_state()
    shards = get_sp_group().all_gather(local.contiguous(), separate_tensors=True)
    bounds = state.pp_patches_start_idx_local
    return torch.cat(
        [
            shard[..., bounds[patch] : bounds[patch + 1], :]
            for patch in range(state.num_pipeline_patch)
            for shard in shards
        ],
        dim=-2,
    )


def _local_rows(full: torch.Tensor) -> torch.Tensor:
    """The rows of ``full`` this sequence-parallel rank holds."""
    return torch.cat(
        [full[..., start:end, :] for start, end in get_runtime_state().pp_patches_start_end_idx_global],
        dim=-2,
    )


def _step_with_noise(scheduler, model_output, timestep, sample, noise, return_dict):
    """Diffusers' DDPM update with an explicit variance-noise tensor.

    DDPMScheduler.step has no variance_noise argument. Keep its posterior update
    here so a PipeFusion patch can supply its own noise without changing any
    Diffusers module globals. The scheduler still owns the timestep, variance,
    and thresholding calculations.
    """
    prev_t = scheduler.previous_timestep(timestep)
    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ("learned", "learned_range"):
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for the DDPMScheduler."
        )

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    variance = scheduler._get_variance(timestep, predicted_variance=predicted_variance)
    if scheduler.variance_type == "fixed_small_log":
        variance = variance * noise
    elif scheduler.variance_type == "learned_range":
        variance = torch.exp(0.5 * variance) * noise
    else:
        variance = variance**0.5 * noise
    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample, pred_original_sample)
    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


@xFuserSchedulerWrappersRegister.register(DDPMScheduler)
class xFuserDDPMSchedulerWrapper(xFuserSchedulerBaseWrapper):
    def __init__(self, module: DDPMScheduler):
        super().__init__(module)
        self._step_noise = None

    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """DDPMScheduler.step on a latent split across sequence-parallel ranks.

        DDPM adds fresh noise of ``model_output.shape`` at every step. On a
        rank's share of the rows, every rank would draw the same noise from its
        identically seeded generator, so the shares would get copies of one
        noise patch instead of the slices of one full-size draw, and the result
        would drift from a single-device run. Step the whole latent instead and
        keep this rank's rows, so the generator advances exactly as it does on
        one device.

        PipeFusion's patch mode steps one patch at a time; see _step_patch.
        """
        if get_runtime_state().patch_mode:
            return self._step_patch(model_output, timestep, sample, generator, return_dict)
        if get_sequence_parallel_world_size() == 1:
            return self.module.step(model_output, timestep, sample, generator, return_dict=return_dict)

        output = self.module.step(
            _gather_sequence_parallel_rows(model_output),
            timestep,
            _gather_sequence_parallel_rows(sample),
            generator,
            return_dict=True,
        )
        prev_sample = _local_rows(output.prev_sample)
        pred_original_sample = _local_rows(output.pred_original_sample)
        if not return_dict:
            return (prev_sample, pred_original_sample)
        return DDPMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def _step_patch(self, model_output, timestep, sample, generator, return_dict):
        """DDPMScheduler.step on one PipeFusion patch, with its rows of one full-size noise draw.

        Stepping a patch alone would draw noise of the patch's shape, unrelated to
        the full-size draw a single device makes (and identical on every
        sequence-parallel rank). Draw the whole latent's noise at the first patch
        of each step and give each patch, on each rank, its own rows.
        """
        if timestep <= 0:
            # Diffusers adds no noise at the last step and draws none.
            self._step_noise = None
            return self.module.step(model_output, timestep, sample, generator, return_dict=return_dict)

        state = get_runtime_state()
        if state.pipeline_patch_idx == 0:
            full_shape = (
                sample.shape[0],
                sample.shape[1],
                state.input_config.height // state.vae_scale_factor,
                state.input_config.width // state.vae_scale_factor,
            )
            self._step_noise = randn_tensor(
                full_shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        start, end = state.pp_patches_start_end_idx_global[state.pipeline_patch_idx]
        noise = self._step_noise[..., start:end, :]
        # The cache belongs to this scheduler's current step and is replaced at
        # the first patch of every request/step. Release it after the last patch.
        if state.pipeline_patch_idx == len(state.pp_patches_start_end_idx_global) - 1:
            self._step_noise = None
        if noise.shape != sample.shape:
            raise RuntimeError(f"DDPM noise for a patch has shape {tuple(noise.shape)}, expected {tuple(sample.shape)}")
        return _step_with_noise(self.module, model_output, timestep, sample, noise, return_dict)
