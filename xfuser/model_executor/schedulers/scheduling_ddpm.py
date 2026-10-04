from typing import Optional, Tuple, Union

import torch

import diffusers.schedulers.scheduling_ddpm as diffusers_ddpm

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


@xFuserSchedulerWrappersRegister.register(diffusers_ddpm.DDPMScheduler)
class xFuserDDPMSchedulerWrapper(xFuserSchedulerBaseWrapper):
    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[diffusers_ddpm.DDPMSchedulerOutput, Tuple]:
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
        return diffusers_ddpm.DDPMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def _step_patch(self, model_output, timestep, sample, generator, return_dict):
        """DDPMScheduler.step on one PipeFusion patch, with its rows of one full-size noise draw.

        Stepping a patch alone would draw noise of the patch's shape, unrelated to
        the full-size draw a single device makes (and identical on every
        sequence-parallel rank). Draw the whole latent's noise at the first patch
        of each step and give each patch, on each rank, its own rows.
        """
        if timestep <= 0:
            # Diffusers adds no noise at the last step and draws none.
            return self.module.step(model_output, timestep, sample, generator, return_dict=return_dict)

        state = get_runtime_state()
        if state.pipeline_patch_idx == 0:
            full_shape = (
                sample.shape[0],
                sample.shape[1],
                state.input_config.height // state.vae_scale_factor,
                state.input_config.width // state.vae_scale_factor,
            )
            self._step_noise = diffusers_ddpm.randn_tensor(
                full_shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        start, end = state.pp_patches_start_end_idx_global[state.pipeline_patch_idx]
        noise = self._step_noise[..., start:end, :]

        def patch_noise(shape, **kwargs):
            if tuple(shape) != tuple(noise.shape):
                raise RuntimeError(f"DDPM noise for a patch has shape {tuple(noise.shape)}, expected {tuple(shape)}")
            return noise

        # DDPMScheduler.step takes no noise argument (unlike DDIM's variance_noise);
        # it draws through the module-level name randn_tensor. Swap that name for
        # the duration of this call so the rest of the step stays diffusers' code.
        # The swap is process-global: any other DDPM step running in this process
        # at the same time would receive this patch's noise.
        draw = diffusers_ddpm.randn_tensor
        diffusers_ddpm.randn_tensor = patch_noise
        try:
            return self.module.step(model_output, timestep, sample, generator, return_dict=return_dict)
        finally:
            diffusers_ddpm.randn_tensor = draw
