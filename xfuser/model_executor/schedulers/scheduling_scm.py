from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.distributed

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
from diffusers.schedulers import SCMScheduler

from xfuser.core.distributed import get_runtime_state
from .register import xFuserSchedulerWrappersRegister
from .base_scheduler import xFuserSchedulerBaseWrapper


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->SCM
class SCMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

@xFuserSchedulerWrappersRegister.register(SCMScheduler)
class xFuserSCMSchedulerWrapper(xFuserSchedulerBaseWrapper):

    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float,
        sample: torch.FloatTensor,
        generator: torch.Generator = None,
        return_dict: bool = True,
    ) -> Union[SCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_scm.SCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.SCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_scm.SCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # 2. compute alphas, betas
        t = self.timesteps[self.step_index + 1]
        s = self.timesteps[self.step_index]

        # 4. Different Parameterization:
        parameterization = self.config.prediction_type

        if parameterization == "trigflow":
            pred_x0 = torch.cos(s) * sample - torch.sin(s) * model_output
        else:
            raise ValueError(f"Unsupported parameterization: {parameterization}")

        # 5. Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
            latent_height = get_runtime_state().input_config.height // get_runtime_state().vae_scale_factor // get_runtime_state().backbone_patch_size
            latent_width = get_runtime_state().input_config.width // get_runtime_state().vae_scale_factor // get_runtime_state().backbone_patch_size
            b, c, h, w = model_output.shape
            noise = randn_tensor((b, c, latent_height, latent_width), device=model_output.device, generator=generator)

            noise_list = [
                noise[:, :, start_idx:end_idx, :]
                for start_idx, end_idx in get_runtime_state().pp_patches_start_end_idx_global
            ]
            noise = torch.cat(noise_list, dim=-2) * self.config.sigma_data

            # noise = (
            #     randn_tensor(model_output.shape, device=model_output.device, generator=generator)
            #     * self.config.sigma_data
            # )
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------
            prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise
        else:
            prev_sample = pred_x0

        self._step_index += 1

        if not return_dict:
            return (prev_sample, pred_x0)

        return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
