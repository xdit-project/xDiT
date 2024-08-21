from typing import Optional, Tuple, Union

import torch
import torch.distributed

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim_cogvideox import (
    CogVideoXDDIMScheduler,
    DDIMSchedulerOutput,
)

from xfuser.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_runtime_state,
)
from .register import xFuserSchedulerWrappersRegister
from .base_scheduler import xFuserSchedulerBaseWrapper


@xFuserSchedulerWrappersRegister.register(CogVideoXDDIMScheduler)
class xFuserCogVideoXDDIMSchedulerWrapper(xFuserSchedulerBaseWrapper):

    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        *args,
        **kwargs,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        return self.module.step(*args, **kwargs)
