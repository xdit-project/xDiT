from typing import Optional, Tuple, Union

import torch
import torch.distributed

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler,
    DDPMSchedulerOutput,
)

from xfuser.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_runtime_state,
)
from .register import xFuserSchedulerWrappersRegister
from .base_scheduler import xFuserSchedulerBaseWrapper


@xFuserSchedulerWrappersRegister.register(DDPMScheduler)
class xFuserDDPMSchedulerWrapper(xFuserSchedulerBaseWrapper):
    
    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        *args,
        generator=None,
        **kwargs,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
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
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        return self.module.step(*args, generator, **kwargs)