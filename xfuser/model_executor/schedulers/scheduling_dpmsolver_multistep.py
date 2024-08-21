from typing import Optional, Tuple, Union

import torch
import torch.distributed

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
    SchedulerOutput,
)

from xfuser.core.distributed import get_runtime_state
from .register import xFuserSchedulerWrappersRegister
from .base_scheduler import xFuserSchedulerBaseWrapper


@xFuserSchedulerWrappersRegister.register(DPMSolverMultistepScheduler)
class xFuserDPMSolverMultistepSchedulerWrapper(xFuserSchedulerBaseWrapper):

    @xFuserSchedulerBaseWrapper.check_to_use_naive_step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        if (
            get_runtime_state().patch_mode
            and get_runtime_state().pipeline_patch_idx == 0
            and self.model_outputs[-1] is None
        ):
            self.model_outputs[-1] = torch.zeros(
                [
                    model_output.shape[0],
                    model_output.shape[1],
                    get_runtime_state().pp_patches_start_idx_local[-1],
                    model_output.shape[3],
                ],
                device=model_output.device,
                dtype=model_output.dtype,
            )
        if get_runtime_state().pipeline_patch_idx == 0:
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]

        if (
            get_runtime_state().patch_mode
            and get_runtime_state().pipeline_patch_idx == 0
        ):
            assert len(self.model_outputs) >= 2
            self.model_outputs[-1] = torch.zeros_like(self.model_outputs[-2])
        if get_runtime_state().patch_mode:
            self.model_outputs[-1][
                :,
                :,
                get_runtime_state()
                .pp_patches_start_idx_local[
                    get_runtime_state().pipeline_patch_idx
                ] : get_runtime_state()
                .pp_patches_start_idx_local[get_runtime_state().pipeline_patch_idx + 1],
                :,
            ] = model_output
        else:
            self.model_outputs[-1] = model_output

        #! ORIGIN:
        # for i in range(self.config.solver_order - 1):
        #     self.model_outputs[i] = self.model_outputs[i + 1]
        # self.model_outputs[-1] = model_output
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if (
            self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]
            and variance_noise is None
        ):
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=torch.float32,
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if get_runtime_state().patch_mode:
            model_outputs = []
            for output in self.model_outputs:
                model_outputs.append(
                    output[
                        :,
                        :,
                        get_runtime_state()
                        .pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx
                        ] : get_runtime_state()
                        .pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx + 1
                        ],
                        :,
                    ]
                )
        else:
            model_outputs = self.model_outputs
        #! ---------------------------------------- ADD ABOVE ----------------------------------------

        if (
            self.config.solver_order == 1
            or self.lower_order_nums < 1
            or lower_order_final
        ):
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise
            )
        elif (
            self.config.solver_order == 2
            or self.lower_order_nums < 2
            or lower_order_second
        ):
            prev_sample = self.multistep_dpm_solver_second_order_update(
                model_outputs, sample=sample, noise=noise
            )
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                model_outputs, sample=sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        # * increase step index only when the last pipeline patch is done (or not in patch mode)
        if (
            not get_runtime_state().patch_mode
            or get_runtime_state().pipeline_patch_idx
            == get_runtime_state().num_pipeline_patch - 1
        ):
            self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)
