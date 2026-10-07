"""Flow-matching scheduler that steps the video and audio latents on paired timesteps.

Vendored from Tencent-Hunyuan/Prism ``hymm/diffusion/schedulers/flow_match_pair.py``
for inference only. Behaviour is kept exactly, including that ``step_from_to``
snaps each timestep to the nearest point of the training grid built at
construction (``num_train_timesteps`` points at the config's ``shift``).
"""

import math

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowMatchPairScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        train_sigmas = self._sigmas(num_train_timesteps, shift)
        self.train_sigmas = train_sigmas
        self.train_timesteps = train_sigmas * num_train_timesteps
        self.pair_timesteps = None

    def _sigmas(self, num_steps, shift, device=None, dtype=torch.float32):
        c = self.config
        if c.extra_one_step:
            sigmas = torch.linspace(c.sigma_max, c.sigma_min, num_steps + 1, device=device, dtype=dtype)[:-1]
        else:
            sigmas = torch.linspace(c.sigma_max, c.sigma_min, num_steps, device=device, dtype=dtype)
        if c.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        if c.exponential_shift:
            if c.exponential_shift_mu is None:
                raise RuntimeError("exponential_shift is enabled but exponential_shift_mu is not provided")
            exp_mu = math.exp(float(c.exponential_shift_mu))
            sigmas = exp_mu / (exp_mu + (1 / sigmas - 1))
        else:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        if c.shift_terminal is not None:
            one_minus_z = 1 - sigmas
            sigmas = 1 - one_minus_z / (one_minus_z[-1] / (1 - c.shift_terminal))
        if c.reverse_sigmas:
            sigmas = 1 - sigmas
        return sigmas

    def set_pair_timesteps(self, num_inference_steps: int, visual_shift: float, audio_shift: float):
        """``[num_inference_steps, 2]`` (video, audio) timesteps, each column at its own shift."""
        self.pair_timesteps = (
            torch.stack(
                [
                    self._sigmas(num_inference_steps, float(visual_shift)),
                    self._sigmas(num_inference_steps, float(audio_shift)),
                ],
                dim=1,
            )
            * self.config.num_train_timesteps
        )
        return self.pair_timesteps

    def timestep_to_sigma(self, timestep):
        idx = torch.argmin((self.train_timesteps - torch.tensor(float(timestep))).abs())
        return self.train_sigmas[idx]

    def step_from_to(self, model_output, timestep_from, timestep_to, sample):
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            sigma_to = torch.tensor(1.0 if (self.config.inverse_timesteps or self.config.reverse_sigmas) else 0.0)
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        return sample + model_output * (sigma_to - sigma_from)
