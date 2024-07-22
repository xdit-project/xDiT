# adpated from https://github.com/huggingface/diffusers/blob/v0.29.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py

import torch
from typing import Union, Tuple, Optional
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
    randn_tensor,
)

from legacy.pipefuser.utils import DistriConfig
from legacy.pipefuser.logger import init_logger

logger = init_logger(__name__)


class FlowMatchEulerDiscreteSchedulerPiP(FlowMatchEulerDiscreteScheduler):
    def init(self, distri_config: DistriConfig):
        self.distri_config = distri_config

    def step(
        self, *args, **kwargs
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        batch_idx = kwargs.pop("batch_idx", None)
        tmp_step_index = self._step_index
        output = super().step(*args, **kwargs)
        if batch_idx is not None and batch_idx < self.distri_config.pp_num_patch - 1:
            self._step_index = tmp_step_index
        return output
