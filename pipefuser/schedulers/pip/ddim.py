# adpated from https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/schedulers/scheduling_ddim.py

import torch
from typing import Union, Tuple, Optional
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput

from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DDIMSchedulerPiP(DDIMScheduler):
    def init(self, distri_config: DistriConfig):
        self.distri_config = distri_config

    def step(self, *args, **kwargs) -> Union[DDIMSchedulerOutput, Tuple]:
        patch_idx = kwargs.pop("patch_idx", None)
        return super().step(*args, **kwargs)
