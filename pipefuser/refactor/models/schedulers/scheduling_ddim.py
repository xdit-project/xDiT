import torch
from typing import Union, Tuple, Optional
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput

from pipefuser.refactor.config.config import ParallelConfig, RuntimeConfig
from pipefuser.refactor.models.schedulers import (
    PipeFuserSchedulerWrappersRegister
)
from pipefuser.refactor.models.base_model import (
    PipeFuserModelBaseWrapper
)

@PipeFuserSchedulerWrappersRegister.register(DDIMScheduler)
class PipeFuserDDIMSchedulerWrapper(PipeFuserModelBaseWrapper):
    def __init__(
        self,
        scheduler: DDIMScheduler,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            module=scheduler,
            parallel_config=parallel_config,
            runtime_config=runtime_config
        )

    def step(
        self,
        *args,
        **kwargs
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        patch_idx = kwargs.pop("patch_idx", None)
        return self.module.step(*args, **kwargs)