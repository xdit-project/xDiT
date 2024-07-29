from typing import Union, Tuple, Optional
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput

from pipefuser.model_executor.schedulers import (
    PipeFuserSchedulerWrappersRegister,
    PipeFuserSchedulerBaseWrapper
)


@PipeFuserSchedulerWrappersRegister.register(DDIMScheduler)
class PipeFuserDDIMSchedulerWrapper(PipeFuserSchedulerBaseWrapper):
    def __init__(
        self,
        scheduler: DDIMScheduler,
    ):
        super().__init__(module=scheduler,)

    def step(
        self,
        *args,
        **kwargs
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        patch_idx = kwargs.pop("patch_idx", None)
        return self.module.step(*args, **kwargs)