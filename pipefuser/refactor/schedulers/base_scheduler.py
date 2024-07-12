from abc import abstractmethod, ABCMeta

from diffusers.schedulers import SchedulerMixin
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.refactor.config.config import InputConfig, ParallelConfig, RuntimeConfig

class PipeFuserSchedulerBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        module: SchedulerMixin,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            module=module,
            parallel_config=parallel_config, 
            runtime_config=runtime_config
        )

    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
