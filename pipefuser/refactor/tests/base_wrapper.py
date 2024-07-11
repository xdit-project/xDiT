from abc import abstractmethod, ABCMeta
from typing import Optional
from torch import nn

from pipefuser.refactor.config.config import ParallelConfig, RuntimeConfig


class PipeFuserBaseWrapper(nn.Module, metaclass=ABCMeta):

    def __init__(
        self, 
        module: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__()
        print(22, module)
        self.module = module
        print(24, self.module)
        self.parallel_config = parallel_config
        self.runtime_config = runtime_config
        self.forward_round_counter = 0
        self.current_patch_idx = 0

    def __getattr__(self, name: str):
        return getattr(self.module, name)
        
    def __call__(self, *args, **kwargs):
        if callable(self.module):
            return self.module(*args, **kwargs)
        raise TypeError("Inner 'Transformer' object is not callable")

    def __str__(self):
        return str(self.module)

    def set_config(
        self, 
        *,
        parallel_config: Optional[ParallelConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
    ):
        self.parallel_config = parallel_config or self.parallel_config
        self.runtime_config = runtime_config or self.runtime_config

    def reset_counter(self):
        self.forward_round_counter = 0
        self.current_patch_idx = 0

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def patch_step(self):
        self.current_patch_idx += 1
        if self.current_patch_idx == \
                self.parallel_config.pp_config.num_pipeline_patch:
            self.current_patch_idx = 0
            self.forward_round_counter += 1

    def round_step(self):
        self.forward_round_counter += 1
        self.current_patch_idx = 0

    def in_warmup_stage(self) -> bool:
        return self.forward_round_counter < self.runtime_config.warmup_steps

    # def __getattr__(self, name: str):
    #     module = super().__getattr__("module")
    #     # module = self.__dict__["module"]
    #     return getattr(module, name)

    # def __call__(self, *args, **kwargs):
    #     if callable(self.module):
    #         return self.module(*args, **kwargs)
    #     raise TypeError("Inner 'Transformer' object is not callable")

    # def __str__(self):
    #     return str(self.module)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass