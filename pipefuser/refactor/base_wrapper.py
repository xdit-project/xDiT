from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import Any, Optional
from torch import nn

from pipefuser.refactor.config.config import InputConfig, ParallelConfig, RuntimeConfig


class PipeFuserBaseWrapper(metaclass=ABCMeta):

    def __init__(
        self, 
        module: Any,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        self.module = module
        self.module_type = type(module)
        self.parallel_config = parallel_config
        self.runtime_config = runtime_config
        self.input_config: InputConfig = None

    def __getattr__(self, name: str):
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(f"module {type(self.module).__name__} has no "
                                 f"attribute {name}")

    def __str__(self):
        return str(self.module)

    @staticmethod
    def forward_check_condition(func):
        @wraps(func)
        def check_condition_fn(self, *args, **kwargs):
            if self.input_config is None:
                raise ValueError("InputConfig is not set, please set it before "
                                 "calling forward")
            if (self.input_config.height % 
                self.parallel_config.pp_config.num_pipeline_patch != 0):
                raise ValueError(
                    f"height; {self.input_config.height} must be divisible by "
                    f"num_pipeline_patch: "
                    f"{self.parallel_config.pp_config.num_pipeline_patch}"
                )
            return func(self, *args, **kwargs)
        return check_condition_fn

    @abstractmethod
    def set_patched_mode(self, patched: bool):
        pass

    @abstractmethod
    def reset_patch_idx(self):
        pass

    @abstractmethod
    def set_input_config(self, input_config: InputConfig):
        pass