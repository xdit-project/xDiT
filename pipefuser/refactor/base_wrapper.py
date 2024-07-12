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
        self.forward_round_counter = 0
        self.current_patch_idx = 0

    def __getattr__(self, name: str):
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(f"module {type(self.module).__name__} has no "
                                 f"attribute {name}")

    def __str__(self):
        return str(self.module)

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

    @staticmethod
    def forward_check_condition(func):
        @wraps(func)
        def check_condition_fn(self, *args, **kwargs):
            if self.input_config is None:
                raise ValueError("InputConfig is not set, please set it before "
                                 "calling forward")
            return func(self, *args, **kwargs)
        return check_condition_fn

    @abstractmethod
    def set_input_config(self, input_config: InputConfig):
        pass