from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import List

from diffusers.schedulers import SchedulerMixin
from xfuser.distributed import (
    get_pipeline_parallel_world_size, get_sequence_parallel_world_size
)
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper

class xFuserSchedulerBaseWrapper(xFuserBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        module: SchedulerMixin,
    ):
        super().__init__(module=module,)

    def __setattr__(self, name, value):
        if name == 'module':
            super().__setattr__(name, value)
        elif (hasattr(self, 'module') and 
              self.module is not None and 
              hasattr(self.module, name)):
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @staticmethod
    def check_to_use_naive_step(func):
        @wraps(func)
        def check_naive_step_fn(self, *args, **kwargs):
            if (
                get_pipeline_parallel_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            ):
                return self.module.step(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return check_naive_step_fn
