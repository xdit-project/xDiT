from abc import  ABCMeta
from functools import wraps
from typing import Any

from xfuser.distributed import ps, rs


class xFuserBaseWrapper(metaclass=ABCMeta):

    def __init__(self, module: Any,):
        self.module = module
        self.module_type = type(module)

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
            if (
                ps.get_pipeline_parallel_world_size() == 1
                and ps.get_classifier_free_guidance_world_size() == 1
                and ps.get_sequence_parallel_world_size() == 1
            ):
                return func(self, *args, **kwargs)
            if not rs.get_runtime_state().is_ready():
                raise ValueError(
                    "Runtime state is not ready, please call RuntimeState.set_input_parameters "
                    "before calling forward"
                )
            return func(self, *args, **kwargs)
        return check_condition_fn