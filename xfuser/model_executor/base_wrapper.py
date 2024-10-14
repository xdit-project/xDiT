from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import Any, List, Optional

from xfuser.core.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.fast_attention import get_fast_attn_enable


class xFuserBaseWrapper(metaclass=ABCMeta):

    def __init__(
        self,
        module: Any,
    ):
        self.module = module
        self.module_type = type(module)

    def __getattr__(self, name: str):
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(
                f"module {type(self.module).__name__} has no " f"attribute {name}"
            )

    def __str__(self):
        return str(self.module)

    @staticmethod
    def forward_check_condition(func):
        @wraps(func)
        def check_condition_fn(self, *args, **kwargs):
            if (
                get_pipeline_parallel_world_size() == 1
                and get_classifier_free_guidance_world_size() == 1
                and get_sequence_parallel_world_size() == 1
                and get_tensor_model_parallel_world_size() == 1
                and get_fast_attn_enable() == False
            ):
                return func(self, *args, **kwargs)
            if not get_runtime_state().is_ready():
                raise ValueError(
                    "Runtime state is not ready, please call RuntimeState.set_input_parameters "
                    "before calling forward"
                )
            return func(self, *args, **kwargs)

        return check_condition_fn
