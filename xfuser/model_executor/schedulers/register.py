from typing import Dict, Type
import torch
import torch.nn as nn

from xfuser.logger import init_logger
from xfuser.model_executor.schedulers.base_scheduler import xFuserSchedulerBaseWrapper

logger = init_logger(__name__)

class xFuserSchedulerWrappersRegister:
    _XFUSER_SCHEDULER_MAPPING: Dict[
        Type[nn.Module], 
        Type[xFuserSchedulerBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_scheduler_class: Type[nn.Module]):
        def decorator(xfuser_scheduler_class: Type[nn.Module]):
            if not issubclass(xfuser_scheduler_class, 
                              xFuserSchedulerBaseWrapper):
                raise ValueError(
                    f"{xfuser_scheduler_class.__class__.__name__} is not "
                    f"a subclass of xFuserSchedulerBaseWrapper"
                )
            cls._XFUSER_SCHEDULER_MAPPING[origin_scheduler_class] = \
                xfuser_scheduler_class
            return xfuser_scheduler_class
        return decorator

    @classmethod
    def get_wrapper(
        cls, 
        scheduler: nn.Module
    ) -> xFuserSchedulerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (origin_scheduler_class,
             wrapper_class) in cls._XFUSER_SCHEDULER_MAPPING.items():
            if isinstance(scheduler, origin_scheduler_class):
                if ((candidate is None and candidate_origin is None) or 
                    origin_scheduler_class == scheduler.__class__ or
                    issubclass(origin_scheduler_class, candidate_origin)):
                    candidate_origin = origin_scheduler_class
                    candidate = wrapper_class

        if candidate is None:
            logger.info(f"Scheduler class {scheduler.__class__.__name__} "
                         f"is not supported by xFuser")
            return None
        else:
            return candidate