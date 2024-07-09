from typing import Dict, Type
import torch
import torch.nn as nn

from pipefuser.logger import init_logger
from pipefuser.modules.parallel_utils.models import PipeFuserModelBaseWrapper

logger = init_logger(__name__)

class PipeFuserSchedulerWrappers:
    _PIPEFUSER_SCHEDULER_MAPPING: Dict[
        Type[nn.Module], 
        Type[PipeFuserModelBaseWrapper]
    ]

    @classmethod
    def register(cls, origin_scheduler_class: Type[nn.Module]):
        def decorator(pipefusion_scheduler_class: Type[nn.Module]):
            if not issubclass(pipefusion_scheduler_class, 
                              PipeFuserModelBaseWrapper):
                raise ValueError(
                    f"{pipefusion_scheduler_class.__class__.__name__} is not "
                    f"a subclass of PipeFuserModelBaseWrapper"
                )
            cls._PIPEFUSER_SCHEDULER_MAPPING[origin_scheduler_class] = \
                pipefusion_scheduler_class
            return pipefusion_scheduler_class
        return decorator

    @classmethod
    def get_wrapper(
        cls, 
        transformer: nn.Module
    ) -> PipeFuserModelBaseWrapper:
        candidate = None
        for (origin_scheduler_class,
             wrapper_class) in cls._PIPEFUSER_SCHEDULER_MAPPING.items():
            if isinstance(transformer, origin_scheduler_class):
                if (candidate is None or 
                    origin_scheduler_class == transformer.__class__ or
                    issubclass(origin_scheduler_class, candidate)):
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(f"Transformer class {transformer.__class__.__name__} "
                         f"is not supported by PipeFuser")
        else:
            return candidate