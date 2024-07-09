from typing import Dict, Type
import torch
import torch.nn as nn

from pipefuser.logger import init_logger
from pipefuser.modules.parallel_utils.transformer import (
    PipeFuserTransformerBaseWrapper
)

logger = init_logger(__name__)

class PipeFuserTransformerWrappers:
    _PIPEFUSER_TRANSFORMER_MAPPING: Dict[
        Type[nn.Module], 
        Type[PipeFuserTransformerBaseWrapper]
    ]

    @classmethod
    def register(cls, origin_transformer_class: Type[nn.Module]):
        def decorator(pipefusion_transformer_class: Type[nn.Module]):
            if not issubclass(pipefusion_transformer_class, 
                              PipeFuserTransformerBaseWrapper):
                raise ValueError(
                    f"{pipefusion_transformer_class.__class__.__name__} is not "
                    f"a subclass of PipeFuserTransformerBaseWrapper"
                )
            cls._PIPEFUSER_TRANSFORMER_MAPPING[origin_transformer_class] = \
                pipefusion_transformer_class
            return pipefusion_transformer_class
        return decorator

    @classmethod
    def get_wrapper(
        cls, 
        transformer: nn.Module
    ) -> PipeFuserTransformerBaseWrapper:
        for (origin_transformer_class,
             wrapper_class) in cls._PIPEFUSER_TRANSFORMER_MAPPING.items():
             #TODO check if subclass is legally
            if isinstance(transformer, origin_transformer_class):
                return wrapper_class
        raise ValueError(f"Transformer class {transformer.__class__.__name__} "
                         f"is not supported by PipeFuser")