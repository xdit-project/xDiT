from typing import Dict, Type
import torch
import torch.nn as nn

from pipefuser.logger import init_logger
from pipefuser.modules.parallel_utils.layers.base_layer import (
    PipeFuserLayerBaseWrapper
)

logger = init_logger(__name__)

class PipeFuserLayerWrappers:
    _PIPEFUSER_LAYER_MAPPING: Dict[
        Type[nn.Module], 
        Type[PipeFuserLayerBaseWrapper]
    ]

    @classmethod
    def register(cls, origin_layer_class: Type[nn.Module]):
        def decorator(pipefusion_layer_wrapper: Type[PipeFuserLayerBaseWrapper]):
            if not issubclass(pipefusion_layer_wrapper,
                              PipeFuserLayerBaseWrapper):
                raise ValueError(
                    f"{pipefusion_layer_wrapper.__class__.__name__} is not a "
                    f"subclass of PipeFuserLayerBaseWrapper")
            cls._PIPEFUSER_LAYER_MAPPING[origin_layer_class] = \
                pipefusion_layer_wrapper
            return pipefusion_layer_wrapper
        return decorator

    @classmethod
    def get_wrapper(
        cls, 
        layer: nn.Module
    ) -> PipeFuserLayerBaseWrapper:
        for (origin_layer_class, 
             pipefusion_layer_wrapper) in cls._PIPEFUSER_LAYER_MAPPING.items():
             #TODO check if subclass is legally
            if isinstance(layer, origin_layer_class):
                return pipefusion_layer_wrapper
        raise ValueError(f"Layer class {layer.__class__.__name__} "
                         f"is not supported by PipeFuser")