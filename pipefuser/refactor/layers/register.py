from typing import Dict, Type
import torch
import torch.nn as nn

from pipefuser.logger import init_logger
from pipefuser.refactor.layers.base_layer import PipeFuserLayerBaseWrapper

logger = init_logger(__name__)


class PipeFuserLayerWrappersRegister:
    _PIPEFUSER_LAYER_MAPPING: Dict[Type[nn.Module], Type[PipeFuserLayerBaseWrapper]] = (
        {}
    )

    @classmethod
    def register(cls, origin_layer_class: Type[nn.Module]):
        def decorator(pipefusion_layer_wrapper: Type[PipeFuserLayerBaseWrapper]):
            if not issubclass(pipefusion_layer_wrapper, PipeFuserLayerBaseWrapper):
                raise ValueError(
                    f"{pipefusion_layer_wrapper.__class__.__name__} is not a "
                    f"subclass of PipeFuserLayerBaseWrapper"
                )
            cls._PIPEFUSER_LAYER_MAPPING[origin_layer_class] = pipefusion_layer_wrapper
            return pipefusion_layer_wrapper

        return decorator

    @classmethod
    def get_wrapper(cls, layer: nn.Module) -> PipeFuserLayerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_layer_class,
            pipefusion_layer_wrapper,
        ) in cls._PIPEFUSER_LAYER_MAPPING.items():
            if isinstance(layer, origin_layer_class):
                if (
                    (candidate is None and candidate_origin is None)
                    or origin_layer_class == layer.__class__
                    or issubclass(origin_layer_class, candidate_origin)
                ):
                    candidate_origin = origin_layer_class
                    candidate = pipefusion_layer_wrapper

        if candidate is None:
            raise ValueError(
                f"Layer class {layer.__class__.__name__} "
                f"is not supported by PipeFuser"
            )
        else:
            return candidate
