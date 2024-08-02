from typing import Dict, Type
import torch
import torch.nn as nn

from xfuser.logger import init_logger
from xfuser.model_executor.layers.base_layer import xFuserLayerBaseWrapper

logger = init_logger(__name__)


class xFuserLayerWrappersRegister:
    _XFUSER_LAYER_MAPPING: Dict[
        Type[nn.Module], Type[xFuserLayerBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_layer_class: Type[nn.Module]):
        def decorator(xfuser_layer_wrapper: Type[xFuserLayerBaseWrapper]):
            if not issubclass(xfuser_layer_wrapper, xFuserLayerBaseWrapper):
                raise ValueError(
                    f"{xfuser_layer_wrapper.__class__.__name__} is not a "
                    f"subclass of xFuserLayerBaseWrapper"
                )
            cls._XFUSER_LAYER_MAPPING[origin_layer_class] = xfuser_layer_wrapper
            return xfuser_layer_wrapper

        return decorator

    @classmethod
    def get_wrapper(cls, layer: nn.Module) -> xFuserLayerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_layer_class,
            xfuser_layer_wrapper,
        ) in cls._XFUSER_LAYER_MAPPING.items():
            if isinstance(layer, origin_layer_class):
                if (
                    (candidate is None and candidate_origin is None)
                    or origin_layer_class == layer.__class__
                    or issubclass(origin_layer_class, candidate_origin)
                ):
                    candidate_origin = origin_layer_class
                    candidate = xfuser_layer_wrapper

        if candidate is None:
            raise ValueError(
                f"Layer class {layer.__class__.__name__} "
                f"is not supported by xFuser"
            )
        else:
            return candidate
