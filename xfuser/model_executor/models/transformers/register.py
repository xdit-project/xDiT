from typing import Dict, Type
import torch
import torch.nn as nn

from xfuser.logger import init_logger
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)

logger = init_logger(__name__)


class xFuserTransformerWrappersRegister:
    _XFUSER_TRANSFORMER_MAPPING: Dict[
        Type[nn.Module], Type[xFuserTransformerBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_transformer_class: Type[nn.Module]):
        def decorator(xfuser_transformer_class: Type[nn.Module]):
            if not issubclass(
                xfuser_transformer_class, xFuserTransformerBaseWrapper
            ):
                raise ValueError(
                    f"{xfuser_transformer_class.__class__.__name__} is not "
                    f"a subclass of xFuserTransformerBaseWrapper"
                )
            cls._XFUSER_TRANSFORMER_MAPPING[origin_transformer_class] = (
                xfuser_transformer_class
            )
            return xfuser_transformer_class

        return decorator

    @classmethod
    def get_wrapper(cls, transformer: nn.Module) -> xFuserTransformerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_transformer_class,
            wrapper_class,
        ) in cls._XFUSER_TRANSFORMER_MAPPING.items():
            if origin_transformer_class is None:
                continue
            if isinstance(transformer, origin_transformer_class):
                if (
                    candidate is None
                    or origin_transformer_class == transformer.__class__
                    or issubclass(origin_transformer_class, candidate_origin)
                ):
                    candidate_origin = origin_transformer_class
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(
                f"Transformer class {transformer.__class__.__name__} "
                f"is not supported by xFuser"
            )
        else:
            return candidate
