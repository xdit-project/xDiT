from typing import Dict, Type
import torch
import torch.nn as nn

from pipefuser.logger import init_logger
from pipefuser.models.transformers.base_transformer import (
    PipeFuserTransformerBaseWrapper,
)

logger = init_logger(__name__)


class PipeFuserTransformerWrappersRegister:
    _PIPEFUSER_TRANSFORMER_MAPPING: Dict[
        Type[nn.Module], Type[PipeFuserTransformerBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_transformer_class: Type[nn.Module]):
        def decorator(pipefuser_transformer_class: Type[nn.Module]):
            if not issubclass(
                pipefuser_transformer_class, PipeFuserTransformerBaseWrapper
            ):
                raise ValueError(
                    f"{pipefuser_transformer_class.__class__.__name__} is not "
                    f"a subclass of PipeFuserTransformerBaseWrapper"
                )
            cls._PIPEFUSER_TRANSFORMER_MAPPING[origin_transformer_class] = (
                pipefuser_transformer_class
            )
            return pipefuser_transformer_class

        return decorator

    @classmethod
    def get_wrapper(cls, transformer: nn.Module) -> PipeFuserTransformerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_transformer_class,
            wrapper_class,
        ) in cls._PIPEFUSER_TRANSFORMER_MAPPING.items():
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
                f"is not supported by PipeFuser"
            )
        else:
            return candidate
