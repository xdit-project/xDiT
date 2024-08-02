from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Type, Union

import torch.nn as nn
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.model_executor.layers import *
from xfuser.distributed import ps
from xfuser.logger import init_logger

logger = init_logger(__name__)


# class xFuserModelBaseWrapper(xFuserBaseWrapper, metaclass=ABCMeta):
class xFuserModelBaseWrapper(nn.Module, xFuserBaseWrapper, metaclass=ABCMeta):
    wrapped_layers: List[xFuserLayerBaseWrapper]

    def __init__(self, module: nn.Module):
        super().__init__()
        super(nn.Module, self).__init__(module=module,)

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(
                f"module {type(self.module).__name__} has no " f"attribute {name}"
            )

    def reset_activation_cache(self):
        for layer in self.wrapped_layers:
            if hasattr(layer, "activation_cache"):
                layer.activation_cache = None
            else:
                logger.info(
                    f"layer {type(layer)} has no attribute "
                    f"activation_cache, do not need to reset"
                )

    def _wrap_layers(
        self,
        model: Optional[nn.Module] = None,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List[str] = [],
        submodule_addition_args: Dict[str, Dict] = {},
    ) -> Union[nn.Module, None]:
        wrapped_layers = []
        wrap_self_module = False
        if model is None:
            wrap_self_module = True
            model = self.module

        for name, module in model.named_modules():
            if isinstance(module, xFuserLayerBaseWrapper):
                continue

            for subname, submodule in module.named_children():
                need_wrap = subname in submodule_name_to_wrap
                for class_to_wrap in submodule_classes_to_wrap:
                    if isinstance(submodule, class_to_wrap):
                        need_wrap = True
                        break

                if need_wrap:
                    wrapper = xFuserLayerWrappersRegister.get_wrapper(submodule)
                    additional_args = submodule_addition_args.get(subname, {})
                    logger.info(
                        f"[RANK {ps.get_world_group().rank}] "
                        f"Wrapping {name}.{subname} in model class "
                        f"{model.__class__.__name__} with "
                        f"{wrapper.__name__}"
                    )
                    if additional_args is not {}:
                        setattr(
                            module,
                            subname,
                            wrapper(
                                submodule,
                                **additional_args,
                            ),
                        )
                    else:
                        setattr(
                            module,
                            subname,
                            wrapper(submodule),
                        )
                    # if isinstance(getattr(module, subname), xFuserPatchEmbedWrapper):
                    wrapped_layers.append(getattr(module, subname))
        self.wrapped_layers = wrapped_layers
        if wrap_self_module:
            self.module = model
        else:
            return model

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
