from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Type, Union

import torch.nn as nn
from pipefuser.config.config import ParallelConfig, RuntimeConfig
from pipefuser.modules.parallel_utils.base_wrapper import PipeFuserBaseWrapper
from pipefuser.modules.parallel_utils.layers.register import \
    PipeFuserLayerWrappers
from pipefuser.modules.parallel_utils.layers.base_layer import \
    PipeFuserLayerBaseWrapper
from pipefuser.logger import init_logger

logger = init_logger(__name__)

class PipeFuserModelBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        module: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            module=module,
            parallel_config=parallel_config, 
            runtime_config=runtime_config
        )

    def __getattr__(self, name: str):
        return getattr(self.module, name)

    def __setattr__(self, name: str, value):
        setattr(self.module, name, value)

    def __call__(self, *args, **kwargs):
        if callable(self.module):
            return self.module(*args, **kwargs)
        raise TypeError("Inner 'Transformer' object is not callable")

    def __str__(self):
        return str(self.module)

    def _wrap_layers(
        self, 
        model: Optional[nn.Module] = None, 
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List[str] = [],
        submodule_addition_args: Dict[str, Dict] = {},
    ) -> Union[nn.Module, None]:
        wrap_self_module = False
        if model is None:
            wrap_self_module = True
            model = self.module

        for name, module in model.named_modules():
            if isinstance(module, PipeFuserLayerBaseWrapper):
                continue

            for subname, submodule in module.named_children():
                need_wrap = subname in submodule_name_to_wrap.keys()
                for class_to_wrap in submodule_classes_to_wrap:
                    if isinstance(submodule, class_to_wrap):
                        need_wrap = True
                        break

                if need_wrap:
                    wrapper = PipeFuserLayerWrappers.get_wrapper(submodule)
                    additional_args = submodule_addition_args.get(subname, {})
                    logger.info(f"Wrapping {name}.{subname} in model class: "
                                f"{self.module.__class__.__name__} with "
                                f"{wrapper.__name__}")
                    if additional_args is not {}:
                        setattr(
                            obj=module,
                            name=subname, 
                            value=wrapper(
                                submodule, 
                                self.parallel_config, 
                                self.runtime_config,
                                **additional_args
                            )
                        )
                    else:
                        setattr(
                            obj=module,
                            name=subname, 
                            value=wrapper(
                                submodule, 
                                self.parallel_config, 
                                self.runtime_config,
                            )
                        )
        if wrap_self_module:
            self.module = model
        else:
            return model


    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
