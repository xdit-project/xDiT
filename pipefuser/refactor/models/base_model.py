from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Type, Union

import torch.nn as nn
from pipefuser.refactor.config.config import ParallelConfig, RuntimeConfig
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.refactor.layers import *
from pipefuser.logger import init_logger

logger = init_logger(__name__)

# class PipeFuserModelBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):
class PipeFuserModelBaseWrapper(nn.Module, PipeFuserBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        module: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        # super().__init__(
        #     module=module,
        #     parallel_config=parallel_config, 
        #     runtime_config=runtime_config
        # )
        super().__init__()
        super(nn.Module, self).__init__(
            module=module,
            parallel_config=parallel_config, 
            runtime_config=runtime_config
        )

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return getattr(self.module, name)

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
                    wrapper = PipeFuserLayerWrappersRegister.get_wrapper(submodule)
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
