from abc import abstractmethod, ABCMeta
from typing import List

import torch
import torch.nn as nn

from pipefuser.config.config import InputConfig, ParallelConfig, RuntimeConfig
from pipefuser.model_executor.base_wrapper import PipeFuserBaseWrapper


class PipeFuserLayerBaseWrapper(nn.Module, PipeFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(self, module: nn.Module):
        super().__init__()
        super(nn.Module, self).__init__(module=module)
        self.activation_cache = None

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

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
