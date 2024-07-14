from abc import abstractmethod, ABCMeta
from typing import List

import torch
import torch.nn as nn

from pipefuser.refactor.config.config import InputConfig, ParallelConfig, RuntimeConfig
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper

class PipeFuserLayerBaseWrapper(nn.Module ,PipeFuserBaseWrapper, metaclass=ABCMeta):
    
    def __init__(
        self, 
        module: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__()
        super(nn.Module, self).__init__(
            module=module,
            parallel_config=parallel_config, 
            runtime_config=runtime_config
        )
        self.activation_cache = None
        self.num_pipeline_patch = \
            self.parallel_config.pp_config.num_pipeline_patch
        self.patched_mode = False
        self.current_patch_idx = 0

    def set_patched_mode(self, patched: bool):
        self.patched_mode = patched

    def reset_patch_idx(self):
        self.current_patch_idx = 0

    def patch_step(self):
        self.current_patch_idx += 1
        if self.current_patch_idx == \
                self.parallel_config.pp_config.num_pipeline_patch:
            self.current_patch_idx = 0

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
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(f"module {type(self.module).__name__} has no "
                                 f"attribute {name}")

    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
