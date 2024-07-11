from abc import abstractmethod, ABCMeta
from typing import List

import torch
import torch.nn as nn

from pipefuser.refactor.config.config import ParallelConfig, RuntimeConfig
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper

class PipeFuserLayerBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):
    
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
        self.activation_cache = None
        self.num_pipeline_patch = \
            self.parallel_config.pp_config.num_pipeline_patch

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
