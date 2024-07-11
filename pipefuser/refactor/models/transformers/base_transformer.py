from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import Dict, List, Optional, Type
import torch
import torch.nn as nn

from pipefuser.refactor.config.config import ParallelConfig, InputConfig, RuntimeConfig
from pipefuser.refactor.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)
from pipefuser.logger import init_logger
from pipefuser.refactor.models.base_model import PipeFuserModelBaseWrapper

logger = init_logger(__name__)

class PipeFuserTransformerBaseWrapper(PipeFuserModelBaseWrapper, metaclass=ABCMeta):
    # transformer: original transformer model (for example Transformer2DModel)
    def __init__(
        self, 
        transformer: nn.Module, 
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
        input_config: Optional[InputConfig] = None,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ):
        self.input_config = input_config
        transformer = self._convert_transformer_for_pipeline(
            transformer,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args
        )
        print(41, transformer)
        super().__init__(
            module=transformer,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        print(40, self.module)


    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config

    def _convert_transformer_for_pipeline(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ) -> nn.Module:
        if get_pipeline_parallel_world_size() == 1:
            return transformer
        else:
            transformer = self._split_transformer_blocks(transformer)
            return self._wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=submodule_classes_to_wrap,
                submodule_name_to_wrap=submodule_name_to_wrap,
                submodule_addition_args=submodule_addition_args,
            )


    def _split_transformer_blocks(self, transformer: nn.Module):
        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                f"'{transformer.__class__.__name__}' object has no attribute "
                f"'transformer_blocks'. To use pipeline parallelism with"
                f"object {transformer.__class__.__name__}, please implement "
                f"custom _split_transformer_blocks method in derived class"
            )

        # transformer layer split
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        if self.parallel_config.pp_config.attn_layer_num_for_pp is not None:
            attn_layer_num_for_pp = \
                self.parallel_config.pp_config.attn_layer_num_for_pp
            assert (sum(attn_layer_num_for_pp) ==
                    len(transformer.transformer_blocks)), (
                        "Sum of attn_layer_num_for_pp should be equal to the "
                        "number of transformer blocks") 
            if pp_rank == 0:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    :attn_layer_num_for_pp[0]
                ]
            else:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    sum(attn_layer_num_for_pp[: pp_rank-1]):
                    sum(attn_layer_num_for_pp[: pp_rank])
                ]
        else:
            num_blocks_per_stage = (
                len(transformer.transformer_blocks) + pp_world_size - 1
            ) // pp_world_size
            start_idx = pp_rank * num_blocks_per_stage
            end_idx = min((pp_rank + 1) * num_blocks_per_stage,
                          len(transformer.transformer_blocks))
            transformer.transformer_blocks = transformer.transformer_blocks[
                start_idx:end_idx
            ]
        # position embedding
        if pp_rank != 1:
            transformer.pos_embed = None
        return transformer

    
    @staticmethod
    def forward_check_condition(func):
        @wraps(func)
        def check_condition_fn(self, *args, **kwargs):
            if self.input_config is None:
                raise ValueError("InputConfig is not set, please set it before "
                                 "calling forward")
            return func(self, *args, **kwargs)
        return check_condition_fn

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass