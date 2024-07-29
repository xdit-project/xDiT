from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn

from xfuser.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
)
from xfuser.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.models import xFuserModelBaseWrapper

logger = init_logger(__name__)


class xFuserTransformerBaseWrapper(xFuserModelBaseWrapper, metaclass=ABCMeta):
    # transformer: original transformer model (for example Transformer2DModel)
    def __init__(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ):
        transformer = self._convert_transformer_for_parallel(
            transformer,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
        )
        super().__init__(module=transformer)

    def _convert_transformer_for_parallel(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ) -> nn.Module:
        if get_pipeline_parallel_world_size() == 1 \
            and get_sequence_parallel_world_size() == 1:
            return transformer
        else:
            transformer = self._split_transformer_blocks(transformer)
            return self._wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=submodule_classes_to_wrap,
                submodule_name_to_wrap=submodule_name_to_wrap,
                submodule_addition_args=submodule_addition_args,
            )

    def _split_transformer_blocks(
        self,
        transformer: nn.Module,
    ):
        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                f"'{transformer.__class__.__name__}' object has no attribute "
                f"'transformer_blocks'. To use pipeline parallelism with"
                f"object {transformer.__class__.__name__}, please implement "
                f"custom _split_transformer_blocks method in derived class"
            )

        # transformer layer split
        attn_layer_num_for_pp = get_runtime_state().parallel_config.pp_config.attn_layer_num_for_pp
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        if attn_layer_num_for_pp is not None:
            assert sum(attn_layer_num_for_pp) == len(transformer.transformer_blocks), (
                "Sum of attn_layer_num_for_pp should be equal to the "
                "number of transformer blocks"
            )
            if pp_rank == 0:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    : attn_layer_num_for_pp[0]
                ]
            else:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    sum(attn_layer_num_for_pp[: pp_rank - 1]) : sum(
                        attn_layer_num_for_pp[:pp_rank]
                    )
                ]
        else:
            num_blocks_per_stage = (
                len(transformer.transformer_blocks) + pp_world_size - 1
            ) // pp_world_size
            start_idx = pp_rank * num_blocks_per_stage
            end_idx = min(
                (pp_rank + 1) * num_blocks_per_stage,
                len(transformer.transformer_blocks),
            )
            transformer.transformer_blocks = transformer.transformer_blocks[
                start_idx:end_idx
            ]
        # position embedding
        if pp_rank != 0:
            transformer.pos_embed = None
        return transformer

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _get_patch_height_width(self) -> Tuple[int, int]:
        patch_size = get_runtime_state().backbone_patch_size
        vae_scale_factor = get_runtime_state().vae_scale_factor
        width = get_runtime_state().input_config.width // patch_size // vae_scale_factor
        
        if get_runtime_state().patch_mode:
            height = (
                get_runtime_state().pp_patches_height[get_runtime_state().pipeline_patch_idx]
                // patch_size
            )
        else:
            height = sum(get_runtime_state().pp_patches_height) // patch_size 
        return height, width