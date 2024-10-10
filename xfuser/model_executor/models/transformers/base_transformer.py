from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn

from xfuser.core.distributed import (
    get_pipeline_parallel_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.models import xFuserModelBaseWrapper

logger = init_logger(__name__)


class StageInfo:
    def __init__(self):
        self.after_flags: Dict[str, bool] = {}


class xFuserTransformerBaseWrapper(xFuserModelBaseWrapper, metaclass=ABCMeta):
    # transformer: original transformer model (for example Transformer2DModel)
    def __init__(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
        transformer_blocks_name: List[str] = ["transformer_blocks"],
    ):
        self.stage_info = None
        transformer = self._convert_transformer_for_parallel(
            transformer,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
            transformer_blocks_name=transformer_blocks_name,
        )
        super().__init__(module=transformer)

    def _convert_transformer_for_parallel(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
        transformer_blocks_name: List[str] = [],
    ) -> nn.Module:
        if (
            get_pipeline_parallel_world_size() == 1
            and get_sequence_parallel_world_size() == 1
            and get_tensor_model_parallel_world_size() == 1
        ):
            return transformer
        else:
            transformer = self._split_transformer_blocks(
                transformer, transformer_blocks_name
            )
            transformer = self._wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=submodule_classes_to_wrap,
                submodule_name_to_wrap=submodule_name_to_wrap,
                submodule_addition_args=submodule_addition_args,
            )
            self._register_cache()
            return transformer

    def _split_transformer_blocks(
        self,
        transformer: nn.Module,
        blocks_name: List[str] = [],
    ):
        for block_name in blocks_name:
            if not hasattr(transformer, block_name):
                raise AttributeError(
                    f"'{transformer.__class__.__name__}' object has no attribute "
                    f"'{block_name}'."
                )

        # transformer layer split
        attn_layer_num_for_pp = (
            get_runtime_state().parallel_config.pp_config.attn_layer_num_for_pp
        )
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        blocks_list = {
            block_name: getattr(transformer, block_name) for block_name in blocks_name
        }
        num_blocks_list = [len(blocks) for blocks in blocks_list.values()]
        self.blocks_idx = {
            name: [sum(num_blocks_list[:i]), sum(num_blocks_list[: i + 1])]
            for i, name in enumerate(blocks_name)
        }
        if attn_layer_num_for_pp is not None:
            assert sum(attn_layer_num_for_pp) == sum(num_blocks_list), (
                "Sum of attn_layer_num_for_pp should be equal to the "
                "number of all the transformer blocks"
            )
            stage_block_start_idx = sum(attn_layer_num_for_pp[:pp_rank])
            stage_block_end_idx = sum(attn_layer_num_for_pp[: pp_rank + 1])

        else:
            num_blocks_per_stage = (
                sum(num_blocks_list) + pp_world_size - 1
            ) // pp_world_size
            stage_block_start_idx = pp_rank * num_blocks_per_stage
            stage_block_end_idx = min(
                (pp_rank + 1) * num_blocks_per_stage,
                sum(num_blocks_list),
            )

        self.stage_info = StageInfo()
        for name, [blocks_start, blocks_end] in zip(
            self.blocks_idx.keys(), self.blocks_idx.values()
        ):
            if (
                blocks_end <= stage_block_start_idx
                or stage_block_end_idx <= blocks_start
            ):
                setattr(transformer, name, nn.ModuleList([]))
                self.stage_info.after_flags[name] = False
            elif stage_block_start_idx <= blocks_start:
                if blocks_end <= stage_block_end_idx:
                    self.stage_info.after_flags[name] = True
                else:
                    setattr(
                        transformer,
                        name,
                        blocks_list[name][: -(blocks_end - stage_block_end_idx)],
                    )
                    self.stage_info.after_flags[name] = False
            elif blocks_start < stage_block_start_idx:
                if blocks_end <= stage_block_end_idx:
                    setattr(
                        transformer,
                        name,
                        blocks_list[name][stage_block_start_idx - blocks_start :],
                    )
                    self.stage_info.after_flags[name] = True
                else:  # blocks_end > stage_layer_end_idx
                    setattr(
                        transformer,
                        name,
                        blocks_list[name][
                            stage_block_start_idx
                            - blocks_start : stage_block_end_idx
                            - blocks_end
                        ],
                    )
                    self.stage_info.after_flags[name] = False

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
                get_runtime_state().pp_patches_height[
                    get_runtime_state().pipeline_patch_idx
                ]
                // patch_size
            )
        else:
            height = sum(get_runtime_state().pp_patches_height) // patch_size
        return height, width
