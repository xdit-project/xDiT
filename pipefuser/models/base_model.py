from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Type, Union
from functools import wraps

import torch.nn as nn
from pipefuser.config.config import InputConfig, ParallelConfig, RuntimeConfig
from pipefuser.base_wrapper import PipeFuserBaseWrapper
from pipefuser.distributed import get_world_group
from pipefuser.layers import *
from pipefuser.logger import init_logger

logger = init_logger(__name__)


# class PipeFuserModelBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):
class PipeFuserModelBaseWrapper(nn.Module, PipeFuserBaseWrapper, metaclass=ABCMeta):
    wrapped_layers: List[PipeFuserLayerBaseWrapper]

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
            runtime_config=runtime_config,
        )
        self.patched_mode = False

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

    def set_patched_mode(self, patched: bool):
        self.patched_mode = patched
        for layer in self.wrapped_layers:
            layer.set_patched_mode(patched)

    def set_num_pipeline_patch_and_patches_height(
        self, 
        num_pipeline_patch: int, 
        patches_height: List[List[int]], 
        patches_start_idx: List[List[int]],
        pp_patches_height: List[int],
        pp_patches_start_idx_local: List[int],
        pp_patches_start_end_idx: List[List[int]],
        pp_patches_token_start_end_idx: List[List[int]],
    ):
        self.num_pipeline_patch = num_pipeline_patch
        self.patches_height = patches_height
        self.patches_start_idx = patches_start_idx
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx = pp_patches_start_end_idx
        self.pp_patches_token_start_end_idx = pp_patches_token_start_end_idx
        for layer in self.wrapped_layers:
            layer.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch=num_pipeline_patch, 
                patches_height=patches_height, 
                patches_start_idx=patches_start_idx,
                pp_patches_height=pp_patches_height,
                pp_patches_start_idx_local=pp_patches_start_idx_local,
                pp_patches_start_end_idx=pp_patches_start_end_idx,
                pp_patches_token_start_end_idx=pp_patches_token_start_end_idx
            )

    def reset_patch_idx(self):
        self.current_patch_idx = 0
        for layer in self.wrapped_layers:
            layer.reset_patch_idx()

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
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
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
            if isinstance(module, PipeFuserLayerBaseWrapper):
                continue

            for subname, submodule in module.named_children():
                need_wrap = subname in submodule_name_to_wrap
                for class_to_wrap in submodule_classes_to_wrap:
                    if isinstance(submodule, class_to_wrap):
                        need_wrap = True
                        break

                if need_wrap:
                    wrapper = PipeFuserLayerWrappersRegister.get_wrapper(submodule)
                    additional_args = submodule_addition_args.get(subname, {})
                    logger.info(
                        f"[RANK {get_world_group().rank}] "
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
                                parallel_config,
                                runtime_config,
                                **additional_args,
                            ),
                        )
                    else:
                        setattr(
                            module,
                            subname,
                            wrapper(
                                submodule,
                                parallel_config,
                                runtime_config,
                            ),
                        )
                    # if isinstance(getattr(module, subname), PipeFuserPatchEmbedWrapper):
                    wrapped_layers.append(getattr(module, subname))
        self.wrapped_layers = wrapped_layers
        if wrap_self_module:
            self.module = model
        else:
            return model

    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config
        for submodule in self.module.modules():
            if hasattr(submodule, "set_input_config"):
                submodule.set_input_config(input_config)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
