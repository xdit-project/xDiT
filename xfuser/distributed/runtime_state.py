from abc import ABCMeta
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.distributed

from xfuser.config.config import ParallelConfig, RuntimeConfig, InputConfig, EngineConfig
from xfuser.logger import init_logger
from xfuser.distributed.parallel_state import (
    destroy_distributed_environment, 
    destroy_model_parallel, 
    get_pipeline_parallel_rank, 
    get_pp_group, 
    get_sequence_parallel_rank, 
    get_sequence_parallel_world_size, 
    init_distributed_environment, 
    initialize_model_parallel, 
    model_parallel_is_initialized,
)
    
logger = init_logger(__name__)

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RuntimeState(metaclass=ABCMeta):
    parallel_config: ParallelConfig
    runtime_config: RuntimeConfig
    input_config: InputConfig
    num_pipeline_patch: int
    ready: bool = False
    def __init__(self, config: EngineConfig):
        self.parallel_config = config.parallel_config
        self.runtime_config = config.runtime_config
        self.input_config = InputConfig()
        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        self.ready = False

        self._check_distributed_env(config.parallel_config)

    def is_ready(self):
        return self.ready

    def _check_distributed_env(
        self,
        parallel_config: ParallelConfig,
    ):
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not torch.distributed.is_initialized():
                init_distributed_environment()
            initialize_model_parallel(
                data_parallel_degree=parallel_config.dp_degree,
                classifier_free_guidance_degree=parallel_config.cfg_degree,
                sequence_parallel_degree=parallel_config.sp_degree,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_degree=parallel_config.tp_degree,
                pipeline_parallel_degree=parallel_config.pp_degree,
            )
    
    def destory_distributed_env(self):
        if model_parallel_is_initialized():
            destroy_model_parallel()
        destroy_distributed_environment()


class DiTRuntimeState(RuntimeState):
    patch_mode: bool
    pipeline_patch_idx: int
    vae_scale_factor: int
    backbone_patch_size: int
    pp_patches_height: Optional[List[int]]
    pp_patches_start_idx_local: Optional[List[int]]
    pp_patches_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_start_end_idx: Optional[List[List[int]]]
    pp_patches_token_num: Optional[List[int]]
    # Storing the shape of a tensor that is not latent but requires pp communication 
    #   torch.Size: size of tensor
    #   int: number of recv buffer it needs
    pipeline_comm_extra_tensors_info: List[Tuple[str, List[int], int]]

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        super().__init__(config)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self._check_model_and_parallel_config(
            pipeline=pipeline, 
            parallel_config=config.parallel_config
        )
        self._set_model_parameters(
            vae_scale_factor=pipeline.vae_scale_factor,
            backbone_patch_size=pipeline.transformer.config.patch_size,
            backbone_in_channel=pipeline.transformer.config.in_channels,
            backbone_inner_dim=pipeline.transformer.config.num_attention_heads * pipeline.transformer.config.attention_head_dim,
        )
        self.pipeline_comm_extra_tensors_info = []

    def set_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.input_config.num_inference_steps = num_inference_steps or self.input_config.num_inference_steps
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        if (seed is not None and seed != self.input_config.seed):
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            (height and self.input_config.height != height) or 
            (width and self.input_config.width != width) or 
            (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._input_size_change(height, width, batch_size)

        self.ready = True

    def set_video_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        video_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.input_config.num_inference_steps = num_inference_steps or self.input_config.num_inference_steps
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        if (seed is not None and seed != self.input_config.seed):
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            (height and self.input_config.height != height) or 
            (width and self.input_config.width != width) or 
            (video_length and self.input_config.video_length != video_length) or
            (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._video_input_size_change(height, width, video_length, batch_size)

        self.ready = True
    
    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0

    def _check_model_and_parallel_config(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
    ):
        num_heads = pipeline.transformer.config.num_attention_heads
        ulysses_degree = parallel_config.sp_config.ulysses_degree
        if num_heads % ulysses_degree != 0 or num_heads < ulysses_degree:
            raise RuntimeError(
                f"transformer backbone has {num_heads} heads, which is not "
                f"divisible by or smaller than ulysses_degree "
                f"{ulysses_degree}."
            )

    def _set_model_parameters(
        self, 
        vae_scale_factor: int, 
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int
    ):
        self.vae_scale_factor = vae_scale_factor
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel
    
    def _input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        self._calc_patches_metadata()
        self._reset_recv_buffer()
        
    def _video_input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        video_length: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.video_length = video_length or self.input_config.video_length
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        self._calc_patches_metadata()
        self._reset_recv_buffer()

    def _calc_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor = self.vae_scale_factor
        latents_height = self.input_config.height // vae_scale_factor
        latents_width = self.input_config.width // vae_scale_factor

        if latents_height % num_sp_patches != 0:
            raise ValueError("The height of the input is not divisible by the number of sequence parallel devices")

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1) // (patch_size * num_sp_patches)
        ) * (patch_size * num_sp_patches)
        # get the number of pipeline that matches patch height requirements
        num_pipeline_patch = (
            latents_height + pipeline_patches_height - 1
        ) // pipeline_patches_height
        if num_pipeline_patch != self.num_pipeline_patch:
            logger.warning(
                f"Pipeline patches num changed from "
                f"{self.num_pipeline_patch} to {num_pipeline_patch} due "
                f"to input size and parallelisation requirements"
            )
        pipeline_patches_height_list = [
            pipeline_patches_height for _ in range(num_pipeline_patch - 1)
        ]
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (num_pipeline_patch - 1)
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met ")
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i]) for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[pp_patch_idx * num_sp_patches: (pp_patch_idx + 1) * num_sp_patches]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[pp_patch_idx * num_sp_patches: (pp_patch_idx + 1) * num_sp_patches + 1]
            for pp_patch_idx in range(num_pipeline_patch)
        ]

        pp_patches_height = [
            sp_patches_height[sp_patch_idx]
            for sp_patches_height in pp_sp_patches_height
        ]
        pp_patches_start_idx_local = [0] + [
            sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
        ]
        pp_patches_start_end_idx_global = [
            sp_patches_start_idx[sp_patch_idx: sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_end_idx = pp_patches_token_start_end_idx
        self.pp_patches_token_num = pp_patches_token_num


    def _reset_recv_buffer(self):
        # calc communicator buffer metadata
        batch_size = self.input_config.batch_size
        if get_pipeline_parallel_rank() != 0:
            batch_size = batch_size * (2 // self.parallel_config.cfg_degree)
            hidden_dim = self.backbone_inner_dim
            num_patches_tokens = [
                end - start
                for start, end in self.pp_patches_token_start_end_idx
            ]
            patches_shape = [
                [batch_size, tokens, hidden_dim]
                for tokens in num_patches_tokens 
            ]
            feature_map_shape = [
                batch_size,
                sum(num_patches_tokens),
                hidden_dim,
            ]
        #TODO: if use distributed scheduler alone sp devices, edit pp rank0 the logic
        else:
            latents_channels = self.backbone_in_channel
            latents_width = self.input_config.width // self.vae_scale_factor
            patches_shape = [
                [batch_size, latents_channels, patch_height, latents_width]
                for patch_height in self.pp_patches_height
            ]
            feature_map_shape = [
                batch_size,
                latents_channels,
                self.pp_patches_start_idx_local[-1],
                latents_width,
            ]

        # reset pipeline communicator buffer
        get_pp_group().set_recv_buffer(
            num_pipefusion_patches=self.num_pipeline_patch,
            patches_shape_list=patches_shape,
            feature_map_shape=feature_map_shape,
            dtype=self.runtime_config.dtype,
        )


# _RUNTIME: Optional[RuntimeState] = None
#TODO: change to RuntimeState after implementing the unet
_RUNTIME: Optional[DiTRuntimeState] = None

def runtime_state_is_initialized():
    return _RUNTIME is not None

def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME

def initialize_runtime_state(pipeline: DiffusionPipeline, engine_config: EngineConfig):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning("Runtime state is already initialized, reinitializing with pipeline...")
    if hasattr(pipeline, "transformer"):
        _RUNTIME = DiTRuntimeState(pipeline=pipeline, config=engine_config)