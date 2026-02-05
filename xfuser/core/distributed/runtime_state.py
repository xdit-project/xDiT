from abc import ABCMeta
import random
from typing import List, Optional

import numpy as np
import torch
from torch.cuda import manual_seed as device_manual_seed
from torch.cuda import manual_seed_all as device_manual_seed_all
import diffusers
from diffusers import DiffusionPipeline
import torch.distributed

try:
    import torch_musa
    from torch_musa.core.random import manual_seed as device_manual_seed
    from torch_musa.core.random import manual_seed_all as device_manual_seed_all
except ModuleNotFoundError:
    pass

import xfuser.envs as envs
from xfuser.envs import PACKAGES_CHECKER
if envs._is_npu():
    from torch.npu import manual_seed as device_manual_seed
    from torch.npu import manual_seed_all as device_manual_seed_all

from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.config.config import (
    ParallelConfig,
    RuntimeConfig,
    InputConfig,
    EngineConfig,
)
from xfuser.logger import init_logger
from .parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from xfuser.config.args import xFuserArgs

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed(seed)
    device_manual_seed_all(seed)


class RuntimeState(metaclass=ABCMeta):
    attention_backend: AttentionBackendType = AttentionBackendType.SDPA_FLASH
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
        attention_backend = self._select_attention_backend(config)
        self.set_attention_backend(attention_backend)

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
                vae_parallel_size=parallel_config.vae_parallel_size,
            )

    def destroy_distributed_env(self):
        if model_parallel_is_initialized():
            destroy_model_parallel()
        destroy_distributed_environment()

    def set_attention_backend(self, attention_backend: str | AttentionBackendType):
        """
        Set the attention backend for the current environment.
        Given attention_backend can be either AttentionBackendType or a string with the name of the backend.
        """
        if isinstance(attention_backend, str):
            try:
                attention_backend = AttentionBackendType[attention_backend.upper()]
            except:
                pass

        if not isinstance(attention_backend, AttentionBackendType):
            raise ValueError(f"Value '{attention_backend}' is not a valid attention backend.")

        self._check_if_backend_compatible_with_current_configuration(attention_backend)
        self.attention_backend = attention_backend
        logger.warning("Using {} as attention backend.".format(self.attention_backend.name))
        if attention_backend in [AttentionBackendType.FLASH_3_FP8, AttentionBackendType.AITER_FP8]:
            logger.warning("FP8 attention backend is enabled. This may cause poor quality outputs, consider using hybrid attention if possible.")


    def _select_attention_backend(self, engine_config: Optional[EngineConfig] = None):
        """
        Select the best attention backend for the current environment.
        """
        if engine_config and engine_config.runtime_config.attention_backend:
            backend = AttentionBackendType[engine_config.runtime_config.attention_backend.upper()]

        elif envs._is_hip():
            if env_info["has_aiter"]:
                backend = AttentionBackendType.AITER
            elif env_info["has_flash_attn"]:
                backend = AttentionBackendType.FLASH
            else:
                backend = AttentionBackendType.SDPA

        elif env_info["has_flash_attn_4"]:
            backend = AttentionBackendType.FLASH_4
        elif env_info["has_flash_attn_3"]:
            backend = AttentionBackendType.FLASH_3
        elif torch.backends.cudnn.is_available():
            backend = AttentionBackendType.CUDNN
        elif env_info["has_flash_attn"]:
            backend = AttentionBackendType.FLASH
        elif env_info["has_npu_flash_attn"]:
            backend = AttentionBackendType.NPU
        else:
            backend = AttentionBackendType.SDPA

        return backend

    def _check_if_backend_compatible_with_current_configuration(self, attention_backend: AttentionBackendType):
        """
        Check if the selected attention backend is compatible with the current configuration.
        """
        if attention_backend in [AttentionBackendType.SDPA,
                                 AttentionBackendType.SDPA_MATH,
                                 AttentionBackendType.FLASH_4,
                                 AttentionBackendType.AITER_FP8,
                                 AttentionBackendType.AITER_SAGE]:
            if self.parallel_config.ring_degree > 1:
                raise RuntimeError("Selected attention backend does not support ring parallelism.")
        if attention_backend == AttentionBackendType.AITER_FP8:
            try:
                from aiter import flash_attn_fp8_pertensor_func
            except ImportError:
                raise RuntimeError("AITER fp8 flash attention is not available, please update AITER")
        elif attention_backend == AttentionBackendType.AITER_SAGE:
            try:
                import aiter.ops.triton.attention
                from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
            except ImportError:
                raise RuntimeError("AITER Sage attention is not available, please update AITER") from None



class UnetRuntimeState(RuntimeState):

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        super().__init__(config)
        self.sanity_check()

    def sanity_check(self):
        if self.parallel_config.world_size > 1:
            if not(self.parallel_config.cfg_degree == 2 and self.parallel_config.world_size == 2):
                raise RuntimeError("UnetRuntimeState only supports 2 GPUs with CFG Parallel")


class DiTRuntimeState(RuntimeState):
    patch_mode: bool
    pipeline_patch_idx: int
    vae_scale_factor: int
    vae_scale_factor_spatial: int
    vae_scale_factor_temporal: int
    backbone_patch_size: int
    pp_patches_height: Optional[List[int]]
    pp_patches_start_idx_local: Optional[List[int]]
    pp_patches_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_start_idx_local: Optional[List[int]]
    pp_patches_token_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_num: Optional[List[int]]
    max_condition_sequence_length: int
    split_text_embed_in_sp: bool

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        self.use_hybrid_fp8_attn = False
        super().__init__(config)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self._check_model_and_parallel_config(
            pipeline=pipeline, parallel_config=config.parallel_config
        )
        try:
            self._check_pipeline_class_name(pipeline, config)
        except Exception:
            # Keeps backward compatatability with existing pipeline classes.
            pass

    def _check_pipeline_class_name(self, pipeline: DiffusionPipeline, config: EngineConfig):
        self.cogvideox = False
        self.consisid = False
        self.hunyuan_video = False
        if pipeline.__class__.__name__.startswith(("CogVideoX", "ConsisID", "HunyuanVideo", "Wan")):
            if pipeline.__class__.__name__.startswith("CogVideoX"):
                self.cogvideox = True
            elif pipeline.__class__.__name__.startswith("ConsisID"):
                self.consisid = True
            else:
                self.hunyuan_video = True
            self._set_cogvideox_parameters(
                vae_scale_factor_spatial=pipeline.vae_scale_factor_spatial,
                vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
                backbone_patch_size=pipeline.transformer.config.patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.num_attention_heads
                * pipeline.transformer.config.attention_head_dim,
            )
        elif pipeline.__class__.__name__.startswith("ZImage"):
            self._set_model_parameters(
                vae_scale_factor=pipeline.vae_scale_factor,
                backbone_patch_size=pipeline.transformer.config.all_patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.n_heads
                * pipeline.transformer.config.axes_dims[-1]
            )
        else:
            vae_scale_factor = getattr(pipeline, "vae_scale_factor", 0)
            if pipeline.__class__.__name__.startswith("Flux") and diffusers.__version__ >= '0.32':
                vae_scale_factor *= 2
            self._set_model_parameters(
                vae_scale_factor=vae_scale_factor,
                backbone_patch_size=pipeline.transformer.config.patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.num_attention_heads
                * pipeline.transformer.config.attention_head_dim,
            )

    def increment_step_counter(self):
        """
        Keep track of the current denoising step, and set fp8 flag based on the current step.
        Used for hybrid fp8 attention, when toggling between bf16 and fp8 is needed.
        When the entire denoising process is over, the step counter is reset to 0.
        """
        if self.use_hybrid_fp8_attn:
            self.attention_backend = self.fp8_decision_vector[self.step_counter]
            self.step_counter = self.step_counter + 1
            if self.step_counter >= self.total_steps:
                self.step_counter = torch.tensor(0, dtype=torch.int)

    def set_hybrid_attn_parameters(self, fp8_decision_vector: torch.Tensor):
        """
        Set the parameters for hybrid fp8 attention.
        fp8_decision_vector: A boolean tensor of length equal to the total number of denoising steps.
        Each element indicates whether to use fp8 attention (True) or bf16 attention (False).
        """
        # Hybrid attention does not neccessarily set fp8 as the first attention backend.
        # Thus, we need to check fp8 availability in advance if hybrid attention is enabled.
        if envs.PACKAGES_CHECKER.packages_info["has_aiter"]:
            self._check_if_backend_compatible_with_current_configuration(AttentionBackendType.AITER_FP8)
            self.fp8_decision_vector = [AttentionBackendType.AITER_FP8 if use_fp8 else AttentionBackendType.AITER for use_fp8 in fp8_decision_vector]
        elif envs.PACKAGES_CHECKER.packages_info["has_flash_attn_3"]:
            self._check_if_backend_compatible_with_current_configuration(AttentionBackendType.FLASH_3_FP8)
            self.fp8_decision_vector = [AttentionBackendType.FLASH_3_FP8 if use_fp8 else AttentionBackendType.FLASH_3 for use_fp8 in fp8_decision_vector]
        else:
            raise RuntimeError("Hybrid fp8 attention is currently only supported for AITER and FlashAttention v3 backends.")

        # The below needs to be torch tensors to avoid recompilations with torch.compile.
        self.total_steps = torch.tensor(len(fp8_decision_vector), dtype=torch.int)
        self.step_counter = torch.tensor(0, dtype=torch.int)
        self.use_hybrid_fp8_attn = torch.tensor(True, dtype=torch.bool)

    def set_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        max_condition_sequence_length: Optional[int] = None,
        split_text_embed_in_sp: bool = True,
    ):
        self.input_config.num_inference_steps = (
            num_inference_steps or self.input_config.num_inference_steps
        )
        self.max_condition_sequence_length = max_condition_sequence_length
        self.split_text_embed_in_sp = split_text_embed_in_sp
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        if seed is not None and seed != self.input_config.seed:
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            not self.ready
            or (height and self.input_config.height != height)
            or (width and self.input_config.width != width)
            or (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._input_size_change(height, width, batch_size)

        self.ready = True

    def set_video_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        split_text_embed_in_sp: bool = True,
    ):
        self.input_config.num_inference_steps = (
            num_inference_steps or self.input_config.num_inference_steps
        )
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        self.split_text_embed_in_sp = split_text_embed_in_sp
        if seed is not None and seed != self.input_config.seed:
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            not self.ready
            or (height and self.input_config.height != height)
            or (width and self.input_config.width != width)
            or (num_frames and self.input_config.num_frames != num_frames)
            or (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._video_input_size_change(height, width, num_frames, batch_size)

        self.ready = True

    def _set_cogvideox_parameters(
        self,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int,
    ):
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

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
        num_heads = self._get_model_attention_heads(pipeline)
        ulysses_degree = parallel_config.sp_config.ulysses_degree
        if num_heads % ulysses_degree != 0 or num_heads < ulysses_degree:
            raise RuntimeError(
                f"transformer backbone has {num_heads} heads, which is not "
                f"divisible by or smaller than ulysses_degree "
                f"{ulysses_degree}."
            )

    def _get_model_attention_heads(self, pipeline: DiffusionPipeline) -> int:
        if "num_attention_heads" in pipeline.transformer.config:
            return pipeline.transformer.config.num_attention_heads
        elif "n_heads" in pipeline.transformer.config:
            return pipeline.transformer.config.n_heads
        else:
            raise RuntimeError(
                "Cannot find the number of attention heads in transformer config. Model is not supported."
            )

    def _set_model_parameters(
        self,
        vae_scale_factor: int,
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int,
    ):
        self.vae_scale_factor = vae_scale_factor
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

    def _input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
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
        num_frames: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.num_frames = num_frames or self.input_config.num_frames
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        if self.cogvideox:
            self._calc_cogvideox_patches_metadata()
        elif self.consisid:
            self._calc_consisid_patches_metadata()
        elif self.hunyuan_video:
            # TODO: implement the hunyuan video patches metadata
            pass
        else:
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
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
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
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
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
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _calc_cogvideox_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor_spatial = self.vae_scale_factor_spatial
        latents_height = self.input_config.height // vae_scale_factor_spatial
        latents_width = self.input_config.width // vae_scale_factor_spatial
        latents_frames = (
            self.input_config.num_frames - 1
        ) // self.vae_scale_factor_temporal + 1

        if latents_height % num_sp_patches != 0:
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
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
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
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
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _calc_consisid_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor_spatial = self.vae_scale_factor_spatial
        latents_height = self.input_config.height // vae_scale_factor_spatial
        latents_width = self.input_config.width // vae_scale_factor_spatial
        latents_frames = (
            self.input_config.num_frames - 1
        ) // self.vae_scale_factor_temporal + 1

        if latents_height % num_sp_patches != 0:
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
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
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
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
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _reset_recv_buffer(self):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype=self.runtime_config.dtype)

    def _reset_recv_skip_buffer(self, num_blocks_per_stage):
        batch_size = self.input_config.batch_size
        batch_size = batch_size * (2 // self.parallel_config.cfg_degree)
        hidden_dim = self.backbone_inner_dim
        num_patches_tokens = [
            end - start for start, end in self.pp_patches_token_start_end_idx_global
        ]
        patches_shape = [
            [num_blocks_per_stage, batch_size, tokens, hidden_dim]
            for tokens in num_patches_tokens
        ]
        feature_map_shape = [
            num_blocks_per_stage,
            batch_size,
            sum(num_patches_tokens),
            hidden_dim,
        ]
        # reset pipeline communicator buffer
        get_pp_group().set_skip_tensor_recv_buffer(
            patches_shape_list=patches_shape,
            feature_map_shape=feature_map_shape,
        )



class ExternalRuntimeState(RuntimeState):
    """
    Runtime state for running xDiT components outside xDiT.
    This can be used to test individual components in tests without
    having to setup a full distributed environment.
    """
    def __init__(self):
        # Creating config with default params
        config, _ = xFuserArgs().create_config()
        super().__init__(config)


    def _check_distributed_env(self, parallel_config):
        pass


# _RUNTIME: Optional[RuntimeState] = None
# TODO: change to RuntimeState after implementing the unet
_RUNTIME: Optional[DiTRuntimeState] = None


def runtime_state_is_initialized():
    return _RUNTIME is not None


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME


def initialize_runtime_state(pipeline: Optional[DiffusionPipeline] = None, engine_config: Optional[EngineConfig] = None):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning(
            "Runtime state is already initialized, reinitializing with pipeline..."
        )
    if hasattr(pipeline, "transformer"):
        _RUNTIME = DiTRuntimeState(pipeline=pipeline, config=engine_config)
    elif hasattr(pipeline, "unet"):
        _RUNTIME = UnetRuntimeState(pipeline=pipeline, config=engine_config)
    elif not pipeline:
        _RUNTIME = ExternalRuntimeState()

