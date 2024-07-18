from abc import ABCMeta, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist

from diffusers import DiffusionPipeline
from pipefuser.refactor.config.config import (
    EngineConfig,
    InputConfig,
    ParallelConfig,
    RuntimeConfig,
)
from pipefuser.logger import init_logger
from pipefuser.refactor.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    set_random_seed,
    get_pp_group,
)
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.refactor.schedulers import *
from pipefuser.refactor.models.transformers import *


logger = init_logger(__name__)


class PipeFuserPipelineBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        self.module: DiffusionPipeline

        self._check_model_and_parallel_config(pipeline, parallel_config)
        self._check_distributed_env(parallel_config, runtime_config)
        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(
                transformer,
                parallel_config=parallel_config,
                runtime_config=runtime_config,
            )
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(
                scheduler,
                parallel_config=parallel_config,
                runtime_config=runtime_config,
            )

        super().__init__(
            module=pipeline,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )

    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "set_input_config"
        ):
            self.module.transformer.set_input_config(input_config)
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "set_input_config"
        ):
            self.module.unet.set_input_config(input_config)
        if hasattr(self.module, "vae") and hasattr(self.module.vae, "set_input_config"):
            self.module.vae.set_input_config(input_config)
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "set_input_config"
        ):
            self.module.scheduler.set_input_config(input_config)

    def set_num_pipeline_patch_and_patches_height(
        self, num_pipeline_patch: int, patches_height: List[int]
    ):
        self.num_pipeline_patch = num_pipeline_patch
        self.patches_height = patches_height
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.transformer.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height
            )
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.unet.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height
            )
        if hasattr(self.module, "vae") and hasattr(
            self.module.vae, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.vae.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height
            )
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.scheduler.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height
            )

    def set_patched_mode(self, patched: bool):
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "set_patched_mode"
        ):
            self.module.transformer.set_patched_mode(patched)
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "set_patched_mode"
        ):
            self.module.unet.set_patched_mode(patched)
        if hasattr(self.module, "vae") and hasattr(self.module.vae, "set_patched_mode"):
            self.module.vae.set_patched_mode(patched)
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "set_patched_mode"
        ):
            self.module.scheduler.set_patched_mode(patched)

    def reset_patch_idx(self):
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "reset_patch_idx"
        ):
            self.module.transformer.reset_patch_idx()
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "reset_patch_idx"
        ):
            self.module.unet.reset_patch_idx()
        if hasattr(self.module, "vae") and hasattr(self.module.vae, "reset_patch_idx"):
            self.module.vae.reset_patch_idx()
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "reset_patch_idx"
        ):
            self.module.scheduler.reset_patch_idx()

    def reset_activation_cache(self):
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "reset_activation_cache"
        ):
            self.module.transformer.reset_activation_cache()
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "reset_activation_cache"
        ):
            self.module.unet.reset_activation_cache()
        if hasattr(self.module, "vae") and hasattr(
            self.module.vae, "reset_activation_cache"
        ):
            self.module.vae.reset_activation_cache()
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "reset_activation_cache"
        ):
            self.module.scheduler.reset_activation_cache()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def _check_distributed_env(
        self,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        set_random_seed(runtime_config.seed)
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not dist.is_initialized():
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
            # get_pp_group().set_hyper_parameters(
            #     dtype=runtime_config.dtype,
            #     num_pipefusion_patches=parallel_config.pp_config.num_pipeline_patch,
            # )

    def _check_model_and_parallel_config(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
    ):
        if hasattr(pipeline, "transformer"):
            num_heads = pipeline.transformer.config.num_attention_heads
            ulysses_degree = parallel_config.sp_config.ulysses_degree
            if num_heads % ulysses_degree != 0 or num_heads < ulysses_degree:
                raise RuntimeError(
                    f"transformer backbone has {num_heads} heads, which is not "
                    f"divisible by or smaller than ulysses_degree "
                    f"{ulysses_degree}."
                )

    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info("Transformer backbone found, paralleling transformer...")
        wrapper = PipeFuserTransformerWrappersRegister.get_wrapper(transformer)
        transformer = wrapper(
            transformer=transformer,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info("UNet Backbone found")
        raise NotImplementedError("UNet parallelisation is not supported yet")

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info("Scheduler found, paralleling scheduler...")
        wrapper = PipeFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(
            scheduler=scheduler,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        return scheduler

    def _check_input_change_and_adjust(
        self,
        batch_size: int,
        height: int,
        width: int,
    ):
        if (height is not None or width is not None) and (
            height != self.input_config.height or width != self.input_config.width
        ):
            self.input_config.height = height or self.input_config.height
            self.input_config.width = width or self.input_config.width
            self.set_input_config(self.input_config)

            # TODO add shape & num_patch parameters
            get_pp_group().reset_buffer()
            self.reset_activation_cache()

        if hasattr(self.module, "transformer"):
            latents_height = height // self.module.vae_scale_factor
            latents_width = width // self.module.vae_scale_factor
            patch_size = self.module.transformer.config.patch_size
            pipeline_patches_height = (
                latents_height + self.num_pipeline_patch - 1
            ) // self.num_pipeline_patch
            pipeline_patches_height = (
                (pipeline_patches_height + patch_size - 1) // patch_size
            ) * patch_size
            pipeline_patches_num = (
                latents_height + pipeline_patches_height - 1
            ) // pipeline_patches_height
            pipeline_patches_height_list = [
                pipeline_patches_height for _ in range(pipeline_patches_num - 1)
            ]
            pipeline_patches_height_list.append(
                latents_height - pipeline_patches_height * (pipeline_patches_num - 1)
            )
            if pipeline_patches_num != self.num_pipeline_patch:
                logger.warning(
                    f"Pipeline patches num changed from "
                    f"{self.num_pipeline_patch} to {pipeline_patches_num} due "
                    f"to input size and model feature"
                )
            if (
                pipeline_patches_num != self.num_pipeline_patch
                or pipeline_patches_height_list != self.patches_height
            ):
                # sublayers activation cache reset
                self.set_num_pipeline_patch_and_patches_height(
                    pipeline_patches_num, pipeline_patches_height_list
                )
                if get_pipeline_parallel_rank() != 0:
                    batch_size = batch_size * (2 // self.parallel_config.cfg_degree)
                    hidden_dim = self.module.transformer.inner_dim
                    num_pipline_patches_tokens = [
                        (latents_width // patch_size) * (patch_height // patch_size)
                        for patch_height in pipeline_patches_height_list
                    ]
                    patches_shape = [
                        [batch_size, tokens, hidden_dim]
                        for tokens in num_pipline_patches_tokens
                    ]
                    feature_map_shape = [
                        batch_size,
                        (latents_width // patch_size) * (latents_height // patch_size),
                        hidden_dim,
                    ]
                else:
                    latents_channels = self.module.transformer.config.in_channels
                    patches_shape = [
                        [batch_size, latents_channels, patch_height, latents_width]
                        for patch_height in pipeline_patches_height_list
                    ]
                    feature_map_shape = [
                        batch_size,
                        latents_channels,
                        latents_height,
                        latents_width,
                    ]

                # reset pipeline communicator buffer
                get_pp_group().set_recv_buffer(
                    num_pipefusion_patches=pipeline_patches_num,
                    patches_shape_list=patches_shape,
                    feature_map_shape=feature_map_shape,
                    dtype=self.runtime_config.dtype,
                )

    def forward(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
