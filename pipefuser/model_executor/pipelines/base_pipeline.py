from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.distributed as dist

from diffusers import DiffusionPipeline
from pipefuser.config.config import (
    EngineConfig,
    InputConfig,
    ParallelConfig,
    RuntimeConfig,
)
from pipefuser.distributed.parallel_state import get_classifier_free_guidance_world_size
from pipefuser.distributed.runtime_state import get_runtime_state, initialize_runtime_state, runtime_state_is_initialized
from pipefuser.logger import init_logger
from pipefuser.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_pipeline_parallel_world_size,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
)
from pipefuser.model_executor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.model_executor.schedulers import *
from pipefuser.model_executor.models.transformers import *


logger = init_logger(__name__)


class PipeFuserPipelineBaseWrapper(PipeFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
    ):
        self.module: DiffusionPipeline
        self._init_runtime_state(pipeline=pipeline, engine_config=engine_config)

        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(transformer)
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)

        super().__init__(module=pipeline)

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

    def _init_runtime_state(self, pipeline: DiffusionPipeline, engine_config: EngineConfig):
        initialize_runtime_state(pipeline=pipeline, engine_config=engine_config)

    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
        logger.info("Transformer backbone found, paralleling transformer...")
        wrapper = PipeFuserTransformerWrappersRegister.get_wrapper(transformer)
        transformer = wrapper(transformer=transformer)
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
    ):
        logger.info("UNet Backbone found")
        raise NotImplementedError("UNet parallelisation is not supported yet")

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
    ):
        logger.info("Scheduler found, paralleling scheduler...")
        wrapper = PipeFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(scheduler=scheduler)
        return scheduler

    def _init_sync_pipeline(self, latents: torch.Tensor):
        get_runtime_state().set_patched_mode(patch_mode=False)

        latents_list = [
            latents[:, :, start_idx: end_idx,:]
            for start_idx, end_idx in get_runtime_state().pp_patches_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        return latents


    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
    ):
        get_runtime_state().set_patched_mode(patch_mode=True)

        if get_pipeline_parallel_rank() == 0:
            # get latents computed in warmup stage
            # ignore latents after the last timestep
            #! if no warmup stage, use the input latents
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(latents.split(get_runtime_state().pp_patches_height, dim=2))
        elif get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
            patch_latents = list(latents.split(get_runtime_state().pp_patches_height, dim=2))
        else:
            patch_latents = [None for _ in range(get_runtime_state().num_pipeline_patch)]

        recv_timesteps = (
            num_timesteps - 1 if get_pipeline_parallel_rank() == 0 else num_timesteps
        )
        for _ in range(recv_timesteps):
            for patch_idx in range(get_runtime_state().num_pipeline_patch):
                get_pp_group().add_pipeline_recv_task(patch_idx)

        return patch_latents

    def _scheduler_step(
        self,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        t: Union[float, torch.Tensor],
        extra_step_kwargs: Dict,
    ):
        # compute previous image: x_t -> x_t-1
        return self.scheduler.step(
            noise_pred,
            t,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

    @staticmethod
    def enable_data_parallel(func):
        @wraps(func)
        def data_parallel_fn(self, *args, **kwargs):
            prompt = kwargs.get("prompt", None)
            negative_prompt = kwargs.get("negative_prompt", "")
            # dp_degree <= batch_size
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            if batch_size > 1:
                dp_degree = get_runtime_state().parallel_config.dp_degree
                dp_group_rank = get_world_group().rank // get_data_parallel_world_size()
                dp_group_batch_size = (batch_size + dp_degree - 1) // dp_degree
                start_batch_idx = dp_group_rank * dp_group_batch_size
                end_batch_idx = min((dp_group_rank + 1) * dp_group_batch_size, batch_size)
                prompt = prompt[start_batch_idx:end_batch_idx]
                if isinstance(negative_prompt, List):
                    negative_prompt = negative_prompt[start_batch_idx:end_batch_idx]
                kwargs["prompt"] = prompt
                kwargs["negative_prompt"] = negative_prompt
            return func(self, *args, **kwargs)
        return data_parallel_fn

    @staticmethod
    def check_to_use_naive_forward(func):
        @wraps(func)
        def check_naive_forward_fn(self, *args, **kwargs):
            if (
                get_pipeline_parallel_world_size() == 1
                and get_classifier_free_guidance_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            ):
                return self.module(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return check_naive_forward_fn

    def forward(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
