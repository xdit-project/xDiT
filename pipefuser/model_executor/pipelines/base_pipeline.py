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
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.transformer.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height, patches_start_idx, pp_patches_height, pp_patches_start_idx_local, pp_patches_start_end_idx, pp_patches_token_start_end_idx
            )
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.unet.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height, patches_start_idx, pp_patches_height, pp_patches_start_idx_local, pp_patches_start_end_idx, pp_patches_token_start_end_idx
            )
        if hasattr(self.module, "vae") and hasattr(
            self.module.vae, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.vae.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height, patches_start_idx, pp_patches_height, pp_patches_start_idx_local, pp_patches_start_end_idx, pp_patches_token_start_end_idx
            )
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "set_num_pipeline_patch_and_patches_height"
        ):
            self.module.scheduler.set_num_pipeline_patch_and_patches_height(
                num_pipeline_patch, patches_height, patches_start_idx, pp_patches_height, pp_patches_start_idx_local, pp_patches_start_end_idx, pp_patches_token_start_end_idx
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
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not dist.is_initialized():
                init_distributed_environment(random_seed=runtime_config.seed)
            initialize_model_parallel(
                data_parallel_degree=parallel_config.dp_degree,
                classifier_free_guidance_degree=parallel_config.cfg_degree,
                sequence_parallel_degree=parallel_config.sp_degree,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_degree=parallel_config.tp_degree,
                pipeline_parallel_degree=parallel_config.pp_degree,
            )

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
        orig_height: Optional[int] = None,
        orig_width: Optional[int] = None,
    ):
        if (height is not None or width is not None) and (
            height != self.input_config.height or width != self.input_config.width
        ) or (orig_height is not None or orig_width is not None) and (
            orig_height != self.input_config.orig_height or orig_width != self.input_config.orig_width
        ):

            self.input_config.height = height or self.input_config.height
            self.input_config.width = width or self.input_config.width
            self.input_config.orig_height = orig_height or self.input_config.orig_height
            self.input_config.orig_width = orig_width or self.input_config.orig_width
            self.set_input_config(self.input_config)

            # TODO add shape & num_patch parameters
            # sublayers activation cache reset
            self.reset_activation_cache()

        if hasattr(self.module, "transformer"):
            num_sp_patches = get_sequence_parallel_world_size()
            sp_patch_idx = get_sequence_parallel_rank()
            # Pipeline patches
            latents_height = height // self.module.vae_scale_factor
            latents_width = width // self.module.vae_scale_factor
            if latents_height % num_sp_patches != 0:
                raise ValueError("The height of the input is not divisible by the number of sequence parallel devices")
            patch_size = self.module.transformer.config.patch_size
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
            pipeline_patches_height_list = [
                pipeline_patches_height for _ in range(num_pipeline_patch - 1)
            ]
            last_pp_patch_height = latents_height - pipeline_patches_height * (num_pipeline_patch - 1)
            if last_pp_patch_height % (patch_size * num_sp_patches) != 0:
                raise ValueError(
                    f"The height of the last pipeline patch is {last_pp_patch_height}, "
                    f"which is not a multiple of (patch_size * num_sp_patches): "
                    f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                    f"or sp_degree argument so that the condition are met ")
            pipeline_patches_height_list.append(last_pp_patch_height)
            if num_pipeline_patch != self.num_pipeline_patch:
                logger.warning(
                    f"Pipeline patches num changed from "
                    f"{self.num_pipeline_patch} to {num_pipeline_patch} due "
                    f"to input size and parallelisation requirements"
                )
            
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


            if (
                num_pipeline_patch != self.num_pipeline_patch
                or pp_sp_patches_height != self.patches_height
                or pp_sp_patches_start_idx != self.patches_start_idx
            ):
                pp_patches_height = [
                    sp_patches_height[sp_patch_idx]
                    for sp_patches_height in pp_sp_patches_height
                ]
                pp_patches_start_idx_local = [0] + [
                    sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
                ]
                pp_patches_start_end_idx = [
                    sp_patches_start_idx[sp_patch_idx: sp_patch_idx + 2]
                    for sp_patches_start_idx in pp_sp_patches_start_idx
                ]
                pp_patches_token_start_end_idx = [
                    [
                        (latents_width // patch_size) * (start_idx // patch_size),
                        (latents_width // patch_size) * (end_idx // patch_size),
                    ]
                    for start_idx, end_idx in pp_patches_start_end_idx
                ]

                # set runtime patches metadata for parallel
                self.set_num_pipeline_patch_and_patches_height(
                    num_pipeline_patch=num_pipeline_patch, 
                    patches_height=pp_sp_patches_height, 
                    patches_start_idx=pp_sp_patches_start_idx, 
                    pp_patches_height=pp_patches_height,
                    pp_patches_start_idx_local=pp_patches_start_idx_local,
                    pp_patches_start_end_idx=pp_patches_start_end_idx,
                    pp_patches_token_start_end_idx=pp_patches_token_start_end_idx
                )

                # calc communicator buffer metadata
                if get_pipeline_parallel_rank() != 0:
                    batch_size = batch_size * (2 // self.parallel_config.cfg_degree)
                    hidden_dim = self.module.transformer.inner_dim
                    num_patches_tokens = [
                        (latents_width // patch_size) * (patch_height // patch_size)
                        for patch_height in pp_patches_height
                    ]
                    patches_shape = [
                        [batch_size, tokens, hidden_dim]
                        for tokens in num_patches_tokens 
                    ]
                    feature_map_shape = [
                        batch_size,
                        (latents_width // patch_size) * (sum(pp_patches_height) // patch_size),
                        hidden_dim,
                    ]
                #TODO: if use distributed scheduler alone sp devices, edit pp rank0 the logic
                else:
                    latents_channels = self.module.transformer.config.in_channels
                    patches_shape = [
                        [batch_size, latents_channels, sp_patches_height[sp_patch_idx], latents_width]
                        for sp_patches_height in pp_sp_patches_height
                    ]
                    feature_map_shape = [
                        batch_size,
                        latents_channels,
                        sum(pp_patches_height),
                        latents_width,
                    ]

                # reset pipeline communicator buffer
                get_pp_group().set_recv_buffer(
                    num_pipefusion_patches=num_pipeline_patch,
                    patches_shape_list=patches_shape,
                    feature_map_shape=feature_map_shape,
                    dtype=self.runtime_config.dtype,
                )

    def _init_sync_pipeline(self, latents: torch.Tensor):
        self.set_patched_mode(patched=False)
        self.reset_patch_idx()

        latents_list = [
            latents[:, :, start_idx: end_idx,:]
            for start_idx, end_idx in self.pp_patches_start_end_idx
        ]
        latents = torch.cat(latents_list, dim=-2)
        return latents


    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
    ):
        self.set_patched_mode(patched=True)
        self.reset_patch_idx()

        if get_pipeline_parallel_rank() == 0:
            # get latents computed in warmup stage
            # ignore latents after the last timestep
            #! if no warmup stage, use the input latents
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(latents.split(self.pp_patches_height, dim=2))
        elif get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
            patch_latents = list(latents.split(self.pp_patches_height, dim=2))
        else:
            patch_latents = [None for _ in range(self.num_pipeline_patch)]

        recv_timesteps = (
            num_timesteps - 1 if get_pipeline_parallel_rank() == 0 else num_timesteps
        )
        for _ in range(recv_timesteps):
            for patch_idx in range(self.num_pipeline_patch):
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
                dp_degree = self.parallel_config.dp_degree
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

    def forward(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
