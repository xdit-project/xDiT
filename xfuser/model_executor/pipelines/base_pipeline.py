from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers import DiffusionPipeline
from xfuser.config.config import (
    EngineConfig,
    InputConfig,
)
from xfuser.logger import init_logger
from xfuser.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
    get_cfg_group,
    get_sp_group,
    get_runtime_state, 
    initialize_runtime_state
)
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.model_executor.schedulers import *
from xfuser.model_executor.models.transformers import *


logger = init_logger(__name__)


class xFuserPipelineBaseWrapper(xFuserBaseWrapper, metaclass=ABCMeta):

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

    def prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        prompt = (
            [""] * input_config.batch_size
            if input_config.batch_size > 1
            else ""
        )
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            use_resolution_binning=input_config.use_resolution_binning,
            num_inference_steps=steps,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    def _init_runtime_state(self, pipeline: DiffusionPipeline, engine_config: EngineConfig):
        initialize_runtime_state(pipeline=pipeline, engine_config=engine_config)

    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
        if (
            get_pipeline_parallel_world_size() == 1
            and get_sequence_parallel_world_size() == 1
            and get_classifier_free_guidance_world_size() == 1
        ):
            logger.info(
                "Transformer backbone found, but model parallelism is not enabled, "
                "use naive model"
            )
        else:
            logger.info("Transformer backbone found, paralleling transformer...")
            wrapper = xFuserTransformerWrappersRegister.get_wrapper(transformer)
            transformer = wrapper(transformer)
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
        wrapper = xFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(scheduler)
        return scheduler

    @abstractmethod
    def __call__(self):
        pass

    def _set_extra_comm_tensor_for_pipeline(
        self,
        extra_tensors_shape_dict: List[Tuple[str, List[int], int]] = []
    ):
        if (
            get_runtime_state().pipeline_comm_extra_tensors_info == \
                extra_tensors_shape_dict
        ):
            return
        for name, shape, cnt in extra_tensors_shape_dict:
            get_pp_group().set_extra_tensors_recv_buffer(name, shape, cnt)
        get_runtime_state().pipeline_comm_extra_tensors_info = extra_tensors_shape_dict

        

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
    
    def _process_cfg_split_batch(
        self,
        concat_group_0_negative: torch.Tensor,
        concat_group_0: torch.Tensor,
        concat_group_1_negative: torch.Tensor,
        concat_group_1: torch.Tensor,
    ):
        if get_classifier_free_guidance_world_size() == 1:
            concat_group_0 = torch.cat(
                [concat_group_0_negative, concat_group_0], dim=0
            )
            concat_group_1 = torch.cat(
                [concat_group_1_negative, concat_group_1], dim=0
            )
        elif get_classifier_free_guidance_rank() == 0:
            concat_group_0 = concat_group_0_negative
            concat_group_1 = concat_group_1_negative
        elif get_classifier_free_guidance_rank() == 1:
            concat_group_0 = concat_group_0
            concat_group_1 = concat_group_1
        else:
            raise ValueError("Invalid classifier free guidance rank")
        return concat_group_0, concat_group_1

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

    # synchronized compute the whole feature map in each pp stage
    def _sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        guidance_scale: float,
        timesteps: List[int],
        num_warmup_steps: int,
        extra_step_kwargs: List,
        added_cond_kwargs: Dict,
        progress_bar,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        sync_only: bool = False,
    ):
        latents = self._init_sync_pipeline(latents)
        for i, t in enumerate(timesteps):
            if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
                last_timestep_latents = latents

            # when there is only one pp stage, no need to recv
            if get_pipeline_parallel_world_size() == 1:
                pass
            # all ranks should recv the latent from the previous rank except
            #   the first rank in the first pipeline forward which should use
            #   the input latent
            elif get_pipeline_parallel_rank() == 0 and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()

            latents = self._backbone_forward(
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                t=t,
                guidance_scale=guidance_scale,
            )

            if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
                latents = self._scheduler_step(
                    latents, last_timestep_latents, t, extra_step_kwargs
                )
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if (
                sync_only
                and get_pipeline_parallel_rank()
                == get_pipeline_parallel_world_size() - 1
                and i == len(timesteps) - 1
            ):
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)

        if (sync_only and 
            get_sequence_parallel_world_size() > 1 and
            get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1
        ):
            sp_degree = get_sequence_parallel_world_size()
            sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
            latents_list = []
            for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        :, 
                        get_runtime_state().pp_patches_start_idx_local[pp_patch_idx]:
                        get_runtime_state().pp_patches_start_idx_local[pp_patch_idx+1],
                        :
                    ]
                    for sp_patch_idx in range(sp_degree)
                ]
            latents = torch.cat(latents_list, dim=-2)

        return latents

    # * implement of pipefusion
    def _async_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        guidance_scale: float,
        timesteps: List[int],
        num_warmup_steps: int,
        extra_step_kwargs: List,
        added_cond_kwargs: Dict,
        progress_bar,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        patch_latents = self._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps,
        )
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if (get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1)
            else None
        )

        first_async_recv = True
        for i, t in enumerate(timesteps):
            for patch_idx in range(num_pipeline_patch):
                if (
                    get_pipeline_parallel_rank()
                    == get_pipeline_parallel_world_size() - 1
                ):
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if get_pipeline_parallel_rank() == 0 and i == 0:
                    pass
                else:
                    if first_async_recv:
                        get_pp_group().recv_next()
                        first_async_recv = False
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                        idx=patch_idx
                    )
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    else:
                        get_pp_group().recv_next()
                patch_latents[patch_idx] = self._backbone_forward(
                    latents=patch_latents[patch_idx],
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    t=t,
                    guidance_scale=guidance_scale,
                )
                if (
                    get_pipeline_parallel_rank()
                    == get_pipeline_parallel_world_size() - 1
                ):
                    patch_latents[patch_idx] = self._scheduler_step(
                        patch_latents[patch_idx],
                        last_patch_latents[patch_idx],
                        t,
                        extra_step_kwargs,
                    )
                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(patch_latents[patch_idx])
                else:
                    get_pp_group().pipeline_isend(patch_latents[patch_idx])
                
                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                assert callback is None, "callback not supported in async " "pipeline"
                if (
                    callback is not None
                    and i + num_pipeline_warmup_steps % callback_steps == 0
                ):
                    step_idx = (i + num_pipeline_warmup_steps) // getattr(
                        self.scheduler, "order", 1
                    )
                    callback(step_idx, t, patch_latents[patch_idx])

        latents = None
        if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
            latents = torch.cat(patch_latents, dim=2)
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ..., 
                            get_runtime_state().pp_patches_start_idx_local[pp_patch_idx]:
                            get_runtime_state().pp_patches_start_idx_local[pp_patch_idx+1],
                            :
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
        return latents

    def _backbone_forward(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        added_cond_kwargs: Dict,
        t: Union[float, torch.Tensor],
        guidance_scale: float,
    ):
        if get_pipeline_parallel_rank() == 0:
            latents = torch.cat(
                [latents] * (2 // get_classifier_free_guidance_world_size())
            )
            latents = self.scheduler.scale_model_input(latents, t)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latents.device.type == "mps"

            if isinstance(current_timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            current_timestep = torch.tensor(
                [current_timestep], dtype=dtype, device=latents.device
            )
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(latents.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep.expand(latents.shape[0])
        noise_pred = self.transformer(
            latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=current_timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # classifier free guidance
        if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
            if get_classifier_free_guidance_world_size() == 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            elif get_classifier_free_guidance_world_size() == 2:
                noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                    noise_pred, separate_tensors=True
                )
            latents = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if (
                self.transformer.config.out_channels // 2
                == self.transformer.config.in_channels
            ):
                latents = latents.chunk(2, dim=1)[0]
        else:
            latents = noise_pred

        return latents