import os
from typing import Callable, Dict, List, Tuple, Callable, Optional, Union

import torch
import torch.distributed
from diffusers import HunyuanDiTPipeline
from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import (
    SUPPORTED_SHAPE,
    map_to_standard_shapes,
    get_resize_crop_region_for_grid,
    rescale_noise_cfg,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from xfuser.config import EngineConfig
from xfuser.logger import init_logger
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_pipeline_parallel_rank,
    get_cfg_group,
    get_pp_group,
    get_sequence_parallel_world_size,
    get_sp_group,
    is_dp_last_group,
    is_pipeline_last_stage,
    is_pipeline_first_stage,
    get_world_group
)
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)


@xFuserPipelineWrapperRegister.register(HunyuanDiTPipeline)
class xFuserHunyuanDiTPipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = HunyuanDiTPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        use_resolution_binning: bool = True,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation with HunyuanDiT.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds_2` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            negative_prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds_2` is passed directly.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable[[int, int, Dict], None]`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A callback function or a list of callback functions to be called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                A list of tensor inputs that should be passed to the callback function. If not defined, all tensor
                inputs will be passed.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Rescale the noise_cfg according to `guidance_rescale`. Based on findings of [Common Diffusion Noise
                Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
            original_size (`Tuple[int, int]`, *optional*, defaults to `(1024, 1024)`):
                The original size of the image. Used to calculate the time ids.
            target_size (`Tuple[int, int]`, *optional*):
                The target size of the image. Used to calculate the time ids.
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to `(0, 0)`):
                The top left coordinates of the crop. Used to calculate the time ids.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                Whether to use resolution binning or not. If `True`, the input resolution will be mapped to the closest
                standard resolution. Supported resolutions are 1024x1024, 1280x1280, 1024x768, 1152x864, 1280x960,
                768x1024, 864x1152, 960x1280, 1280x768, and 768x1280. It is recommended to set this to `True`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        height = int((height // 16) * 16)
        width = int((width // 16) * 16)

        if use_resolution_binning and (height, width) not in SUPPORTED_SHAPE:
            width, height = map_to_standard_shapes(width, height)
            height = int(height)
            width = int(width)
            logger.warning(
                f"Reshaped to (height, width)=({height}, {width}), Supported shapes are {SUPPORTED_SHAPE}"
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        #! ---------------------------------------- ADDED BELOW ----------------------------------------
        # * set runtime state input parameters
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )
        if get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2:
            num_blocks_per_stage = len(self.transformer.blocks)
            get_runtime_state()._reset_recv_skip_buffer(num_blocks_per_stage)
        #! ---------------------------------------- ADDED ABOVE ----------------------------------------

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=77,
            text_encoder_index=0,
        )
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask_2,
            max_sequence_length=256,
            text_encoder_index=1,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7 create image_rotary_emb, style embedding & time ids
        grid_height = height // 8 // self.transformer.config.patch_size
        grid_width = width // 8 // self.transformer.config.patch_size
        base_size = 512 // 8 // self.transformer.config.patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size
        )
        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
        )

        style = torch.tensor([0], device=device)

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        # * dealing with cfg degree
        if self.do_classifier_free_guidance:
            (
                prompt_embeds,
                prompt_attention_mask,
            ) = self._process_cfg_split_batch(
                negative_prompt_embeds,
                prompt_embeds,
                negative_prompt_attention_mask,
                prompt_attention_mask,
            )
            (
                prompt_embeds_2,
                prompt_attention_mask_2,
            ) = self._process_cfg_split_batch(
                negative_prompt_embeds_2,
                prompt_embeds_2,
                negative_prompt_attention_mask_2,
                prompt_attention_mask_2,
            )
            if get_classifier_free_guidance_world_size() == 1:
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
                style = torch.cat([style] * 2, dim=0)

        #! ORIGIN
        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        #     prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        #     prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
        #     add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        #     style = torch.cat([style] * 2, dim=0)
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device)
        prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
        add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=device).repeat(
            batch_size * num_images_per_prompt, 1
        )
        style = style.to(device=device).repeat(batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > num_pipeline_warmup_steps
            ):
                # * warmup stage
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    prompt_embeds_2=prompt_embeds_2,
                    prompt_attention_mask_2=prompt_attention_mask_2,
                    add_time_ids=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    device=device,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                # * pipefusion stage
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    prompt_embeds_2=prompt_embeds_2,
                    prompt_attention_mask_2=prompt_attention_mask_2,
                    add_time_ids=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    device=device,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    prompt_embeds_2=prompt_embeds_2,
                    prompt_attention_mask_2=prompt_attention_mask_2,
                    add_time_ids=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    device=device,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

        # 8. Decode latents (only rank 0)
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        def vae_decode(latents):
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            return image
        
        if not output_type == "latent":
            if get_runtime_state().runtime_config.use_parallel_vae:
                latents = self.gather_broadcast_latents(latents)
                vae_decode(latents)
            else:
                if is_dp_last_group():
                    vae_decode(latents)
        if self.is_dp_last_group():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            if not output_type == "latent":
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        else:
            return None

    #! ---------------------------------------- ADD ABOVE ----------------------------------------

    def _init_sync_pipeline(self, latents: torch.Tensor, image_rotary_emb):
        latents = super()._init_sync_pipeline(latents)
        image_rotary_emb = (
            torch.cat(
                [
                    image_rotary_emb[0][start_token_idx:end_token_idx, ...]
                    for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                ],
                dim=0,
            ),
            torch.cat(
                [
                    image_rotary_emb[1][start_token_idx:end_token_idx, ...]
                    for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                ],
                dim=0,
            ),
        )
        return latents, image_rotary_emb

    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
    ):
        patch_latents = super()._init_async_pipeline(
            num_timesteps,
            latents,
            num_pipeline_warmup_steps,
        )

        if get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2:
            for _ in range(num_timesteps):
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_skip_task(patch_idx)

        return patch_latents

    # synchronized compute the whole feature map in each pp stage
    def _sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_attention_mask_2: torch.Tensor,
        add_time_ids: torch.Tensor,
        style: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        guidance_scale: float,
        guidance_rescale: float,
        timesteps: List[int],
        num_warmup_steps: int,
        extra_step_kwargs: List,
        progress_bar,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sync_only: bool = False,
    ):
        latents, image_rotary_emb = self._init_sync_pipeline(latents, image_rotary_emb)
        skips = None
        for i, t in enumerate(timesteps):
            if is_pipeline_last_stage():
                last_timestep_latents = latents

            # when there is only one pp stage, no need to recv
            if get_pipeline_parallel_world_size() == 1:
                pass
            # all ranks should recv the latent from the previous rank except
            #   the first rank in the first pipeline forward which should use
            #   the input latent
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if (
                    get_pipeline_parallel_rank()
                    >= get_pipeline_parallel_world_size() // 2
                ):
                    skips = get_pp_group().pipeline_recv_skip()

            latents = self._backbone_forward(
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                prompt_embeds_2=prompt_embeds_2,
                prompt_attention_mask_2=prompt_attention_mask_2,
                add_time_ids=add_time_ids,
                style=style,
                image_rotary_emb=image_rotary_emb,
                t=t,
                device=device,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                skips=skips,
            )

            if is_pipeline_last_stage():
                latents = self.scheduler.step(
                    latents,
                    t,
                    last_timestep_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
            elif (
                get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2
            ):
                pass
            else:
                latents, skips = latents

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )
                prompt_embeds_2 = callback_outputs.pop(
                    "prompt_embeds_2", prompt_embeds_2
                )
                negative_prompt_embeds_2 = callback_outputs.pop(
                    "negative_prompt_embeds_2", negative_prompt_embeds_2
                )

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if (
                    get_pipeline_parallel_rank()
                    < get_pipeline_parallel_world_size() // 2
                ):
                    get_pp_group().pipeline_send_skip(skips)

        if (
            sync_only
            and get_sequence_parallel_world_size() > 1
            and is_pipeline_last_stage()
        ):
            sp_degree = get_sequence_parallel_world_size()
            sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
            latents_list = []
            for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        :,
                        get_runtime_state()
                        .pp_patches_start_idx_local[pp_patch_idx] : get_runtime_state()
                        .pp_patches_start_idx_local[pp_patch_idx + 1],
                        :,
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
        prompt_embeds_2: torch.Tensor,
        prompt_attention_mask_2: torch.Tensor,
        add_time_ids: torch.Tensor,
        style: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        guidance_scale: float,
        guidance_rescale: float,
        timesteps: List[int],
        num_warmup_steps: int,
        extra_step_kwargs: List,
        progress_bar,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
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
        full_image_rotary_emb = image_rotary_emb
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if (is_pipeline_last_stage())
            else None
        )

        first_async_recv = True
        skips = None
        for i, t in enumerate(timesteps):
            for patch_idx in range(num_pipeline_patch):
                start_token_idx, end_token_idx = (
                    get_runtime_state().pp_patches_token_start_end_idx_global[patch_idx]
                )
                image_rotary_emb = (
                    full_image_rotary_emb[0][start_token_idx:end_token_idx, :],
                    full_image_rotary_emb[1][start_token_idx:end_token_idx, :],
                )

                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        get_pp_group().recv_next()
                        if (
                            get_pipeline_parallel_rank()
                            >= get_pipeline_parallel_world_size() // 2
                        ):
                            get_pp_group().recv_skip_next()
                        first_async_recv = False
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                        idx=patch_idx
                    )
                    if (
                        get_pipeline_parallel_rank()
                        >= get_pipeline_parallel_world_size() // 2
                    ):
                        skips = get_pp_group().get_pipeline_recv_skip_data(
                            idx=patch_idx
                        )
                patch_latents[patch_idx] = self._backbone_forward(
                    latents=patch_latents[patch_idx],
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    prompt_embeds_2=prompt_embeds_2,
                    prompt_attention_mask_2=prompt_attention_mask_2,
                    add_time_ids=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    t=t,
                    device=device,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    skips=skips,
                )
                if is_pipeline_last_stage():
                    patch_latents[patch_idx] = self.scheduler.step(
                        patch_latents[patch_idx],
                        t,
                        last_patch_latents[patch_idx],
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(
                            patch_latents[patch_idx], segment_idx=patch_idx
                        )
                elif (
                    get_pipeline_parallel_rank()
                    >= get_pipeline_parallel_world_size() // 2
                ):
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )
                else:
                    patch_latents[patch_idx], skips = patch_latents[patch_idx]
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )
                    get_pp_group().pipeline_isend_skip(skips)

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    else:
                        get_pp_group().recv_next()
                        if (
                            get_pipeline_parallel_rank()
                            >= get_pipeline_parallel_world_size() // 2
                        ):
                            get_pp_group().recv_skip_next()

                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )
                prompt_embeds_2 = callback_outputs.pop(
                    "prompt_embeds_2", prompt_embeds_2
                )
                negative_prompt_embeds_2 = callback_outputs.pop(
                    "negative_prompt_embeds_2", negative_prompt_embeds_2
                )

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=2)
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(
                    latents, separate_tensors=True
                )
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ...,
                            get_runtime_state()
                            .pp_patches_start_idx_local[
                                pp_patch_idx
                            ] : get_runtime_state()
                            .pp_patches_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
        return latents

    def _backbone_forward(
        self,
        latents: torch.FloatTensor,
        prompt_embeds: torch.FloatTensor,
        prompt_attention_mask: torch.FloatTensor,
        prompt_embeds_2: torch.FloatTensor,
        prompt_attention_mask_2: torch.FloatTensor,
        add_time_ids: torch.Tensor,
        style: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        t: Union[float, torch.Tensor],
        device: torch.device,
        guidance_scale: float,
        guidance_rescale: float,
        skips: torch.FloatTensor,
    ):
        if is_pipeline_first_stage():
            if self.do_classifier_free_guidance:
                latents = torch.cat(
                    [latents] * (2 // get_classifier_free_guidance_world_size())
                )
            latents = self.scheduler.scale_model_input(latents, t)

        # expand scalar t to 1-D tensor to match the 1st dim of latents
        t_expand = torch.tensor([t] * latents.shape[0], device=device).to(
            dtype=latents.dtype
        )

        # predict the noise residual
        noise_pred = self.transformer(
            latents,
            t_expand,
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attention_mask,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attention_mask_2,
            image_meta_size=add_time_ids,
            style=style,
            image_rotary_emb=image_rotary_emb,
            skips=skips,
            return_dict=False,
        )[0]

        if is_pipeline_last_stage():
            noise_pred, _ = noise_pred.chunk(2, dim=1)

            # perform guidance
            if self.do_classifier_free_guidance:
                if get_classifier_free_guidance_world_size() == 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                elif get_classifier_free_guidance_world_size() == 2:
                    noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                        noise_pred, separate_tensors=True
                    )
                latents = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                latents = rescale_noise_cfg(
                    latents, noise_pred_text, guidance_rescale=guidance_rescale
                )
        else:
            latents = noise_pred

        return latents
