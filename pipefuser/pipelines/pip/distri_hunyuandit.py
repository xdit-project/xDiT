# adpated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
import torch
from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import (
    HunyuanDiTPipeline,
    SUPPORTED_SHAPE,
    map_to_standard_shapes,
    get_resize_crop_region_for_grid,
    rescale_noise_cfg,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import List, Optional, Union, Tuple, Callable, Dict, Final

import torch.distributed as dist
from pipefuser.utils import DistriConfig, PipelineParallelismCommManager
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriHunyuanDiTPiP(HunyuanDiTPipeline):
    def init(self, distri_config: DistriConfig):
        # if distri_config.rank != 0 or distri_config.rank != distri_config.world_size - 1:
        # self.scheduler = None
        # if torch.distributed.get_rank != 0:
        if distri_config.rank != 0:
            self.vae = None
            self.image_processor = None

        self.batch_idx = 0
        self.distri_config = distri_config

    def set_comm_manager(self, comm_manger: PipelineParallelismCommManager):
        self.comm_manager = comm_manger

    def pip_forward(
        self,
        latents: torch.FloatTensor,
        prompt_embeds: torch.FloatTensor,
        prompt_attention_mask: torch.FloatTensor,
        prompt_embeds_2: torch.FloatTensor,
        prompt_attention_mask_2: torch.FloatTensor,
        add_time_ids: torch.Tensor,
        style: torch.Tensor,
        image_rotary_emb: torch.FloatTensor,
        t: Union[float, torch.Tensor],
        device: torch.device,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_rescale: float,
        skips: torch.FloatTensor,
    ):
        # if latents is None:
        #     logger.info(f"rank {self.distri_config.rank} latents is None")
        # else:
        #     logger.info(f"rank {self.distri_config.rank} latents shape {latents.shape}")
        distri_config = self.distri_config

        if distri_config.rank == 1:
            # expand the latents if we are doing classifier free guidance
            latents = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance and not self.distri_config.split_batch
                else latents
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

        if distri_config.rank == 0:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

            # perform guidance
            if do_classifier_free_guidance:
                if not self.distri_config.split_batch:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                else:
                    noise_pred_uncond = torch.empty_like(noise_pred)
                    noise_pred_text = torch.empty_like(noise_pred)
                    dist.all_gather(
                        [noise_pred_uncond, noise_pred_text],
                        noise_pred,
                        group=self.distri_config.local_dp_group,
                    )
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

        return noise_pred

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

        distri_config = self.distri_config
        assert callback_on_step_end is None

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

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

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
            self.transformer.model.module.inner_dim
            // self.transformer.model.module.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
        )

        style = torch.tensor([0], device=device)

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        if self.do_classifier_free_guidance:
            if not distri_config.split_batch:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                prompt_attention_mask = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask]
                )
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                prompt_attention_mask_2 = torch.cat(
                    [negative_prompt_attention_mask_2, prompt_attention_mask_2]
                )
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
                style = torch.cat([style] * 2, dim=0)
            else:
                if distri_config.batch_idx() == 0:
                    prompt_embeds = negative_prompt_embeds
                    prompt_attention_mask = negative_prompt_attention_mask
                    prompt_embeds_2 = negative_prompt_embeds_2
                    prompt_attention_mask_2 = negative_prompt_attention_mask_2
                elif distri_config.batch_idx() == 1:
                    prompt_embeds = prompt_embeds
                    prompt_attention_mask = prompt_attention_mask
                    prompt_embeds_2 = prompt_embeds_2
                    prompt_attention_mask_2 = prompt_attention_mask_2
                else:
                    raise ValueError("Invalid batch_idx")

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

        dtype = latents.dtype
        pp_num_patch = distri_config.pp_num_patch

        assert self.comm_manager.recv_queue == []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if distri_config.mode != "full_sync":
                warmup_timesteps = timesteps[: distri_config.warmup_steps + 1]
                pip_timesteps = timesteps[distri_config.warmup_steps + 1 :]
            else:
                warmup_timesteps = timesteps
                pip_timesteps = None

            skips = None
            for i, t in enumerate(warmup_timesteps):
                if distri_config.rank == 0:
                    ori_latents = latents

                if distri_config.rank == 1 and i == 0:
                    pass
                else:
                    self.comm_manager.irecv_from_prev(dtype)
                    latents = self.comm_manager.get_data()
                    if (
                        distri_config.rank == 0
                        or distri_config.rank > distri_config.n_device_per_batch // 2
                    ):
                        self.comm_manager.recv_from_skip(
                            latents.dtype, idx=None, is_extra=True
                        )
                        skips = self.comm_manager.get_skip_data(idx=None, is_extra=True)

                latents = self.pip_forward(
                    latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    prompt_embeds_2,
                    prompt_attention_mask_2,
                    add_time_ids,
                    style,
                    image_rotary_emb,
                    t,
                    device,
                    self.do_classifier_free_guidance,
                    guidance_scale,
                    guidance_rescale,
                    skips,
                )

                if distri_config.rank == 0:
                    latents = self.scheduler.step(
                        latents,
                        t,
                        ori_latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                elif distri_config.rank > distri_config.n_device_per_batch // 2:
                    pass
                else:
                    latents, skips = latents

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                self.comm_manager.isend_to_next(latents)

                if (
                    distri_config.rank >= 1
                    and distri_config.rank <= distri_config.n_device_per_batch // 2
                ):
                    self.comm_manager.send_to_skip(skips, is_extra=True)

            assert self.comm_manager.recv_queue == []

            if (
                distri_config.rank == 0
                or distri_config.rank > distri_config.n_device_per_batch // 2
            ):
                for _ in range(len(pip_timesteps)):
                    for batch_idx in range(pp_num_patch):
                        self.comm_manager.recv_from_skip(
                            latents.dtype, idx=batch_idx, is_extra=True
                        )

            if distri_config.rank == 1:
                self.comm_manager.irecv_from_prev(dtype)
                latents = self.comm_manager.get_data()
                for _ in range(len(pip_timesteps) - 1):
                    for batch_idx in range(pp_num_patch):
                        self.comm_manager.irecv_from_prev(idx=batch_idx)
                _, _, c, _ = latents.shape
                latents = list(latents.split(c // pp_num_patch, dim=2))

            else:
                for _ in range(len(pip_timesteps)):
                    for batch_idx in range(pp_num_patch):
                        self.comm_manager.irecv_from_prev(idx=batch_idx)
                if distri_config.rank == 0:
                    _, _, c, _ = latents.shape
                    c //= pp_num_patch
                    tmp = latents
                    latents = []
                    for batch_idx in range(pp_num_patch):
                        latents.append(tmp[..., batch_idx * c : (batch_idx + 1) * c, :])
                    ori_latents = [None for _ in range(pp_num_patch)]
                else:
                    latents = [None for _ in range(pp_num_patch)]

            skips = None
            for i, t in enumerate(pip_timesteps):
                for batch_idx in range(pp_num_patch):

                    if distri_config.rank == 0:
                        ori_latents[batch_idx] = latents[batch_idx]

                    if distri_config.rank == 1 and i == 0:
                        pass
                    else:
                        latents[batch_idx] = self.comm_manager.get_data(idx=batch_idx)
                        if (
                            distri_config.rank == 0
                            or distri_config.rank
                            > distri_config.n_device_per_batch // 2
                        ):
                            skips = self.comm_manager.get_skip_data(
                                idx=batch_idx, is_extra=True
                            )

                    latents[batch_idx] = self.pip_forward(
                        latents[batch_idx],
                        prompt_embeds,
                        prompt_attention_mask,
                        prompt_embeds_2,
                        prompt_attention_mask_2,
                        add_time_ids,
                        style,
                        image_rotary_emb,
                        t,
                        device,
                        self.do_classifier_free_guidance,
                        guidance_scale,
                        guidance_rescale,
                        skips,
                    )

                    if distri_config.rank == 0:
                        latents[batch_idx] = self.scheduler.step(
                            latents[batch_idx],
                            t,
                            ori_latents[batch_idx],
                            **extra_step_kwargs,
                            return_dict=False,
                            batch_idx=batch_idx,
                        )[0]
                        # self.comm_manager.irecv_from_prev(idx=batch_idx)
                        if i != len(pip_timesteps) - 1:
                            self.comm_manager.isend_to_next(latents[batch_idx])
                    elif distri_config.rank > distri_config.n_device_per_batch // 2:
                        self.comm_manager.isend_to_next(latents[batch_idx])
                    else:
                        latents[batch_idx], skips = latents[batch_idx]
                        self.comm_manager.isend_to_next(latents[batch_idx])
                        self.comm_manager.send_to_skip(skips, is_extra=True)

                i += len(warmup_timesteps)
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

            if dist.get_rank() == 0:
                latents = torch.cat(latents, dim=2)
            else:
                return None

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
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
