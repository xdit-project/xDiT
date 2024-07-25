import os
from typing import Dict, List, Tuple, Callable, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
from diffusers import PixArtAlphaPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from diffusers.utils import deprecate
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from pipefuser.model_executor.base_wrapper import PipeFuserBaseWrapper

from pipefuser.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_sequence_parallel_world_size, 
    get_sequence_parallel_rank, 
    get_pp_group,
    get_cfg_group,
    get_sp_group
)
from pipefuser.config import ParallelConfig, RuntimeConfig
from pipefuser.model_executor.pipelines import PipeFuserPipelineBaseWrapper
from .register import PipeFuserPipelineWrapperRegister


@PipeFuserPipelineWrapperRegister.register(PixArtAlphaPipeline)
class PipeFuserPixArtAlphaPipeline(PipeFuserPipelineBaseWrapper):
    def __init__(
        self,
        pipeline: PixArtAlphaPipeline,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            pipeline=pipeline,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
        **kwargs,
    ):
        pipeline = PixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, parallel_config, runtime_config)

    @PipeFuserBaseWrapper.forward_check_condition
    def prepare_run(self, steps: int = 3, sync_steps: int = 1):
        prompt = (
            [""] * self.input_config.batch_size
            if self.input_config.batch_size > 1
            else ""
        )
        self.__call__(
            height=self.input_config.height,
            width=self.input_config.width,
            prompt=prompt,
            use_resolution_binning=self.input_config.use_resolution_binning,
            num_inference_steps=steps,
            num_pipeline_warmup_steps=sync_steps,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(
                self.runtime_config.seed
            ),
        )

    @PipeFuserBaseWrapper.forward_check_condition
    @PipeFuserPipelineBaseWrapper.enable_data_parallel
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # * check pp world size
        if (
            get_pipeline_parallel_world_size() == 1
            and get_classifier_free_guidance_world_size() == 1
            and get_sequence_parallel_world_size() == 1
        ):
            return self.module(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                clean_caption=clean_caption,
                use_resolution_binning=use_resolution_binning,
                max_sequence_length=max_sequence_length,
                **kwargs,
            )

        # 0. parallel environment setting
        # TODO(Eigensystem) move check to decorator
        height = (
            height
            or self.input_config.orig_height
            or self.input_config.height
            or self.transformer.config.sample_size * self.vae_scale_factor
        )
        width = (
            width
            or self.input_config.orig_width
            or self.input_config.width
            or self.transformer.config.sample_size * self.vae_scale_factor
        )

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        # 1. Check inputs. Raise error if not correct
        orig_height = None
        orig_width = None
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )
        if isinstance(prompt, str):
            self._check_input_change_and_adjust(
                batch_size=1, height=height, width=width
            )
        elif isinstance(prompt, List):
            self._check_input_change_and_adjust(
                batch_size=len(prompt), height=height, width=width
            )

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        # * dealing with cfg degree
        if do_classifier_free_guidance:
            if get_classifier_free_guidance_world_size() == 1:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                prompt_attention_mask = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask], dim=0
                )
            elif get_classifier_free_guidance_rank() == 0:
                prompt_embeds = negative_prompt_embeds
                prompt_attention_mask = negative_prompt_attention_mask
            elif get_classifier_free_guidance_rank() == 1:
                prompt_embeds = prompt_embeds
                prompt_attention_mask = prompt_attention_mask
            else:
                raise ValueError("Invalid classifier free guidance rank")

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(
                batch_size * num_images_per_prompt, 1
            )
            aspect_ratio = torch.tensor([float(height / width)]).repeat(
                batch_size * num_images_per_prompt, 1
            )
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if (
                do_classifier_free_guidance
                and get_classifier_free_guidance_world_size() == 1
            ):
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        num_pipeline_warmup_steps = (
            kwargs.pop("num_pipeline_warmup_steps", None)
            or self.runtime_config.warmup_steps
        )
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
                    guidance_scale=guidance_scale,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    progress_bar=progress_bar,
                    callback=callback,
                    callback_steps=callback_steps,
                )
                # * pipefusion stage
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    guidance_scale=guidance_scale,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    num_pipeline_warmup_steps=num_pipeline_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    progress_bar=progress_bar,
                    callback=callback,
                    callback_steps=callback_steps,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    guidance_scale=guidance_scale,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    extra_step_kwargs=extra_step_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    progress_bar=progress_bar,
                    callback=callback,
                    callback_steps=callback_steps,
                    sync_only=True,
                )

        # 8. Decode latents (only rank 0)
        if (
            get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1
            and get_classifier_free_guidance_rank() == get_classifier_free_guidance_world_size() - 1
            and get_sequence_parallel_rank() == get_sequence_parallel_world_size() - 1
        ):
            if not output_type == "latent":
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor, return_dict=False
                )[0]
                if use_resolution_binning:
                    image = self.image_processor.resize_and_crop_tensor(
                        image, orig_width, orig_height
                    )
            else:
                image = latents

            if not output_type == "latent":
                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)
        else:
            return None

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
            for pp_patch_idx in range(self.num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        :, 
                        self.pp_patches_start_idx_local[pp_patch_idx]:
                        self.pp_patches_start_idx_local[pp_patch_idx+1],
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
        num_pipeline_warmup_steps: int,
        extra_step_kwargs: List,
        added_cond_kwargs: Dict,
        progress_bar,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = self.num_pipeline_patch
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

            num_pipeline_warmup_steps = self.runtime_config.warmup_steps
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
                for pp_patch_idx in range(self.num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ..., 
                            self.pp_patches_start_idx_local[pp_patch_idx]:
                            self.pp_patches_start_idx_local[pp_patch_idx+1],
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
