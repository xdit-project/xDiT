# adpated from https://github.com/huggingface/diffusers/blob/v0.29.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
import torch
from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetPipeline,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetPipeline,
    retrieve_timesteps,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from typing import List, Optional, Union, Tuple, Callable, Dict, Final, Any
from legacy.pipefuser.utils import DistriConfig, PipelineParallelismCommManager
from legacy.pipefuser.logger import init_logger
from legacy.pipefuser.modules.dit.pipefusion import DistriSD3CNTransformer2DModel
logger = init_logger(__name__)


class DistriSD3CNPiP(StableDiffusion3ControlNetPipeline):
    def init(self, distri_config: DistriConfig):
        # if distri_config.rank != 0 or distri_config.rank != distri_config.world_size - 1:
        # self.scheduler = None
        if distri_config.rank != 0:
            # self.vae = None
            pass
            # self.image_processor = None

        self.batch_idx = 0
        self.distri_config = distri_config

    def set_comm_manager(self, comm_manger: PipelineParallelismCommManager):
        self.comm_manager = comm_manger

    def pip_forward_cn(
        self,
        latents: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        
        t: Union[float, torch.Tensor],
        pooled_prompt_embeds: torch.FloatTensor,
        timestep_expand_shape: int,
        control_image: torch.Tensor,
        cond_scale:torch.Tensor,
        controlnet_pooled_projections: torch.Tensor,
        
        # block_controlnet_hidden_states: List = None,
    ):
        distri_config = self.distri_config

        if distri_config.rank == 1:
            # expand the latents if we are doing classifier free guidance
            latents = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            # dummy = self.transformer.model.module.pos_embed(latents)

        # print(self.transformer)
        # exit()
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        # timestep = t.expand(latents.shape[0])
        timestep = t.expand(timestep_expand_shape)
        # print("latents",latents.shape)
        # print("control image",control_image.shape)
        # print(self.controlnet)
        # exit()
        
        control_block_samples = self.controlnet(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=controlnet_pooled_projections,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )[0]
        # print(latents.shape)
        # print(self.controlnet)
        # exit()


        return control_block_samples
    def pip_forward(
        self,
        latents: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        
        t: Union[float, torch.Tensor],
        pooled_prompt_embeds: torch.FloatTensor,
        timestep_expand_shape: int,
        control_block_samples: List =  None,
    
        # block_controlnet_hidden_states: List = None,
    ):
        distri_config = self.distri_config

        if distri_config.rank == 1:
            # expand the latents if we are doing classifier free guidance
            latents = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            # dummy = self.transformer.model.module.pos_embed(latents)

        # print(self.transformer)
        # exit()
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        # timestep = t.expand(latents.shape[0])
        timestep = t.expand(timestep_expand_shape)
        # print("latents",latents.shape)
        # print("control image",control_image.shape)
        # print(self.controlnet)
        # exit()
        

        # print(latents.shape)
        # print(self.controlnet)
        # exit()
        noise_pred, encoder_hidden_states = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            block_controlnet_hidden_states = control_block_samples,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        if distri_config.rank == 0:
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        return noise_pred, encoder_hidden_states

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> Union[StableDiffusion3PipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        distri_config = self.distri_config
        assert callback_on_step_end is None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, SD3MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.model.module.dtype

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # print(self.transformer)
        # exit()
        # 3. Prepare control image
        if isinstance(self.controlnet.model.module, SD3ControlNetModel):
            # print(control_image)
            # exit()
            
            # print(latents.shape)
            # print(batch_size)
            # print(num_images_per_prompt)
            # print(control_image.shape)
            # exit()
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
            height_cn, width_cn = control_image.shape[-2:]

            control_image = self.vae.encode(control_image).latent_dist.sample()
            control_image = control_image * self.vae.config.scaling_factor
            # print(control_image.shape)
            # exit()
        elif isinstance(self.controlnet.model.module, SD3MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_image(
                    image=control_image_,
                    width=width_cn,
                    height=height_cn,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=False,
                )

                control_image_ = self.vae.encode(control_image_).latent_dist.sample()
                control_image_ = control_image_ * self.vae.config.scaling_factor

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        if controlnet_pooled_projections is None:
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = controlnet_pooled_projections or pooled_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

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

        # 6. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet.model.module, SD3ControlNetModel) else keeps)
        # print(controlnet_keep)
        # exit()
###########################################

        # 6. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        dtype = latents.dtype
        pp_num_patch = distri_config.pp_num_patch

        timestep_expand_shape = (
            latents.shape[0] * 2
            if self.do_classifier_free_guidance
            else latents.shape[0]
        )

        assert self.comm_manager.recv_queue == []

        assert distri_config.warmup_steps >= 1

        if distri_config.rank == 1:
            encoder_hidden_states = prompt_embeds
            # control_block_samples = []
        else:
            encoder_hidden_states = None
            # control_block_samples = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if distri_config.mode != "full_sync":
                warmup_timesteps = timesteps[: distri_config.warmup_steps + 1]
                pip_timesteps = timesteps[distri_config.warmup_steps + 1 :]
            else:
                warmup_timesteps = timesteps
                pip_timesteps = None

            for i, t in enumerate(warmup_timesteps):
                if distri_config.rank == 0:
                    ori_latents = latents

                if distri_config.rank == 1 and i == 0:
                    pass
                else:
                    self.comm_manager.irecv_from_prev(dtype)
                    latents = self.comm_manager.get_data()
                    # self.comm_manager.irecv_from_prev(control_image.dtype)
                    # control_image = self.comm_manager.get_data()
                    if distri_config.rank != 1:
                        encoder_hidden_states = self.comm_manager.recv_from_prev(
                            prompt_embeds.dtype, is_extra=True
                        )



                assert self._interrupt == False
                # TBD
                # if self.interrupt:
                # continue

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                # print(control_image.shape)
                next_control_block_samples = self.pip_forward_cn(
                    latents,
                    encoder_hidden_states,
                    t,
                    pooled_prompt_embeds,
                    timestep_expand_shape,
                    cond_scale=cond_scale,
                    control_image = control_image,
                    controlnet_pooled_projections = controlnet_pooled_projections
                )
                latents, next_encoder_hidden_states = self.pip_forward(
                    latents,
                    encoder_hidden_states,
                    t,
                    pooled_prompt_embeds,
                    timestep_expand_shape,
                    control_block_samples = next_control_block_samples
                )
                # print(latents.shape)
                # exit()
                if distri_config.rank == 0:
                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    control_image_dtype = control_image.dtype
                    
                    latents = self.scheduler.step(
                        latents, t, ori_latents, return_dict=False
                    )[0]
                    # print(latents.shape)
                    # print(control_image.shape)
                    # exit()
                    #controlnet
                    # control_image = self.scheduler.step(
                    #     control_image,t ,ori_latents, return_dict=False
                    # )[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if control_image.dtype != control_image_dtype: #controlnet
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            control_image = control_image.to(control_image_dtype)


                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                # TBD
                # if XLA_AVAILABLE:
                #     xm.mark_step()
                self.comm_manager.isend_to_next(latents)
                # self.comm_manager.isend_to_next(control_image) #controlnet
                if distri_config.rank != 0:
                    self.comm_manager.send_to_next(
                        next_encoder_hidden_states, is_extra=True
                    )
                    # for items in range(len(next_control_block_samples)):
                    #     self.comm_manager.send_to_next(
                    #         next_control_block_samples[items], is_extra=True
                    #     )


            assert self.comm_manager.recv_queue == []
            
            if distri_config.rank == 1:
                self.comm_manager.irecv_from_prev(dtype)
                latents = self.comm_manager.get_data()
                # control_image = self.comm_manager.get_data()

                for _ in range(len(pip_timesteps) - 1):
                    for batch_idx in range(pp_num_patch):
                        self.comm_manager.irecv_from_prev(idx=batch_idx)
                _, _, c, _ = latents.shape
                latents = list(latents.split(c // pp_num_patch, dim=2))
                # _, _, c, _ = control_image.shape
                control_image = list(control_image.split(c // pp_num_patch, dim=2))

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
                    c = control_image.shape[2]
                    c //= pp_num_patch
                    tmp = control_image
                    control_image = []
                    for batch_idx in range(pp_num_patch):
                        control_image.append(tmp[..., batch_idx * c : (batch_idx + 1) * c, :])
                    # ori_latents = [None for _ in range(pp_num_patch)]
                else:
                    latents = [None for _ in range(pp_num_patch)]
                    control_image = [None for _ in range(pp_num_patch)]

            for i, t in enumerate(pip_timesteps):

                assert self.interrupt == False

                if distri_config.rank != 1:
                    encoder_hidden_states = self.comm_manager.recv_from_prev(
                        prompt_embeds.dtype, is_extra=True
                    )
                    # for items in range(len(next_control_block_samples)):
                    #     next_control_block_samples[items] = self.comm_manager.recv_from_prev(
                    #         prompt_embeds.dtype, is_extra=True
                    #     )



                for batch_idx in range(pp_num_patch):

                    if distri_config.rank == 0:
                        ori_latents[batch_idx] = latents[batch_idx]
                        # ori_latents_cn[batch_idx] = control_image[batch_idx]

                    if distri_config.rank == 1 and i == 0:
                        pass
                    else:
                        latents[batch_idx] = self.comm_manager.get_data(idx=batch_idx)
                        # control_image[batch_idx] = self.comm_manager.get_data(idx=batch_idx)
                    # print(len(latents))

                    # exit()

                    next_control_block_samples = self.pip_forward_cn(
                    latents[batch_idx],
                    encoder_hidden_states,
                    t,
                    pooled_prompt_embeds,
                    timestep_expand_shape,
                    cond_scale=cond_scale,
                    control_image = control_image[batch_idx],
                    controlnet_pooled_projections = controlnet_pooled_projections

                )
                    latents[batch_idx], next_encoder_hidden_states = self.pip_forward(
                    latents[batch_idx],
                    encoder_hidden_states,
                    t,
                    pooled_prompt_embeds,
                    timestep_expand_shape,
                    next_control_block_samples
                )
                    if distri_config.rank == 0:
                        # compute the previous noisy sample x_t -> x_t-1
                        latents_dtype = latents[batch_idx].dtype
                        control_image_dtype = control_image[batch_idx].dtype
                        latents[batch_idx] = self.scheduler.step(
                            latents[batch_idx],
                            t,
                            ori_latents[batch_idx],
                            return_dict=False,
                            batch_idx=batch_idx,
                        )[0]

                        if latents[batch_idx].dtype != latents_dtype:
                            if torch.backends.mps.is_available():
                                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                latents[batch_idx] = latents[batch_idx].to(
                                    latents_dtype
                                )
                        if control_image[batch_idx].dtype != control_image_dtype:
                            if torch.backends.mps.is_available():
                                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                control_image[batch_idx] = control_image[batch_idx].to(
                                    control_image_dtype
                                )
                        if i != len(pip_timesteps) - 1:
                            self.comm_manager.isend_to_next(latents[batch_idx])
                            # self.comm_manager.isend_to_next(control_image[batch_idx])

                    else:
                        self.comm_manager.isend_to_next(latents[batch_idx])
                        # self.comm_manager.isend_to_next(control_image[batch_idx])

                    if distri_config.rank != 0 and batch_idx == 0:
                        self.comm_manager.send_to_next(
                            next_encoder_hidden_states, is_extra=True
                        )
                        # for items in range(len(next_control_block_samples)):
                        #     self.comm_manager.send_to_next(
                        #         next_control_block_samples[items], is_extra=True
                        #     ) 

                i += len(warmup_timesteps)
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

            # print("***********************Passed************************************")
            
            if distri_config.rank == 0:
                latents = torch.cat(latents, dim=2)

            else:
                return None

        if output_type == "latent":
            image = latents

        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
