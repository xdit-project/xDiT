# adpated from https://github.com/huggingface/diffusers/blob/v0.29.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_statble_diffusion_3 import (
    StableDiffusion3Pipeline,
    StableDiffusion3PipelineOutput,
    retrieve_timesteps
)
from typing import List, Optional, Union, Tuple, Callable, Dict, Final
from pipefuser.utils import DistriConfig, PipelineParallelismCommManager
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriSD3PiP(StableDiffusion3Pipeline):
    def init(self, distri_config: DistriConfig):
        # if distri_config.rank != 0 or distri_config.rank != distri_config.world_size - 1:
        # self.scheduler = None
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
        added_cond_kwargs: Dict,
        t: Union[float, torch.Tensor],
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        latent_channels: int,
    ):
        # if latents is None:
        #     logger.info(f"rank {self.distri_config.rank} latents is None")
        # else:
        #     logger.info(f"rank {self.distri_config.rank} latents shape {latents.shape}")
        distri_config = self.distri_config

        if distri_config.rank == 1:
            latents = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
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

        # predict noise model_output
        # logger.info(f"rank {self.distri_config.rank} latents shape {latents.shape}")
        noise_pred = self.transformer(
            latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=current_timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if distri_config.rank == 0:
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

        return noise_pred

    def scheduler_step(
        self,
        noise_pred: torch.FloatTensor,
        latents: torch.FloatTensor,
        t: Union[float, torch.Tensor],
        extra_step_kwargs: Dict,
        batch_idx: Union[int] = None,
    ):

        # compute previous image: x_t -> x_t-1
        latents = self.scheduler.step(
            noise_pred,
            t,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            batch_idx=batch_idx,
        )[0]

        return latents

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
    ) -> Union[StableDiffusionXLPipelineOutput, Tuple]:
       
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
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
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

        # 6. Denoising loop
        # assert self.comm_manager.recv_queue == []

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        #     if distri_config.mode != "full_sync":
        #         warmup_timesteps = timesteps[: distri_config.warmup_steps + 1]
        #         pip_timesteps = timesteps[distri_config.warmup_steps + 1 :]
        #     else:
        #         warmup_timesteps = timesteps
        #         pip_timesteps = None
        #     for i, t in enumerate(warmup_timesteps):
        #         if distri_config.rank == 0:
        #             ori_latents = latents

        #         if distri_config.rank == 1 and i == 0:
        #             pass
        #         else:
        #             self.comm_manager.irecv_from_prev(dtype)
        #             latents = self.comm_manager.get_data()

        #         latents = self.pip_forward(
        #             latents,
        #             prompt_embeds,
        #             prompt_attention_mask,
        #             added_cond_kwargs,
        #             t,
        #             do_classifier_free_guidance,
        #             guidance_scale,
        #             latent_channels,
        #         )

        #         if distri_config.rank == 0:
        #             latents = self.scheduler_step(
        #                 latents,
        #                 ori_latents,
        #                 t,
        #                 extra_step_kwargs,
        #             )

        #         if i == len(timesteps) - 1 or (
        #             (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        #         ):
        #             progress_bar.update()

        #         self.comm_manager.isend_to_next(latents)

        #     assert self.comm_manager.recv_queue == []

        #     if distri_config.rank == 1:
        #         self.comm_manager.irecv_from_prev(dtype)
        #         latents = self.comm_manager.get_data()
        #         for _ in range(len(pip_timesteps) - 1):
        #             for batch_idx in range(pp_num_patch):
        #                 self.comm_manager.irecv_from_prev(idx=batch_idx)
        #         _, _, c, _ = latents.shape
        #         latents = list(latents.split(c // pp_num_patch, dim=2))

        #     else:
        #         for _ in range(len(pip_timesteps)):
        #             for batch_idx in range(pp_num_patch):
        #                 self.comm_manager.irecv_from_prev(idx=batch_idx)
        #         if distri_config.rank == 0:
        #             _, _, c, _ = latents.shape
        #             c //= pp_num_patch
        #             tmp = latents
        #             latents = []
        #             for batch_idx in range(pp_num_patch):
        #                 latents.append(tmp[..., batch_idx * c : (batch_idx + 1) * c, :])
        #             ori_latents = [None for _ in range(pp_num_patch)]
        #         else:
        #             latents = [None for _ in range(pp_num_patch)]

        #     for i, t in enumerate(pip_timesteps):
        #         for batch_idx in range(pp_num_patch):

        #             if distri_config.rank == 0:
        #                 ori_latents[batch_idx] = latents[batch_idx]

        #             if distri_config.rank == 1 and i == 0:
        #                 pass
        #             else:
        #                 latents[batch_idx] = self.comm_manager.get_data(idx=batch_idx)

        #             latents[batch_idx] = self.pip_forward(
        #                 latents[batch_idx],
        #                 prompt_embeds,
        #                 prompt_attention_mask,
        #                 added_cond_kwargs,
        #                 t,
        #                 do_classifier_free_guidance,
        #                 guidance_scale,
        #                 latent_channels,
        #             )

        #             if distri_config.rank == 0:
        #                 latents[batch_idx] = self.scheduler_step(
        #                     latents[batch_idx],
        #                     ori_latents[batch_idx],
        #                     t,
        #                     extra_step_kwargs,
        #                     batch_idx,
        #                 )
        #                 # self.comm_manager.irecv_from_prev(idx=batch_idx)
        #                 if i != len(pip_timesteps) - 1:
        #                     self.comm_manager.isend_to_next(latents[batch_idx])

        #             else:
        #                 self.comm_manager.isend_to_next(latents[batch_idx])
        #         i += len(warmup_timesteps)
        #         if i == len(timesteps) - 1 or (
        #             (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        #         ):
        #             progress_bar.update()

        #     if distri_config.rank == 0:
        #         latents = torch.cat(latents, dim=2)
        #     else:
        #         return None

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image) 
