# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
from diffusers import FluxControlPipeline
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.image_processor import PipelineImageInput

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_pp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    is_dp_last_group,
    get_world_group,
    get_vae_parallel_group,
    get_dit_world_size,
)
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

from PIL import Image


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

@xFuserPipelineWrapperRegister.register(FluxControlPipeline)
class xFuserPipelineFluxControlPipeline(xFuserPipelineBaseWrapper):
  
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict={},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = FluxControlPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if return_org_pipeline:
            return pipeline
        return cls(pipeline,engine_config, cache_args)
    
    def prepare_run(
        self,
        input_config: InputConfig,
        steps: int = 3,
        sync_steps: int = 1,
    ):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps

        #build a dummy input control image
        control_image = Image.new("RGB", (input_config.width, input_config.height), color=(0, 0, 0))

        self.__call__(
            height=input_config.height,
            width=input_config.width,
            control_image=control_image,
            prompt=prompt,  #TODO: will be removed and replaced with 'prompt_embeds' and 'pooled_prompt_embeds'
            num_inference_steps=steps,
            max_sequence_length=input_config.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.check_model_parallel_state(cfg_parallel_available=False)
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
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

        #! ---------------------------------------- ADDED BELOW ----------------------------------------
        # * set runtime state input parameters
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
        #! ---------------------------------------- ADDED ABOVE ----------------------------------------


        # 3. Prepare text embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 8

        control_image = self.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype,
        )

        if control_image.ndim == 4:
            control_image = self.vae.encode(control_image).latent_dist.sample(generator=generator)
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            height_control_image, width_control_image = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            )

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            #! ------------- New Code -------------
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > num_pipeline_warmup_steps
            ):
                # raise RuntimeError("Async pipeline not supported in flux")
                latents ,control_image = self._sync_pipeline(
                    latents=latents,
                    control_image=control_image,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                latents = self._async_pipeline(
                    latents=latents,
                    control_image=control_image,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents,control_image  = self._sync_pipeline(
                    latents=latents,
                    control_image=control_image,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )


        #! ---------- New Code ----------
        image = None
        def process_latents(latents):
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            return latents


        if output_type == "latent":
            image = latents
        else:
            # ------ New Code ------
            # copied from xfuser/model_executor/pipelines/pipeline_flux.py line: 368
            if get_runtime_state().runtime_config.use_parallel_vae and get_runtime_state().parallel_config.vae_parallel_size >0:
                # VAE is loaded in another worker
                latents = self.gather_latents_for_vae(latents)
                if latents is not None:
                    latents = process_latents(latents)
                self.send_to_vae_decode(latents)
            else:
                if get_runtime_state().runtime_config.use_parallel_vae:
                    latents = self.gather_broadcast_latents(latents)
                    latents = process_latents(latents)
                    image = self.vae.decode(latents, return_dict=False)[0]
                else:
                    if is_dp_last_group():
                        latents = process_latents(latents)
                        image = self.vae.decode(latents, return_dict=False)[0]

        

        if self.is_dp_last_group():
            if output_type == "latent":
                image = latents
            elif image is not None:
                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return FluxPipelineOutput(images=image)
        else:
            return None
        
    def _init_sync_pipeline(
        self, latents: torch.Tensor, latent_image_ids: torch.Tensor, 
        prompt_embeds: torch.Tensor, text_ids: torch.Tensor, control_image:torch.Tensor
    ):
        get_runtime_state().set_patched_mode(patch_mode=False)

        latents_list = [
            latents[:, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        latent_image_ids_list = [
            latent_image_ids[start_idx:end_idx]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latent_image_ids = torch.cat(latent_image_ids_list, dim=-2)

        control_image_list = [
            control_image[:, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        control_image = torch.cat(control_image_list, dim=-2)

        if get_runtime_state().split_text_embed_in_sp:
            if prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0:
                prompt_embeds = torch.chunk(prompt_embeds, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False                

        if get_runtime_state().split_text_embed_in_sp:
            if text_ids.shape[-2] % get_sequence_parallel_world_size() == 0:
                text_ids = torch.chunk(text_ids, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False                

        return latents, latent_image_ids, prompt_embeds, text_ids, control_image
    
    def _sync_pipeline(
        self,
        latents: torch.Tensor,
        control_image: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance,
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sync_only: bool = False,
    ):
        latents, latent_image_ids, prompt_embeds, text_ids, control_image = self._init_sync_pipeline(
            latents, latent_image_ids, prompt_embeds, text_ids, control_image)
        
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue


            if is_pipeline_last_stage():
                last_timestep_latents = latents

            # when there is only one pp stage, no need to recv
            if get_pipeline_parallel_world_size() == 1:
                pass

            # all ranks should recv the latent from the previous rank except
            # the first rank in the first pipeline forward which should use 
            # the input latent
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if not is_pipeline_first_stage():
                    encoder_hidden_state = get_pp_group().pipeline_recv(
                        0, "encoder_hidden_state"
                    )
                
            latents,encoder_hidden_state = self._backbone_forward(
                latents=(
                    torch.cat([latents, control_image], dim=2) if is_pipeline_first_stage() else  latents), #! This is important modify
                encoder_hidden_states=(
                    prompt_embeds if is_pipeline_first_stage() else encoder_hidden_state
                ),
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                guidance=guidance,
                t=t,
            )

            if is_pipeline_last_stage():
                latents_dtype = latents.dtype
                latents = self._scheduler_step(latents, last_timestep_latents, t)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if not is_pipeline_last_stage():
                    get_pp_group().pipeline_send(
                        encoder_hidden_state, name="encoder_hidden_state"
                    )

            # time step for-loop end
        
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
                        get_runtime_state()
                        .pp_patches_token_start_idx_local[pp_patch_idx] : get_runtime_state()
                        .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                        :,
                    ]
                    for sp_patch_idx in range(sp_degree)
                ]
            latents = torch.cat(latents_list, dim=-2)

        return latents, control_image
    
    def _async_pipeline(
        self,
        latents: torch.Tensor,
        control_image: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance,
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        patch_latents, patch_latent_image_ids, patch_control_image = self._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            control_image=control_image,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps,
            latent_image_ids=latent_image_ids,
        )
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if (is_pipeline_last_stage())
            else None
        )

        first_async_recv = True
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            for patch_idx in range(num_pipeline_patch):
                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        if not is_pipeline_first_stage() and patch_idx == 0:
                            get_pp_group().recv_next()
                        get_pp_group().recv_next()
                        first_async_recv = False

                    if not is_pipeline_first_stage() and patch_idx == 0:
                        last_encoder_hidden_states = (
                            get_pp_group().get_pipeline_recv_data(
                                idx=patch_idx, name="encoder_hidden_states"
                            )
                        )
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                        idx=patch_idx
                    )

                patch_latents[patch_idx], next_encoder_hidden_states = (
                    self._backbone_forward(
                        latents=(
                            torch.cat([patch_latents[patch_idx], patch_control_image[patch_idx]], 2)
                            if is_pipeline_first_stage()
                            else patch_latents[patch_idx]
                            ),
                        encoder_hidden_states=(
                            prompt_embeds
                            if is_pipeline_first_stage()
                            else last_encoder_hidden_states
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=patch_latent_image_ids[patch_idx],
                        guidance=guidance,
                        t=t,
                    )
                )
                if is_pipeline_last_stage():
                    latents_dtype = patch_latents[patch_idx].dtype
                    patch_latents[patch_idx] = self._scheduler_step(
                        patch_latents[patch_idx],
                        last_patch_latents[patch_idx],
                        t,
                    )

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds",
                            negative_pooled_prompt_embeds,
                        )

                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(
                            patch_latents[patch_idx], segment_idx=patch_idx
                        )
                else:
                    if patch_idx == 0:
                        get_pp_group().pipeline_isend(
                            next_encoder_hidden_states, name="encoder_hidden_states"
                        )
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    elif is_pipeline_first_stage():
                        get_pp_group().recv_next()
                    else:
                        # recv encoder_hidden_state
                        if patch_idx == num_pipeline_patch - 1:
                            get_pp_group().recv_next()
                        # recv latents
                        get_pp_group().recv_next()

                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=-2)
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
                            .pp_patches_token_start_idx_local[
                                pp_patch_idx
                            ] : get_runtime_state()
                            .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
        return latents

    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        control_image:torch.Tensor,
        num_pipeline_warmup_steps: int,
        latent_image_ids: torch.Tensor,
    ):
        get_runtime_state().set_patched_mode(patch_mode=True)

        if is_pipeline_first_stage():
            # get latents computed in warmup stage
            # ignore latents after the last timestep
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
            patch_control_image = list(
                control_image.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        elif is_pipeline_last_stage():
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
            patch_control_image = list(
                control_image.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        else:
            patch_latents = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]
            patch_control_image = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]

        patch_latent_image_ids = list(
            latent_image_ids[start_idx:end_idx]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        )

        recv_timesteps = (
            num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps
        )

        if is_pipeline_first_stage():
            for _ in range(recv_timesteps):
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)
        else:
            for _ in range(recv_timesteps):
                get_pp_group().add_pipeline_recv_task(0, "encoder_hidden_states")
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)

        return patch_latents, patch_latent_image_ids, patch_control_image

    def _backbone_forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids,
        latent_image_ids,
        guidance,
        t: Union[float, torch.Tensor],
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        ret = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        if self.engine_config.parallel_config.dit_parallel_size > 1:
            noise_pred, encoder_hidden_states = ret
        else:
            noise_pred, encoder_hidden_states = ret, None
        return noise_pred, encoder_hidden_states

    def _scheduler_step(
        self,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        t: Union[float, torch.Tensor],
    ):
        return self.scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
        )[0]
