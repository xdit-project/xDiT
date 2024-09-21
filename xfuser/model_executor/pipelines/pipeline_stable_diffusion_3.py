# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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

import torch
import torch.distributed
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_runtime_state,
    get_cfg_group,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_sequence_parallel_world_size,
    get_sp_group,
    is_dp_last_group,
    get_world_group
)
from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@xFuserPipelineWrapperRegister.register(StableDiffusion3Pipeline)
class xFuserStableDiffusion3Pipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)

    def prepare_run(
        self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1
    ):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

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
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
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
        **kwargs,
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

        #! ---------------------------------------- ADDED BELOW ----------------------------------------
        # * set runtime state input parameters
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )
        #! ---------------------------------------- ADDED ABOVE ----------------------------------------

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
            #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
            (
                prompt_embeds,
                pooled_prompt_embeds,
            ) = self._process_cfg_split_batch(
                negative_prompt_embeds,
                prompt_embeds,
                negative_pooled_prompt_embeds,
                pooled_prompt_embeds,
            )

            #! ORIGIN
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            # pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

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

        # 6. Denoising loop
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
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                # * pipefusion stage
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )

        # * 8. Decode latents (only the last rank in a dp group)

        if output_type != "latent":
            if get_runtime_state().runtime_config.use_parallel_vae:
                latents = get_world_group().broadcast_latent(latents,get_runtime_state().runtime_config.dtype)
                latents = (
                        latents / self.vae.config.scaling_factor
                    ) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
            else:
                if is_dp_last_group():
                    latents = (
                        latents / self.vae.config.scaling_factor
                    ) + self.vae.config.shift_factor
                    image = self.vae.decode(latents, return_dict=False)[0]
        
        if is_dp_last_group():
            if output_type == "latent":
                image = latents
            else:
                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusion3PipelineOutput(images=image)
        else:
            return None

    # synchronized compute the whole feature map in each pp stage
    def _sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sync_only: bool = False,
    ):
        latents = self._init_sync_pipeline(latents)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
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
                if not is_pipeline_first_stage():
                    encoder_hidden_states = get_pp_group().pipeline_recv(
                        0, "encoder_hidden_states"
                    )

            latents, encoder_hidden_states = self._backbone_forward(
                latents=latents,
                encoder_hidden_states=(
                    prompt_embeds
                    if is_pipeline_first_stage()
                    else encoder_hidden_states
                ),
                pooled_prompt_embeds=pooled_prompt_embeds,
                t=t,
            )

            if is_pipeline_last_stage():
                latents_dtype = latents.dtype
                latents = self._scheduler_step(latents, last_timestep_latents, t)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

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
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
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
                        encoder_hidden_states, name="encoder_hidden_states"
                    )

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

    def _init_sd3_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
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
                latents.split(get_runtime_state().pp_patches_height, dim=2)
            )
        elif is_pipeline_last_stage():
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_height, dim=2)
            )
        else:
            patch_latents = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]

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

        return patch_latents

    # * implement of pipefusion
    def _async_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
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
        patch_latents = self._init_sd3_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps,
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
                        latents=patch_latents[patch_idx],
                        encoder_hidden_states=(
                            prompt_embeds
                            if is_pipeline_first_stage()
                            else last_encoder_hidden_states
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds,
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
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        t: Union[float, torch.Tensor],
    ):
        if is_pipeline_first_stage():
            latents = torch.cat(
                [latents] * (2 // get_classifier_free_guidance_world_size())
            )

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0])

        noise_pred, encoder_hidden_states = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # classifier free guidance
        if is_pipeline_last_stage():
            if get_classifier_free_guidance_world_size() == 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            elif get_classifier_free_guidance_world_size() == 2:
                noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                    noise_pred, separate_tensors=True
                )
            latents = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            latents = noise_pred

        return latents, encoder_hidden_states

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
