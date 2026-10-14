"""PipeFusion pipeline for Z-Image generation."""

from contextlib import nullcontext
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image import (
    ZImagePipelineOutput,
    calculate_shift,
    get_default_z_image_sigmas,
    retrieve_timesteps,
)

from xfuser.config import EngineConfig
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_pp_group,
    get_runtime_state,
    is_dp_last_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.model_executor.models.transformers import transformer_z_image as _transformer_z_image  # noqa: F401
from xfuser.model_executor.pipefusion import (
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
    PipeFusionTransport,
)

from .base_pipeline import xFuserPipelineBaseWrapper


class xFuserZImagePipeFusionPipeline(xFuserPipelineBaseWrapper):
    """Run Z-Image on latent-height stripes with stage-local unified layers."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *,
        engine_config: EngineConfig,
        cache_args: Optional[Dict] = None,
        **kwargs,
    ):
        pipeline = ZImagePipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(pipeline, engine_config, cache_args or {})

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    @staticmethod
    def _validate_callback(callback_on_step_end):
        if callback_on_step_end is not None:
            raise NotImplementedError("callback_on_step_end is not supported by Z-Image PipeFusion.")

    def _cfg_prediction(self, output, guidance_scale):
        if not self.do_classifier_free_guidance:
            return output
        if get_classifier_free_guidance_world_size() == 2:
            positive, negative = get_cfg_group().all_gather(output, separate_tensors=True)
        else:
            positive, negative = output.chunk(2, dim=0)
        prediction = positive + guidance_scale * (positive - negative)
        if self._cfg_normalization and float(self._cfg_normalization) > 0:
            dimensions = tuple(range(1, prediction.ndim))
            positive_norm = torch.linalg.vector_norm(positive, dim=dimensions, keepdim=True)
            prediction_norm = torch.linalg.vector_norm(prediction, dim=dimensions, keepdim=True)
            maximum = positive_norm * float(self._cfg_normalization)
            prediction = torch.where(
                prediction_norm > maximum,
                prediction * (maximum / prediction_norm.clamp_min(torch.finfo(prediction.dtype).eps)),
                prediction,
            )
        return prediction

    def _backbone_forward(
        self,
        latents,
        prompt_embeds,
        timestep,
        guidance_scale,
        *,
        pipeline_hidden_states=None,
        patch_start_height=0,
    ):
        cfg_size = get_classifier_free_guidance_world_size()
        model_input = latents.to(self.transformer.dtype)
        if self.do_classifier_free_guidance and cfg_size == 1:
            model_input = model_input.repeat(2, 1, 1, 1)
        timestep_input = ((1000 - timestep.expand(model_input.shape[0])) / 1000).to(model_input.device)
        model_input = list(model_input.unsqueeze(2).unbind(dim=0))
        cache_key = (
            "uncond"
            if self.do_classifier_free_guidance and cfg_size == 2 and get_classifier_free_guidance_rank() == 1
            else "cond"
        )
        cache_context = getattr(self.transformer, "cache_context", None)
        with cache_context(cache_key) if cache_context is not None else nullcontext():
            output = self.transformer(
                model_input,
                timestep_input,
                prompt_embeds,
                pipeline_hidden_states=pipeline_hidden_states,
                patch_start_height=patch_start_height,
                return_dict=False,
            )[0]
        if not is_pipeline_last_stage():
            return output
        output = torch.stack(output).squeeze(2)
        return self._cfg_prediction(output, guidance_scale)

    def _scheduler_step(self, noise_pred, timestep, latents):
        return self.scheduler.step(
            -noise_pred.to(torch.float32),
            timestep,
            latents,
            return_dict=False,
        )[0]

    def _sync_pipeline(
        self,
        *,
        latents,
        prompt_embeds,
        timesteps,
        guidance_scales,
        num_warmup_steps,
        progress_bar,
    ):
        state = get_runtime_state()
        state.set_patched_mode(False)
        for index, (timestep, guidance_scale) in enumerate(zip(timesteps, guidance_scales)):
            previous = latents if is_pipeline_last_stage() else None
            received = (
                get_pp_group().pipeline_recv()
                if get_pipeline_parallel_world_size() > 1 and not is_pipeline_first_stage()
                else None
            )
            output = self._backbone_forward(
                latents,
                prompt_embeds,
                timestep,
                guidance_scale,
                pipeline_hidden_states=None if is_pipeline_first_stage() else received,
            )
            if is_pipeline_last_stage():
                latents = self._scheduler_step(output, timestep, previous)
            else:
                latents = output
                get_pp_group().pipeline_send(latents)
            latents = self._pipefusion_handoff_sync_latents(
                latents,
                name="z_image_sync_latents",
            )
            if index == len(timesteps) - 1 or (
                (index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
        return latents

    def _async_pipeline(
        self,
        *,
        latents,
        prompt_embeds,
        timesteps,
        guidance_scales,
        num_warmup_steps,
        progress_bar,
        layout,
        sync_handoff_complete=False,
    ):
        state = get_runtime_state()
        patch_latents = self._init_async_pipeline(
            len(timesteps),
            latents,
            0 if sync_handoff_complete else state.runtime_config.warmup_steps,
            split_sizes=layout.split_sizes,
            split_dim=layout.split_dim,
            queue_receives=False,
        )
        if not is_pipeline_first_stage() and not is_pipeline_last_stage():
            # Z-Image still needs each patch's latent geometry on intermediate
            # stages to build masks and globally offset RoPE.
            patch_latents = layout.split(latents)
        latent_shapes = [patch.shape for patch in patch_latents]
        cfg_batch = patch_latents[0].shape[0]
        if self.do_classifier_free_guidance and get_classifier_free_guidance_world_size() == 1:
            cfg_batch *= 2
        caption_tokens = max(len(value) + (-len(value)) % 32 for value in prompt_embeds)
        hidden_shapes = [
            torch.Size((cfg_batch, token_count + caption_tokens, self.transformer.config.dim))
            for token_count in layout.token_counts
        ]
        transport = PipeFusionTransport(
            get_pp_group(),
            first_stage=is_pipeline_first_stage(),
            num_steps=len(timesteps),
            num_patches=layout.num_patches,
            recv_shapes=latent_shapes if is_pipeline_first_stage() else hidden_shapes,
            send_shapes=latent_shapes if is_pipeline_last_stage() else hidden_shapes,
        )
        previous = [None] * layout.num_patches if is_pipeline_last_stage() else None

        def prepare_patch(work, _timestep, received):
            patch_index = work.patch_index
            if is_pipeline_last_stage():
                previous[patch_index] = patch_latents[patch_index]
            if is_pipeline_first_stage() and received is not None:
                patch_latents[patch_index] = received
            return received

        def forward_patch(work, timestep_and_guidance, received):
            timestep, guidance_scale = timestep_and_guidance
            patch_index = work.patch_index
            return self._backbone_forward(
                patch_latents[patch_index],
                prompt_embeds,
                timestep,
                guidance_scale,
                pipeline_hidden_states=None if is_pipeline_first_stage() else received,
                patch_start_height=state.pp_patches_start_end_idx_global[patch_index][0],
            )

        def commit_patch(work, timestep_and_guidance, output):
            if is_pipeline_last_stage():
                timestep, _guidance_scale = timestep_and_guidance
                patch_index = work.patch_index
                updated = self._scheduler_step(
                    output,
                    timestep,
                    previous[patch_index],
                )
                patch_latents[patch_index] = updated
                return None if work.step_index == work.num_steps - 1 else updated
            if is_pipeline_first_stage():
                patch_latents[work.patch_index] = output
            return output

        def end_step(step_index, timestep_and_guidance):
            self._pipefusion_end_step(
                step_index,
                timestep_and_guidance[0],
                num_async_steps=len(timesteps),
                pipeline_warmup_steps=state.runtime_config.warmup_steps,
                num_warmup_steps=num_warmup_steps,
                progress_bar=progress_bar,
            )

        def finalize():
            if not is_pipeline_last_stage():
                return None
            return torch.cat(patch_latents, dim=layout.split_dim)

        hooks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=finalize,
            end_step_fn=end_step,
        )
        timestep_inputs = tuple(zip(timesteps, guidance_scales))
        return PipeFusionAsyncDriver(
            transport=transport,
            hooks=hooks,
            advance_patch=state.next_patch,
        ).run(timestep_inputs)

    def _decode_latents(self, latents, output_type, return_dict):
        image = None

        def prepare(value):
            value = value.to(self.vae.dtype)
            return value / self.vae.config.scaling_factor + self.vae.config.shift_factor

        state = get_runtime_state()
        if output_type == "latent":
            image = latents
        elif state.runtime_config.use_parallel_vae and state.parallel_config.vae_parallel_size > 0:
            latents = self.gather_latents_for_vae(latents)
            self.send_to_vae_decode(prepare(latents) if latents is not None else None)
        elif state.runtime_config.use_parallel_vae:
            latents = self.gather_broadcast_latents(latents)
            image = self.vae.decode(prepare(latents), return_dict=False)[0]
        elif is_dp_last_group():
            self._release_transformer_kv_cache()
            image = self.vae.decode(prepare(latents), return_dict=False)[0]
        if image is not None and output_type != "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)
        if not is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        del joint_attention_kwargs, callback_on_step_end_tensor_inputs
        self._validate_callback(callback_on_step_end)
        height = height or 1024
        width = width or 1024
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale or width % vae_scale:
            raise ValueError(f"Z-Image height and width must be divisible by {vae_scale}.")
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        self._interrupt = False
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else len(prompt_embeds)
        if prompt_embeds is None or prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        elif self.do_classifier_free_guidance and negative_prompt_embeds is None:
            raise ValueError("negative_prompt_embeds is required with prompt_embeds and classifier-free guidance.")
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = [value for value in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = [
                    value for value in negative_prompt_embeds for _ in range(num_images_per_prompt)
                ]
        if self.do_classifier_free_guidance:
            if get_classifier_free_guidance_world_size() == 2:
                prompt_embeds = prompt_embeds if get_classifier_free_guidance_rank() == 0 else negative_prompt_embeds
            else:
                prompt_embeds = prompt_embeds + negative_prompt_embeds

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = sigmas if sigmas is not None else get_default_z_image_sigmas(num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self.scheduler.set_begin_index(0)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        guidance_scales = []
        for timestep in timesteps:
            scale = guidance_scale
            if self.do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if float((1000 - timestep.float()) / 1000) > cfg_truncation:
                    scale = 0.0
            guidance_scales.append(scale)

        state = get_runtime_state()
        state.set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max(len(value) for value in prompt_embeds),
            split_text_embed_in_sp=False,
            split_latents_by_rows=True,
        )
        self.transformer._pipefusion_context_cache = None
        warmup = state.runtime_config.warmup_steps
        run_async = self._pipefusion_async_enabled(
            num_timesteps=len(timesteps),
            pipeline_warmup_steps=warmup,
            callback_on_step_end=callback_on_step_end,
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if run_async:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    timesteps=timesteps[:warmup],
                    guidance_scales=guidance_scales[:warmup],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                )
                padded_tokens = [count + (-count) % 32 for count in state.pp_patches_token_num]
                layout = PipeFusionPatchLayout.from_runtime_state(
                    state,
                    split_dim=2,
                    token_counts=padded_tokens,
                    name="z-image-height-stripes",
                )
                with layout.install(state):
                    latents = self._async_pipeline(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        timesteps=timesteps[warmup:],
                        guidance_scales=guidance_scales[warmup:],
                        num_warmup_steps=num_warmup_steps,
                        progress_bar=progress_bar,
                        layout=layout,
                        sync_handoff_complete=True,
                    )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    timesteps=timesteps,
                    guidance_scales=guidance_scales,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                )
        return self._decode_latents(latents, output_type, return_dict)
