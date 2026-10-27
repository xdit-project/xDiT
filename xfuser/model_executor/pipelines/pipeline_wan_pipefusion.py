"""PipeFusion pipeline for Wan2.2 TI2V image-to-video generation."""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

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
from xfuser.model_executor.models.transformers import transformer_wan as _transformer_wan  # noqa: F401
from xfuser.model_executor.pipefusion import (
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
    PipeFusionPatchSchedulerBank,
    PipeFusionTransport,
)

from .base_pipeline import xFuserPipelineBaseWrapper


class xFuserWanTI2VPipeFusionPipeline(xFuserPipelineBaseWrapper):
    """Run Wan on latent-height stripes that retain the full temporal extent."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *,
        engine_config: EngineConfig,
        cache_args: Optional[Dict] = None,
        **kwargs,
    ):
        pipeline = WanImageToVideoPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(pipeline, engine_config, cache_args or {})

    @property
    def interrupt(self):
        return self._interrupt

    def _model_input(self, latents, condition, first_frame_mask):
        return ((1 - first_frame_mask) * condition + first_frame_mask * latents).to(self.transformer.dtype)

    def _timestep(self, timestep, first_frame_mask, batch_size):
        _, patch_height, patch_width = self.transformer.config.patch_size
        values = (first_frame_mask[0][0][:, ::patch_height, ::patch_width] * timestep).flatten()
        return values.unsqueeze(0).expand(batch_size, -1)

    def _scheduler_step(self, noise_pred, previous, timestep):
        return self.scheduler.step(noise_pred, timestep, previous, return_dict=False)[0]

    def _backbone_forward(
        self,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        timestep,
        do_cfg,
        guidance_scale,
        *,
        pipeline_hidden_states=None,
        patch_start_height=0,
    ):
        cfg_size = get_classifier_free_guidance_world_size()
        model_input = (
            self._model_input(latents, condition, first_frame_mask)
            if pipeline_hidden_states is None
            else first_frame_mask.to(self.transformer.dtype)
        )
        if do_cfg and cfg_size == 1:
            model_input = torch.cat((model_input, model_input), dim=0)
        timestep_input = self._timestep(timestep, first_frame_mask, model_input.shape[0])
        cache_key = "uncond" if do_cfg and cfg_size == 2 and get_classifier_free_guidance_rank() == 0 else "cond"
        with self.transformer.cache_context(cache_key):
            output = self.transformer(
                hidden_states=model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds,
                pipeline_hidden_states=pipeline_hidden_states,
                patch_start_height=patch_start_height,
                return_dict=False,
            )[0]
        if not is_pipeline_last_stage():
            return output
        if do_cfg:
            if cfg_size == 2:
                negative, positive = get_cfg_group().all_gather(output, separate_tensors=True)
            else:
                negative, positive = output.chunk(2, dim=0)
            output = negative + guidance_scale * (positive - negative)
        return output

    def _sync_pipeline(
        self,
        *,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scale,
        num_warmup_steps,
        progress_bar,
        sync_only=False,
    ):
        state = get_runtime_state()
        state.set_patched_mode(False)
        for index, timestep in enumerate(timesteps):
            state.increment_step_counter()
            previous = latents if is_pipeline_last_stage() else None
            received = None
            if get_pipeline_parallel_world_size() > 1 and not (is_pipeline_first_stage() and index == 0):
                received = get_pp_group().pipeline_recv()
            if is_pipeline_first_stage() and received is not None:
                latents = received
            output = self._backbone_forward(
                latents,
                condition,
                first_frame_mask,
                prompt_embeds,
                timestep,
                do_cfg,
                guidance_scale,
                pipeline_hidden_states=None if is_pipeline_first_stage() else received,
            )
            if is_pipeline_last_stage():
                latents = self._scheduler_step(output, previous, timestep)
            else:
                latents = output
            if index == len(timesteps) - 1 or (
                (index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if not (sync_only and is_pipeline_last_stage() and index == len(timesteps) - 1):
                get_pp_group().pipeline_send(latents)
        return latents

    def _async_pipeline(
        self,
        *,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scale,
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
        condition_patches = layout.split(condition)
        mask_patches = layout.split(first_frame_mask)
        latent_shapes = [patch.shape for patch in patch_latents]
        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        hidden_shapes = [
            torch.Size((patch_latents[0].shape[0], token_count, hidden_size)) for token_count in layout.token_counts
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
        patch_schedulers = (
            PipeFusionPatchSchedulerBank(
                self.scheduler,
                layout,
                split_state_fields=("model_outputs", "last_sample"),
            )
            if is_pipeline_last_stage()
            else None
        )

        def begin_step(_step_index, _timestep):
            state.increment_step_counter()

        def prepare_patch(work, _timestep, received):
            patch_index = work.patch_index
            if is_pipeline_last_stage():
                previous[patch_index] = patch_latents[patch_index]
            if is_pipeline_first_stage() and received is not None:
                patch_latents[patch_index] = received
            return received

        def forward_patch(work, timestep, received):
            patch_index = work.patch_index
            return self._backbone_forward(
                patch_latents[patch_index],
                condition_patches[patch_index],
                mask_patches[patch_index],
                prompt_embeds,
                timestep,
                do_cfg,
                guidance_scale,
                pipeline_hidden_states=None if is_pipeline_first_stage() else received,
                patch_start_height=state.pp_patches_start_end_idx_global[patch_index][0],
            )

        def commit_patch(work, timestep, output):
            if is_pipeline_last_stage():
                patch_index = work.patch_index
                updated = patch_schedulers.step(
                    patch_index,
                    output,
                    timestep,
                    previous[patch_index],
                    return_dict=False,
                )[0]
                patch_latents[patch_index] = updated
                return None if work.step_index == work.num_steps - 1 else updated
            patch_latents[work.patch_index] = output
            return output

        def finalize():
            if not is_pipeline_last_stage():
                return None
            return torch.cat(patch_latents, dim=layout.split_dim)

        def complete_step(step_index, timestep):
            self._pipefusion_end_step(
                step_index,
                timestep,
                num_async_steps=len(timesteps),
                pipeline_warmup_steps=state.runtime_config.warmup_steps,
                num_warmup_steps=num_warmup_steps,
                progress_bar=progress_bar,
            )

        hooks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=finalize,
            begin_step_fn=begin_step,
            end_step_fn=complete_step,
        )
        return PipeFusionAsyncDriver(
            transport=transport,
            hooks=hooks,
            advance_patch=state.next_patch,
        ).run(timesteps)

    def _decode_latents(self, latents, first_frame_mask, condition, output_type, return_dict):
        if latents is not None and is_pipeline_last_stage():
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        def prepare(value):
            value = value.to(self.vae.dtype)
            mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(value)
            inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(value)
            return value / inv_std + mean

        video = None
        state = get_runtime_state()
        if output_type == "latent":
            video = latents
        elif state.runtime_config.use_parallel_vae and state.parallel_config.vae_parallel_size > 0:
            latents = self.gather_latents_for_vae(latents)
            self.send_to_vae_decode(prepare(latents) if latents is not None else None)
        elif state.runtime_config.use_parallel_vae:
            latents = self.gather_broadcast_latents(latents)
            video = self.vae.decode(prepare(latents), return_dict=False)[0]
        elif is_dp_last_group() and latents is not None:
            self._release_transformer_kv_cache()
            video = self.vae.decode(prepare(latents), return_dict=False)[0]
        if video is not None and output_type != "latent":
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        if not self.is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if callback_on_step_end is not None:
            raise NotImplementedError("callback_on_step_end is not supported by Wan TI2V PipeFusion.")
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        device = self._execution_device
        batch_size = (
            1 if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        )
        do_cfg = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        cfg_size = get_classifier_free_guidance_world_size()
        if do_cfg and cfg_size == 2:
            prompt_embeds = negative_prompt_embeds if get_classifier_free_guidance_rank() == 0 else prompt_embeds
        elif do_cfg:
            prompt_embeds = torch.cat((negative_prompt_embeds, prompt_embeds), dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition, first_frame_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            self.vae.config.z_dim,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        state = get_runtime_state()
        state.set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=False,
        )
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
                    condition=condition,
                    first_frame_mask=first_frame_mask,
                    prompt_embeds=prompt_embeds,
                    timesteps=timesteps[:warmup],
                    do_cfg=do_cfg,
                    guidance_scale=guidance_scale,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    sync_only=True,
                )
                latents = self._pipefusion_handoff_sync_latents(
                    latents,
                    name="pipefusion_sync_latents",
                )
                layout = PipeFusionPatchLayout.from_runtime_state(
                    state,
                    split_dim=3,
                    split_sizes=state.pp_patches_height,
                    name="wan-height-stripes",
                )
                with layout.install(state):
                    latents = self._async_pipeline(
                        latents=latents,
                        condition=condition,
                        first_frame_mask=first_frame_mask,
                        prompt_embeds=prompt_embeds,
                        timesteps=timesteps[warmup:],
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                        num_warmup_steps=num_warmup_steps,
                        progress_bar=progress_bar,
                        layout=layout,
                        sync_handoff_complete=True,
                    )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    condition=condition,
                    first_frame_mask=first_frame_mask,
                    prompt_embeds=prompt_embeds,
                    timesteps=timesteps,
                    do_cfg=do_cfg,
                    guidance_scale=guidance_scale,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    sync_only=True,
                )
        return self._decode_latents(latents, first_frame_mask, condition, output_type, return_dict)
