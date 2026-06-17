"""xFuser-wrapped Cosmos3OmniPipeline with CFG parallelism.

When cfg_parallel_size == 2, each GPU rank runs either the conditional or
unconditional transformer pass. Velocity predictions are gathered across
the CFG group and combined before the scheduler step.
"""

import copy
import os
from typing import Any, Callable, Optional, Union

import torch
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks

from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)

try:
    from cosmos_guardrail import CosmosSafetyChecker
except ImportError:
    CosmosSafetyChecker = None


def _make_xfuser_cosmos3_pipeline_class():
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
        Cosmos3OmniPipeline,
        Cosmos3OmniPipelineOutput,
    )

    class xFuserCosmos3OmniPipeline(Cosmos3OmniPipeline):

        @torch.no_grad()
        def __call__(self, *args, **kwargs):
            try:
                cfg_rank = get_classifier_free_guidance_rank()
                cfg_world_size = get_classifier_free_guidance_world_size()
            except (AssertionError, RuntimeError):
                cfg_rank = 0
                cfg_world_size = 1

            guidance_scale = kwargs.get("guidance_scale", 6.0)
            do_cfg_parallel = (guidance_scale != 1.0) and cfg_world_size == 2

            if not do_cfg_parallel:
                return super().__call__(*args, **kwargs)

            return self._call_with_cfg_parallel(*args, cfg_rank=cfg_rank, **kwargs)

        def _call_with_cfg_parallel(
            self,
            prompt,
            negative_prompt=None,
            image=None,
            num_frames=None,
            height=None,
            width=None,
            fps=24.0,
            num_inference_steps=35,
            guidance_scale=6.0,
            enable_sound=False,
            generator=None,
            latents=None,
            sound_latents=None,
            action_latents=None,
            action=None,
            output_type="pil",
            return_dict=True,
            use_system_prompt=True,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"],
            add_resolution_template=True,
            add_duration_template=True,
            enable_safety_check=True,
            cfg_rank=0,
        ):
            """Full pipeline with CFG parallelism injected into the denoising loop.

            Rank 0 runs the unconditional pass, rank 1 runs the conditional pass.
            After each step, velocity predictions are gathered and combined.
            """
            if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
                callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

            if action is None:
                if num_frames is None:
                    num_frames = 189
                if height is None:
                    height = 720
                if width is None:
                    width = 1280

            self.check_inputs(
                prompt, negative_prompt, image, height, width, num_frames,
                guidance_scale, enable_sound, callback_on_step_end_tensor_inputs, action,
            )

            action_mode = action.mode if action is not None else None
            if action is not None:
                from diffusers.video_processor import VideoProcessor
                from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _ACTION_RESOLUTION_BINS
                num_frames = action.chunk_size + 1
                conditioning_clip = [action.image] if action.image is not None else action.video
                probe = self.video_processor.preprocess_video(conditioning_clip)
                source_h, source_w = int(probe.shape[-2]), int(probe.shape[-1])
                resolution_key = str(action.resolution_tier)
                height, width = VideoProcessor.classify_height_width_bin(
                    source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
                )

            self._current_timestep = None
            self._interrupt = False
            self._guidance_scale = guidance_scale

            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if isinstance(negative_prompt, list):
                negative_prompt = negative_prompt[0] if negative_prompt else ""

            device = self._get_execution_device()
            dtype = self.transformer.dtype

            if enable_safety_check and CosmosSafetyChecker is not None and isinstance(getattr(self, 'safety_checker', None), CosmosSafetyChecker):
                self.safety_checker.to(device)
                try:
                    if not self.safety_checker.check_text_safety(prompt):
                        raise ValueError(f"Unsafe text detected: {prompt}")
                finally:
                    self.safety_checker.to("cpu")

            # Tokenize
            cond_input_ids, uncond_input_ids = self.tokenize_prompt(
                prompt, negative_prompt, num_frames=num_frames, height=height,
                width=width, fps=fps, use_system_prompt=use_system_prompt,
                add_resolution_template=add_resolution_template,
                add_duration_template=add_duration_template,
                action_mode=action_mode,
                action_view_point=action.view_point if action is not None else None,
            )

            # CFG parallel: each rank only prepares its own text segment
            if cfg_rank == 0:
                my_input_ids = uncond_input_ids
            else:
                my_input_ids = cond_input_ids

            my_text_segment = self._prepare_text_segment(my_input_ids, device=device)

            # Prepare latents (same on both ranks)
            (
                latents, sound_latents, action_latents, fps_vision, fps_sound,
                vision_condition_mask, sound_condition_mask, action_condition_mask,
                action_domain_id, action_image_size, raw_action_dim_resolved,
                action_condition_frame_indexes,
            ) = self.prepare_latents(
                image=image, num_frames=num_frames, height=height, width=width,
                fps=fps, latents=latents, sound_latents=sound_latents,
                action_latents=action_latents, generator=generator,
                device=device, dtype=dtype, enable_sound=enable_sound, action=action,
            )
            vision_condition_indexes = torch.nonzero(
                vision_condition_mask[:, 0, 0] > 0, as_tuple=False
            ).flatten()
            vision_condition_indexes = [int(idx.item()) for idx in vision_condition_indexes]
            has_image_condition = bool(vision_condition_indexes)

            # Prepare vision/sound segments (each rank uses its own text segment offset)
            my_vision_segment = self._prepare_vision_segment(
                input_vision_tokens=latents, has_image_condition=has_image_condition,
                mrope_offset=my_text_segment["vision_start_temporal_offset"],
                vision_fps=fps_vision, curr=my_text_segment["und_len"],
                device=device, condition_frame_indexes=vision_condition_indexes,
            )
            my_sound_segment = {}
            if sound_latents is not None:
                my_sound_segment = self._prepare_sound_segment(
                    input_sound_tokens=sound_latents,
                    mrope_offset=my_text_segment["vision_start_temporal_offset"],
                    sound_fps=fps_sound,
                    curr=my_text_segment["und_len"] + my_vision_segment["num_vision_tokens"],
                    device=device,
                )
            my_action_segment = {}
            if action_latents is not None:
                my_action_segment = self._prepare_action_segment(
                    input_action_tokens=action_latents,
                    condition_frame_indexes=action_condition_frame_indexes,
                    mrope_offset=my_text_segment["vision_start_temporal_offset"],
                    action_fps=fps_vision,
                    curr=my_text_segment["und_len"] + my_vision_segment["num_vision_tokens"] + my_sound_segment.get("sound_len", 0),
                    device=device,
                )

            mrope_segments = [my_text_segment["text_mrope_ids"], my_vision_segment["vision_mrope_ids"]]
            if my_sound_segment:
                mrope_segments.append(my_sound_segment["sound_mrope_ids"])
            if my_action_segment:
                mrope_segments.append(my_action_segment["action_mrope_ids"])

            my_packed_static = {
                **my_text_segment, **my_vision_segment, **my_sound_segment, **my_action_segment,
                "position_ids": torch.cat(mrope_segments, dim=1),
                "sequence_length": (my_text_segment["und_len"]
                    + my_vision_segment["num_vision_tokens"]
                    + my_sound_segment.get("sound_len", 0)
                    + my_action_segment.get("action_len", 0)),
            }

            num_noisy_vision_tokens = my_vision_segment["num_noisy_vision_tokens"]
            sound_len = my_sound_segment.get("sound_len")
            action_noisy_len = my_action_segment.get("num_noisy_action_tokens")

            # Schedulers
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            sound_scheduler = copy.deepcopy(self.scheduler) if sound_latents is not None else None
            action_scheduler = copy.deepcopy(self.scheduler) if action_latents is not None else None

            # Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    timestep = t.item()

                    vision_tokens = latents.to(device=device, dtype=dtype)
                    sound_tokens = sound_latents.to(device=device, dtype=dtype) if sound_latents is not None else None
                    action_tokens = action_latents.to(device=device, dtype=dtype) if action_latents is not None else None
                    vision_timesteps_t = torch.full((num_noisy_vision_tokens,), timestep, device=device)
                    sound_timesteps_t = torch.full((sound_len,), timestep, device=device) if sound_tokens is not None else None
                    action_timesteps_t = torch.full((action_noisy_len,), timestep, device=device) if action_tokens is not None else None

                    # Each rank runs ONE transformer pass (cond or uncond)
                    preds_vision, preds_sound, preds_action = self.transformer(
                        input_ids=my_packed_static["input_ids"],
                        text_indexes=my_packed_static["text_indexes"],
                        position_ids=my_packed_static["position_ids"],
                        und_len=my_packed_static["und_len"],
                        sequence_length=my_packed_static["sequence_length"],
                        vision_tokens=[vision_tokens],
                        vision_token_shapes=my_packed_static["vision_token_shapes"],
                        vision_sequence_indexes=my_packed_static["vision_sequence_indexes"],
                        vision_mse_loss_indexes=my_packed_static["vision_mse_loss_indexes"],
                        vision_timesteps=vision_timesteps_t,
                        vision_noisy_frame_indexes=my_packed_static["vision_noisy_frame_indexes"],
                        sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                        sound_token_shapes=my_packed_static.get("sound_token_shapes"),
                        sound_sequence_indexes=my_packed_static.get("sound_sequence_indexes"),
                        sound_mse_loss_indexes=my_packed_static.get("sound_mse_loss_indexes"),
                        sound_timesteps=sound_timesteps_t,
                        sound_noisy_frame_indexes=my_packed_static.get("sound_noisy_frame_indexes"),
                        action_tokens=[action_tokens] if action_tokens is not None else None,
                        action_token_shapes=my_packed_static.get("action_token_shapes"),
                        action_sequence_indexes=my_packed_static.get("action_sequence_indexes"),
                        action_mse_loss_indexes=my_packed_static.get("action_mse_loss_indexes"),
                        action_timesteps=action_timesteps_t,
                        action_noisy_frame_indexes=my_packed_static.get("action_noisy_frame_indexes"),
                        action_domain_ids=[action_domain_id] if action_domain_id is not None else None,
                    )
                    my_v_vision, my_v_sound, my_v_action = self._mask_velocity_predictions(
                        preds_vision, preds_sound,
                        vision_condition_mask=[vision_condition_mask],
                        sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                        preds_action=preds_action,
                        action_condition_mask=[action_condition_mask] if action_condition_mask is not None else None,
                        raw_action_dim=raw_action_dim_resolved,
                    )

                    # Gather velocity predictions across CFG ranks
                    # Rank 0 = uncond, Rank 1 = cond
                    uncond_v_vision, cond_v_vision = get_cfg_group().all_gather(
                        my_v_vision, separate_tensors=True
                    )
                    velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)

                    latents = self.scheduler.step(
                        velocity_vision.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    if sound_scheduler is not None and my_v_sound is not None:
                        uncond_v_sound, cond_v_sound = get_cfg_group().all_gather(
                            my_v_sound, separate_tensors=True
                        )
                        velocity_sound = uncond_v_sound + guidance_scale * (cond_v_sound - uncond_v_sound)
                        sound_latents = sound_scheduler.step(
                            velocity_sound.unsqueeze(0), t, sound_latents.unsqueeze(0), return_dict=False
                        )[0].squeeze(0)

                    has_noisy_action = (
                        action_condition_mask is not None and action_condition_mask.sum() < action_condition_mask.numel()
                    )
                    if action_scheduler is not None and has_noisy_action and my_v_action is not None:
                        uncond_v_action, cond_v_action = get_cfg_group().all_gather(
                            my_v_action, separate_tensors=True
                        )
                        velocity_action = uncond_v_action + guidance_scale * (cond_v_action - uncond_v_action)
                        action_latents = action_scheduler.step(
                            velocity_action.unsqueeze(0), t, action_latents.unsqueeze(0), return_dict=False
                        )[0].squeeze(0)
                        if raw_action_dim_resolved is not None:
                            action_latents[:, raw_action_dim_resolved:] = 0

                    if callback_on_step_end is not None:
                        callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            self._current_timestep = None

            # Decode (same as parent)
            sound = self.decode_sound(sound_latents) if sound_latents is not None else None
            action_output = None
            if action_mode in {"inverse_dynamics", "policy"} and action_latents is not None:
                action_output = action_latents
                if raw_action_dim_resolved is not None:
                    action_output = action_output[:, :raw_action_dim_resolved]
                action_output = [action_output.detach().cpu()]

            if output_type == "latent":
                video = latents
            else:
                in_dtype = latents.dtype
                vae_dtype = self.vae.dtype
                mean = self._vae_latents_mean.to(device=latents.device, dtype=vae_dtype)
                inv_std = self._vae_latents_inv_std.to(device=latents.device, dtype=vae_dtype)
                z_raw = latents.to(vae_dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
                decoded = self.vae.decode(z_raw).sample.to(in_dtype)
                video = self.video_processor.postprocess_video(decoded, output_type=output_type)[0]

            self.maybe_free_model_hooks()

            if not return_dict:
                return (video, sound)
            return Cosmos3OmniPipelineOutput(video=video, sound=sound, action=action_output)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, engine_config=None, **kwargs):
            pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            pipeline.__class__ = cls
            pipeline.engine_config = engine_config
            return pipeline

    return xFuserCosmos3OmniPipeline


_pipeline_cls = None

def get_cosmos3_pipeline_class():
    global _pipeline_cls
    if _pipeline_cls is None:
        _pipeline_cls = _make_xfuser_cosmos3_pipeline_class()
    return _pipeline_cls
