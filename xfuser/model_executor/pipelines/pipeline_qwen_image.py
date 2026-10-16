from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import torch
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

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
from xfuser.model_executor.pipefusion import (
    PipeFusionImagePatchSchedule,
    PipeFusionPatchForward,
    PipeFusionPatchLayout,
    PipeFusionTransport,
)
from xfuser.model_executor.models.transformers import transformer_qwen as _transformer_qwen  # noqa: F401

from .base_pipeline import xFuserPipelineBaseWrapper


class QwenPipeFusionMixin:
    """Shared token-row PipeFusion driver for Qwen Image and base Image Edit."""

    @staticmethod
    def _validate_qwen_callback(callback_on_step_end):
        if callback_on_step_end is not None:
            raise NotImplementedError(
                "callback_on_step_end is not supported by Qwen Image PipeFusion "
                "because callback state cannot be applied consistently across stages."
            )

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def _set_qwen_runtime(self, *, height, width, batch_size, num_inference_steps, max_sequence_length):
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=False,
            split_latents_by_rows=True,
        )

    def _select_cfg_condition(self, cond, uncond, cond_mask, uncond_mask):
        cfg_size = get_classifier_free_guidance_world_size()
        if not self._qwen_do_true_cfg:
            return cond, cond_mask
        if cfg_size == 2:
            if get_classifier_free_guidance_rank() == 0:
                return uncond, uncond_mask
            return cond, cond_mask
        if uncond_mask is None and cond_mask is None:
            mask = None
        else:
            uncond_mask = (
                uncond_mask
                if uncond_mask is not None
                else torch.ones(uncond.shape[:2], dtype=torch.bool, device=uncond.device)
            )
            cond_mask = (
                cond_mask if cond_mask is not None else torch.ones(cond.shape[:2], dtype=torch.bool, device=cond.device)
            )
            mask = torch.cat((uncond_mask, cond_mask), dim=0)
        return torch.cat((uncond, cond), dim=0), mask

    def _cfg_prediction(self, prediction):
        if not self._qwen_do_true_cfg:
            return prediction
        if get_classifier_free_guidance_world_size() == 2:
            negative, positive = get_cfg_group().all_gather(prediction, separate_tensors=True)
        else:
            negative, positive = prediction.chunk(2, dim=0)
        combined = negative + self._qwen_true_cfg_scale * (positive - negative)
        positive_norm = torch.norm(positive, dim=-1, keepdim=True)
        combined_norm = torch.norm(combined, dim=-1, keepdim=True)
        return combined * (positive_norm / combined_norm)

    @contextmanager
    def _cache_context(self, branch):
        # The named Cache-DiT context isolates stage residuals between
        # sequential true-CFG branches; the side map does the same for
        # PipeFusion attention K/V.
        cache_context = getattr(self.transformer, "cache_context", None)
        model_context = cache_context(branch) if cache_context is not None else nullcontext()
        branch_caches = getattr(self, "_qwen_branch_kv", None)
        attention_layers = (
            [block.attn for block in self.transformer.transformer_blocks] if branch_caches is not None else []
        )
        with model_context:
            for layer in attention_layers:
                layer._xdit_kv_cache = branch_caches[branch].get(layer)
            try:
                yield
            finally:
                for layer in attention_layers:
                    branch_caches[branch][layer] = layer._xdit_kv_cache

    def _backbone_forward(
        self,
        latents,
        encoder_hidden_states,
        t,
        *,
        encoder_hidden_states_mask,
        guidance,
        img_shapes,
        image_rotary_emb=None,
    ):
        pure_cfg = self._qwen_do_true_cfg and get_classifier_free_guidance_world_size() == 1
        if pure_cfg:
            if latents.shape[0] != encoder_hidden_states.shape[0]:
                latents = torch.cat((latents, latents), dim=0)
            branches = ("uncond", "cond")
        else:
            branches = (
                "uncond"
                if self._qwen_do_true_cfg
                and get_classifier_free_guidance_world_size() == 2
                and get_classifier_free_guidance_rank() == 0
                else "cond",
            )

        image_outputs = []
        text_outputs = []
        image_states = latents.chunk(len(branches), dim=0)
        text_states = encoder_hidden_states.chunk(len(branches), dim=0)
        text_masks = (
            encoder_hidden_states_mask.chunk(len(branches), dim=0)
            if encoder_hidden_states_mask is not None
            else (None,) * len(branches)
        )
        for branch, image_state, text_state, text_mask in zip(
            branches,
            image_states,
            text_states,
            text_masks,
        ):
            timestep = t.expand(image_state.shape[0]).to(image_state.dtype) / 1000
            with self._cache_context(branch):
                result = self.transformer(
                    hidden_states=image_state,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=text_mask,
                    encoder_hidden_states=text_state,
                    img_shapes=img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
            if isinstance(result, tuple):
                image_output, text_output = result
            else:
                image_output, text_output = result, None
            image_outputs.append(image_output)
            if text_output is not None:
                text_outputs.append(text_output)

        image_output = torch.cat(image_outputs, dim=0) if len(image_outputs) > 1 else image_outputs[0]
        text_output = torch.cat(text_outputs, dim=0) if text_outputs else None
        return image_output, text_output

    def _scheduler_step(self, noise_pred, latents, timestep):
        return self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def _pipefusion_update_qwen_patch(self, work, timestep, noise_pred, previous):
        if noise_pred.shape[1] == 0:
            return previous
        return self._pipefusion_update_patch(work, timestep, self._cfg_prediction(noise_pred), previous)

    def _sync_qwen_pipeline(
        self,
        *,
        latents,
        image_latents,
        condition,
        condition_mask,
        guidance,
        img_shapes,
        generated_tokens,
        timesteps,
        num_warmup_steps,
        progress_bar,
        sync_only=False,
    ):
        state = get_runtime_state()
        state.set_patched_mode(False)
        model_latents = torch.cat((latents, image_latents), dim=1) if image_latents is not None else latents
        encoder_state = condition
        for index, timestep in enumerate(timesteps):
            previous = latents if is_pipeline_last_stage() else None
            if get_pipeline_parallel_world_size() > 1 and not (is_pipeline_first_stage() and index == 0):
                model_latents = get_pp_group().pipeline_recv()
                if not is_pipeline_first_stage():
                    encoder_state = get_pp_group().pipeline_recv(0, "encoder_hidden_state")
            if image_latents is not None and is_pipeline_first_stage() and model_latents.shape[1] == generated_tokens:
                model_latents = torch.cat((model_latents, image_latents.to(model_latents)), dim=1)

            model_latents, encoder_state = self._backbone_forward(
                model_latents,
                condition if is_pipeline_first_stage() else encoder_state,
                timestep,
                encoder_hidden_states_mask=condition_mask,
                guidance=guidance,
                img_shapes=img_shapes,
            )
            if is_pipeline_last_stage():
                noise_pred = self._cfg_prediction(model_latents[:, :generated_tokens])
                latents = self._scheduler_step(noise_pred, previous, timestep)
                model_latents = latents
            if index == len(timesteps) - 1 or (
                (index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if not (sync_only and is_pipeline_last_stage() and index == len(timesteps) - 1):
                if get_pipeline_parallel_world_size() > 1:
                    get_pp_group().pipeline_send(model_latents)
                    if not is_pipeline_last_stage():
                        get_pp_group().pipeline_send(encoder_state, name="encoder_hidden_state")
        return latents if is_pipeline_last_stage() else model_latents

    def _patch_rotary_embeddings(self, layout, img_shapes, text_tokens, device):
        image_freqs, text_freqs = self.transformer.pos_embed(
            img_shapes,
            max_txt_seq_len=text_tokens,
            device=device,
        )
        return [(patch, text_freqs) for patch in image_freqs.split(layout.split_sizes, dim=0)]

    def _async_qwen_pipeline(
        self,
        *,
        latents,
        image_latents,
        condition,
        condition_mask,
        guidance,
        img_shapes,
        timesteps,
        num_warmup_steps,
        progress_bar,
        layout,
        sync_handoff_complete=False,
    ):
        if len(timesteps) == 0:
            return latents
        state = get_runtime_state()
        patch_latents = super()._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=0 if sync_handoff_complete else state.runtime_config.warmup_steps,
            split_sizes=layout.split_sizes,
            split_dim=1,
            queue_receives=False,
            appended_latents=image_latents,
        )
        patch_rope = self._patch_rotary_embeddings(
            layout,
            img_shapes,
            condition_mask.shape[1] if condition_mask is not None else condition.shape[1],
            condition.device,
        )
        transport = PipeFusionTransport(
            get_pp_group(),
            first_stage=is_pipeline_first_stage(),
            num_steps=len(timesteps),
            num_patches=layout.num_patches,
        )
        return PipeFusionImagePatchSchedule(
            patch_latents=patch_latents,
            layout=layout,
            initial_condition=condition,
            first_stage=is_pipeline_first_stage(),
            last_stage=is_pipeline_last_stage(),
            condition_reuse=True,
            forward_patch_fn=PipeFusionPatchForward(
                self._backbone_forward,
                static_kwargs={
                    "encoder_hidden_states_mask": condition_mask,
                    "guidance": guidance,
                    "img_shapes": img_shapes,
                },
                patch_kwargs={"image_rotary_emb": patch_rope},
            ),
            update_last_patch_fn=self._pipefusion_update_qwen_patch,
            finalize_fn=partial(
                self._pipefusion_finalize_token_patches,
                patch_latents=patch_latents,
                state=state,
                layout=layout,
            ),
            end_step_fn=partial(
                self._pipefusion_end_step,
                num_async_steps=len(timesteps),
                pipeline_warmup_steps=state.runtime_config.warmup_steps,
                num_warmup_steps=num_warmup_steps,
                progress_bar=progress_bar,
            ),
            model_name="Qwen Image PipeFusion",
        ).run(
            timesteps=timesteps,
            transport=transport,
            advance_patch=state.next_patch,
        )

    def _denoise_qwen(
        self,
        *,
        latents,
        image_latents,
        condition,
        condition_mask,
        guidance,
        img_shapes,
        timesteps,
        num_warmup_steps,
        progress_bar,
    ):
        state = get_runtime_state()
        warmup = state.runtime_config.warmup_steps
        run_async = self._pipefusion_async_enabled(
            num_timesteps=len(timesteps),
            pipeline_warmup_steps=warmup,
            callback_on_step_end=None,
        )
        generated_tokens = latents.shape[1]
        if run_async:
            latents = self._sync_qwen_pipeline(
                latents=latents,
                image_latents=image_latents,
                condition=condition,
                condition_mask=condition_mask,
                guidance=guidance,
                img_shapes=img_shapes,
                generated_tokens=generated_tokens,
                timesteps=timesteps[:warmup],
                num_warmup_steps=num_warmup_steps,
                progress_bar=progress_bar,
                sync_only=True,
            )
            latents = self._pipefusion_handoff_sync_latents(
                latents,
                name="pipefusion_sync_latents",
            )
            layout = (
                PipeFusionPatchLayout.with_appended_reference_tokens(
                    generated_tokens=generated_tokens,
                    reference_tokens=image_latents.shape[1],
                    num_patches=state.num_pipeline_patch,
                    name="qwen-generated+reference",
                )
                if image_latents is not None
                else PipeFusionPatchLayout.from_runtime_state(
                    state,
                    split_dim=1,
                    split_sizes=state.pp_patches_token_num,
                    name="qwen-image",
                )
            )
            with layout.install(state):
                return self._async_qwen_pipeline(
                    latents=latents,
                    image_latents=image_latents,
                    condition=condition,
                    condition_mask=condition_mask,
                    guidance=guidance,
                    img_shapes=img_shapes,
                    timesteps=timesteps[warmup:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    layout=layout,
                    sync_handoff_complete=True,
                )
        return self._sync_qwen_pipeline(
            latents=latents,
            image_latents=image_latents,
            condition=condition,
            condition_mask=condition_mask,
            guidance=guidance,
            img_shapes=img_shapes,
            generated_tokens=generated_tokens,
            timesteps=timesteps,
            num_warmup_steps=num_warmup_steps,
            progress_bar=progress_bar,
            sync_only=True,
        )

    def _decode_qwen_latents(self, latents, *, height, width, output_type, return_dict):
        image = None

        def prepare_latents(value):
            value = self._unpack_latents(value, height, width, self.vae_scale_factor).to(self.vae.dtype)
            mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(value)
            std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(value)
            return value / std + mean

        if output_type != "latent":
            state = get_runtime_state()
            if state.runtime_config.use_parallel_vae and state.parallel_config.vae_parallel_size > 0:
                latents = self.gather_latents_for_vae(latents)
                if latents is not None:
                    latents = prepare_latents(latents)
                self.send_to_vae_decode(latents)
            elif state.runtime_config.use_parallel_vae:
                latents = self.gather_broadcast_latents(latents)
                image = self.vae.decode(prepare_latents(latents), return_dict=False)[0][:, :, 0]
            elif is_dp_last_group():
                self._release_transformer_kv_cache()
                image = self.vae.decode(prepare_latents(latents), return_dict=False)[0][:, :, 0]
        if not is_dp_last_group():
            return None
        if output_type == "latent":
            image = latents
        elif image is not None:
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)


class xFuserQwenImagePipeFusionPipeline(QwenPipeFusionMixin, xFuserPipelineBaseWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        engine_config: EngineConfig,
        cache_args: Optional[dict] = None,
        **kwargs,
    ):
        pipeline = QwenImagePipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(pipeline, engine_config, cache_args or {})

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        sigmas=None,
        true_cfg_scale=1.0,
        guidance_scale=None,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        output_type="pil",
        return_dict=True,
        attention_kwargs=None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs=("latents",),
        max_sequence_length=512,
        **kwargs: Any,
    ):
        self._validate_qwen_callback(callback_on_step_end)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._attention_kwargs = attention_kwargs or {}
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if prompt is not None else prompt_embeds.shape[0]
        batch_size *= num_images_per_prompt
        self._set_qwen_runtime(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
        )
        device = self._execution_device
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        self._qwen_do_true_cfg = true_cfg_scale > 1 and (
            negative_prompt is not None or negative_prompt_embeds is not None
        )
        self._qwen_branch_kv = (
            {"uncond": {}, "cond": {}}
            if self._qwen_do_true_cfg and get_classifier_free_guidance_world_size() == 1
            else None
        )
        if self._qwen_do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        self._qwen_true_cfg_scale = true_cfg_scale
        condition, condition_mask = self._select_cfg_condition(
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
        )
        channels = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size,
            channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            latents.shape[1],
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
        guidance = (
            torch.full((batch_size,), guidance_scale, device=device)
            if self.transformer.config.guidance_embeds
            else None
        )
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            latents = self._denoise_qwen(
                latents=latents,
                image_latents=None,
                condition=condition,
                condition_mask=condition_mask,
                guidance=guidance,
                img_shapes=img_shapes,
                timesteps=timesteps,
                num_warmup_steps=num_warmup_steps,
                progress_bar=progress_bar,
            )
        return self._decode_qwen_latents(
            latents,
            height=height,
            width=width,
            output_type=output_type,
            return_dict=return_dict,
        )
