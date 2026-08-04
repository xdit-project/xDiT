from typing import Any, Callable

import torch

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)


def _broadcast_object_in_group(value, coordinator):
    payload = [value]
    torch.distributed.broadcast_object_list(
        payload,
        src=coordinator.first_rank,
        group=coordinator.cpu_group,
    )
    return payload[0]


def _clone_generator(
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Generator | list[torch.Generator] | None:
    if generator is None:
        return None
    if isinstance(generator, list):
        return [_clone_generator(item) for item in generator]

    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


def _make_xfuser_ideogram4_pipeline_class():
    from diffusers.models.transformers.transformer_ideogram4 import (
        LLM_TOKEN_INDICATOR,
    )
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
        Ideogram4Pipeline,
        Ideogram4PipelineOutput,
        _expand_tensor_to_effective_batch,
        _logit_normal_sigmas,
        _resolution_aware_mu,
    )

    class xFuserIdeogram4Pipeline(Ideogram4Pipeline):
        @staticmethod
        def _layout_target(transformer):
            return getattr(transformer, "_orig_mod", transformer)

        def encode_prompt(
            self,
            prompt: str | list[str],
            grid_h: int,
            grid_w: int,
            max_sequence_length: int,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = super().encode_prompt(
                prompt=prompt,
                grid_h=grid_h,
                grid_w=grid_w,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            _, _, _, indicator = outputs

            text_counts = (indicator == LLM_TOKEN_INDICATOR).sum(dim=1)
            if not torch.equal(text_counts, text_counts[:1].expand_as(text_counts)):
                raise ValueError(
                    "Ideogram 4 xDiT requires prompts in the same batch to have equal token lengths."
                )

            num_text_tokens = int(text_counts[0].item())
            num_image_tokens = grid_h * grid_w
            num_pad_tokens = max_sequence_length - num_text_tokens

            transformer = self._layout_target(self.transformer)
            unconditional_transformer = self._layout_target(
                self.unconditional_transformer
            )
            transformer._set_sequence_layout(
                num_pad_tokens,
                num_text_tokens,
                num_image_tokens,
            )
            unconditional_transformer._set_sequence_layout(
                0,
                0,
                num_image_tokens,
            )
            return outputs

        def _upsample_prompt_distributed(
            self,
            prompt: str | list[str],
            height: int,
            width: int,
            temperature: float,
            max_new_tokens: int,
            generator: torch.Generator | list[torch.Generator] | None,
        ) -> list[str]:
            try:
                cfg_rank = get_classifier_free_guidance_rank()
                cfg_size = get_classifier_free_guidance_world_size()
            except AssertionError:
                cfg_rank = 0
                cfg_size = 1
            try:
                sp_rank = get_sequence_parallel_rank()
                sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                sp_rank = 0
                sp_size = 1

            captions = None
            if cfg_rank == 0 and sp_rank == 0:
                captions = self.upsample_prompt(
                    prompt,
                    height=height,
                    width=width,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    generator=_clone_generator(generator),
                    device=self._execution_device,
                )

            if cfg_size > 1:
                captions = _broadcast_object_in_group(captions, get_cfg_group())
            if sp_size > 1:
                captions = _broadcast_object_in_group(captions, get_sp_group())
            return captions

        @torch.no_grad()
        def __call__(
            self,
            prompt: str | list[str] | None = None,
            height: int = 2048,
            width: int = 2048,
            num_inference_steps: int = 48,
            guidance_scale: float | None = None,
            guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45
            + (3.0,) * 3,
            mu: float = 0.0,
            std: float = 1.5,
            prompt_upsampling: bool = False,
            prompt_upsampling_temperature: float = 1.0,
            max_sequence_length: int = 2048,
            num_images_per_prompt: int = 1,
            generator: torch.Generator | list[torch.Generator] | None = None,
            latents: torch.Tensor | None = None,
            output_type: str = "pil",
            return_dict: bool = True,
            attention_kwargs: dict[str, Any] | None = None,
            callback_on_step_end: (
                Callable[
                    ["Ideogram4Pipeline", int, int, dict[str, Any]],
                    dict[str, Any],
                ]
                | None
            ) = None,
            callback_on_step_end_tensor_inputs: list[str] | tuple[str, ...] = (
                "latents",
            ),
        ) -> Ideogram4PipelineOutput | tuple[Any]:
            if prompt_upsampling:
                prompt = self._upsample_prompt_distributed(
                    prompt=prompt,
                    height=height,
                    width=width,
                    temperature=prompt_upsampling_temperature,
                    max_new_tokens=max_sequence_length,
                    generator=generator,
                )
                prompt_upsampling = False

            try:
                cfg_rank = get_classifier_free_guidance_rank()
                cfg_size = get_classifier_free_guidance_world_size()
            except AssertionError:
                cfg_rank = 0
                cfg_size = 1

            if cfg_size == 2:
                return self._call_with_cfg_parallel(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_schedule=guidance_schedule,
                    mu=mu,
                    std=std,
                    max_sequence_length=max_sequence_length,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    output_type=output_type,
                    return_dict=return_dict,
                    attention_kwargs=attention_kwargs,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    cfg_rank=cfg_rank,
                )
            if cfg_size != 1:
                raise ValueError(
                    f"Ideogram 4 CFG parallelism requires degree 1 or 2, got {cfg_size}."
                )

            return super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                mu=mu,
                std=std,
                prompt_upsampling=prompt_upsampling,
                prompt_upsampling_temperature=prompt_upsampling_temperature,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )

        def _call_with_cfg_parallel(
            self,
            prompt: str | list[str] | None,
            height: int,
            width: int,
            num_inference_steps: int,
            guidance_scale: float | None,
            guidance_schedule: list[float] | torch.Tensor | None,
            mu: float,
            std: float,
            max_sequence_length: int,
            num_images_per_prompt: int,
            generator: torch.Generator | list[torch.Generator] | None,
            latents: torch.Tensor | None,
            output_type: str,
            return_dict: bool,
            attention_kwargs: dict[str, Any] | None,
            callback_on_step_end: Callable | None,
            callback_on_step_end_tensor_inputs: list[str] | tuple[str, ...],
            cfg_rank: int,
        ) -> Ideogram4PipelineOutput | tuple[Any]:
            self.check_inputs(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )

            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._interrupt = False

            grid_h = height // (self.vae_scale_factor * self.patch_size)
            grid_w = width // (self.vae_scale_factor * self.patch_size)
            num_image_tokens = grid_h * grid_w

            llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
                prompt=prompt,
                grid_h=grid_h,
                grid_w=grid_w,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            llm_features = _expand_tensor_to_effective_batch(
                llm_features,
                batch_size,
                num_images_per_prompt,
            )
            position_ids = _expand_tensor_to_effective_batch(
                position_ids,
                batch_size,
                num_images_per_prompt,
            )
            segment_ids = _expand_tensor_to_effective_batch(
                segment_ids,
                batch_size,
                num_images_per_prompt,
            )
            indicator = _expand_tensor_to_effective_batch(
                indicator,
                batch_size,
                num_images_per_prompt,
            )

            effective_batch_size = batch_size * num_images_per_prompt
            neg_llm_features = torch.zeros(
                effective_batch_size,
                num_image_tokens,
                llm_features.shape[-1],
                dtype=llm_features.dtype,
                device=device,
            )
            neg_position_ids = position_ids[:, max_sequence_length:]
            neg_segment_ids = segment_ids[:, max_sequence_length:]
            neg_indicator = indicator[:, max_sequence_length:]

            schedule_mu = _resolution_aware_mu(
                height=height,
                width=width,
                base_mu=mu,
            )
            sigmas = _logit_normal_sigmas(
                num_inference_steps,
                schedule_mu,
                std=std,
                device=device,
            )
            self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            if guidance_scale is not None:
                guidance_schedule = [float(guidance_scale)] * num_inference_steps
            guidance_weights = torch.as_tensor(
                guidance_schedule,
                dtype=torch.float32,
                device=device,
            )

            latent_dim = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size=effective_batch_size,
                num_image_tokens=num_image_tokens,
                latent_dim=latent_dim,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
            )
            text_z_padding = torch.zeros(
                effective_batch_size,
                max_sequence_length,
                latent_dim,
                dtype=torch.float32,
                device=device,
            )

            llm_features = llm_features.to(self.transformer.dtype)
            neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)

            num_train_timesteps = self.scheduler.config.num_train_timesteps
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for step_index, timestep in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    model_timestep = 1.0 - (timestep.float() / num_train_timesteps)
                    model_timestep = model_timestep.expand(effective_batch_size)

                    if cfg_rank == 1:
                        conditional_input = torch.cat(
                            [text_z_padding, latents],
                            dim=1,
                        ).to(self.transformer.dtype)
                        velocity = self.transformer(
                            hidden_states=conditional_input,
                            timestep=model_timestep.to(self.transformer.dtype),
                            encoder_hidden_states=llm_features,
                            position_ids=position_ids,
                            segment_ids=segment_ids,
                            indicator=indicator,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0][:, max_sequence_length:]
                    else:
                        velocity = self.unconditional_transformer(
                            hidden_states=latents.to(
                                self.unconditional_transformer.dtype
                            ),
                            timestep=model_timestep.to(
                                self.unconditional_transformer.dtype
                            ),
                            encoder_hidden_states=neg_llm_features,
                            position_ids=neg_position_ids,
                            segment_ids=neg_segment_ids,
                            indicator=neg_indicator,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                    (
                        unconditional_velocity,
                        conditional_velocity,
                    ) = get_cfg_group().all_gather(
                        velocity.to(torch.float32),
                        separate_tensors=True,
                    )
                    self._guidance_scale = guidance_schedule[step_index]
                    guidance_weight = guidance_weights[step_index]
                    velocity = (
                        guidance_weight * conditional_velocity
                        + (1.0 - guidance_weight) * unconditional_velocity
                    )
                    latents = self.scheduler.step(
                        -velocity,
                        timestep,
                        latents,
                        return_dict=False,
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {
                            name: locals()[name]
                            for name in callback_on_step_end_tensor_inputs
                        }
                        callback_outputs = callback_on_step_end(
                            self,
                            step_index,
                            timestep,
                            callback_kwargs,
                        )
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

            if output_type == "latent":
                image = latents
            else:
                latents = latents * torch.sqrt(
                    self.vae.bn.running_var + self.vae.config.batch_norm_eps
                ).view(1, 1, -1).to(device=latents.device, dtype=latents.dtype)
                latents = latents + self.vae.bn.running_mean.view(1, 1, -1).to(
                    device=latents.device,
                    dtype=latents.dtype,
                )

                patch_size = self.patch_size
                ae_channels = latents.shape[-1] // (patch_size * patch_size)
                latents = latents.view(
                    effective_batch_size,
                    grid_h,
                    grid_w,
                    patch_size,
                    patch_size,
                    ae_channels,
                )
                latents = latents.permute(0, 5, 1, 3, 2, 4).contiguous()
                latents = latents.view(
                    effective_batch_size,
                    ae_channels,
                    grid_h * patch_size,
                    grid_w * patch_size,
                )
                decoded = self.vae.decode(
                    latents.to(self.vae.dtype),
                    return_dict=False,
                )[0]
                image = self.image_processor.postprocess(
                    decoded.float(),
                    output_type=output_type,
                )

            self.maybe_free_model_hooks()
            if not return_dict:
                return (image,)
            return Ideogram4PipelineOutput(images=image)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            pipeline = super().from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
            pipeline.__class__ = cls
            return pipeline

    return xFuserIdeogram4Pipeline


_pipeline_cls = None


def get_ideogram4_pipeline_class():
    global _pipeline_cls
    if _pipeline_cls is None:
        _pipeline_cls = _make_xfuser_ideogram4_pipeline_class()
    return _pipeline_cls
