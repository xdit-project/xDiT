import os
from collections.abc import Callable

import numpy as np
import torch
from diffusers import Lumina2Pipeline
from diffusers.pipelines.lumina2.pipeline_lumina2 import (
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_runtime_state,
)
from xfuser.envs import get_device_name

from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(Lumina2Pipeline)
class xFuserLumina2Pipeline(xFuserPipelineBaseWrapper):
    """Lumina2 pipeline with Ulysses and classifier-free-guidance parallelism."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        engine_config: EngineConfig,
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = Lumina2Pipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_run(
        self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1
    ):
        del sync_steps
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            output_type=input_config.output_type,
            generator=torch.Generator(device=get_device_name()).manual_seed(42),
            guidance_scale=input_config.guidance_scale,
        )

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
            attention_kwargs=self.attention_kwargs,
        )[0]

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    @xFuserPipelineBaseWrapper.check_model_parallel_state(
        cfg_parallel_available=True,
        sequence_parallel_available=True,
        pipefusion_parallel_available=False,
    )
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: str | list[str] | None = None,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        system_prompt: str | None = None,
        cfg_trunc_ratio: float = 1.0,
        cfg_normalization: bool = True,
        max_sequence_length: int = 256,
    ):
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            system_prompt=system_prompt,
        )

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

        cfg_world_size = get_classifier_free_guidance_world_size()
        if cfg_world_size not in (1, 2):
            raise RuntimeError(
                "Lumina2 CFG parallelism requires a CFG degree of 1 or 2, "
                f"but got {cfg_world_size}."
            )
        if cfg_world_size == 2:
            # All CFG ranks must denoise the same sample even if callers pass
            # generators whose states differ between processes.
            latents = get_cfg_group().broadcast(latents, src=0)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
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
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                truncate_cfg = (i + 1) / num_inference_steps > cfg_trunc_ratio
                current_timestep = 1 - t / self.scheduler.config.num_train_timesteps
                current_timestep = current_timestep.expand(latents.shape[0])

                if cfg_world_size == 2 and self.do_classifier_free_guidance:
                    cfg_rank = get_classifier_free_guidance_rank()
                    if not truncate_cfg:
                        if cfg_rank == 0:
                            local_embeds = negative_prompt_embeds
                            local_mask = negative_prompt_attention_mask
                        else:
                            local_embeds = prompt_embeds
                            local_mask = prompt_attention_mask
                        local_noise_pred = self._predict_noise(
                            latents, current_timestep, local_embeds, local_mask
                        )
                        noise_pred_uncond, noise_pred_cond = get_cfg_group().all_gather(
                            local_noise_pred, separate_tensors=True
                        )
                    else:
                        if cfg_rank == 1:
                            noise_pred_cond = self._predict_noise(
                                latents,
                                current_timestep,
                                prompt_embeds,
                                prompt_attention_mask,
                            )
                        else:
                            noise_pred_cond = torch.empty_like(latents)
                        noise_pred_cond = get_cfg_group().broadcast(
                            noise_pred_cond, src=1
                        )
                        noise_pred_uncond = None
                else:
                    noise_pred_cond = self._predict_noise(
                        latents,
                        current_timestep,
                        prompt_embeds,
                        prompt_attention_mask,
                    )
                    noise_pred_uncond = None
                    if self.do_classifier_free_guidance and not truncate_cfg:
                        noise_pred_uncond = self._predict_noise(
                            latents,
                            current_timestep,
                            negative_prompt_embeds,
                            negative_prompt_attention_mask,
                        )

                if self.do_classifier_free_guidance and not truncate_cfg:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    if cfg_normalization:
                        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_pred = noise_pred * (cond_norm / noise_norm)
                else:
                    noise_pred = noise_pred_cond

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    -noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        image = None
        if self.is_dp_last_group():
            if output_type != "latent":
                latents = (
                    latents / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                image = latents

        self.maybe_free_model_hooks()

        if not self.is_dp_last_group():
            return None
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
