import inspect
import os
from typing import Any

import numpy as np
import torch
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:  # diffusers releases before Qwen-Image-Edit-2509
    QwenImageEditPlusPipeline = None

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from xfuser.config import EngineConfig

from .base_pipeline import xFuserPipelineBaseWrapper
from .pipeline_qwen_image import QwenPipeFusionMixin


def _image_size(image):
    source = image[0] if isinstance(image, list) else image
    if isinstance(source, np.ndarray):
        if source.ndim >= 3:
            return source.shape[-2], source.shape[-3]
        return source.shape[-1], source.shape[-2]
    if isinstance(source, torch.Tensor):
        return source.shape[-1], source.shape[-2]
    return source.size


class _TrueCfgParallelMixin:
    """Run Qwen-Image-Edit's true-CFG branches on the two CFG parallel ranks."""

    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any):
        if get_classifier_free_guidance_world_size() != 2:
            return super().__call__(*args, **kwargs)

        parent_call = super().__call__
        call_signature = inspect.signature(parent_call)
        call_args = call_signature.bind_partial(*args, **kwargs)
        true_cfg_scale = call_args.arguments.get("true_cfg_scale", call_signature.parameters["true_cfg_scale"].default)
        has_negative_prompt = (
            call_args.arguments.get("negative_prompt") is not None
            or call_args.arguments.get("negative_prompt_embeds") is not None
        )
        if true_cfg_scale <= 1 or not has_negative_prompt:
            return parent_call(*args, **kwargs)

        if get_classifier_free_guidance_rank() == 0:
            call_args.arguments["prompt"] = call_args.arguments.get("negative_prompt")
            call_args.arguments["prompt_embeds"] = call_args.arguments.get("negative_prompt_embeds")
            call_args.arguments["prompt_embeds_mask"] = call_args.arguments.get("negative_prompt_embeds_mask")

        # The parent pipeline now executes one local branch. The hook below
        # combines both predictions before its scheduler step.
        call_args.arguments["negative_prompt"] = None
        call_args.arguments["negative_prompt_embeds"] = None
        call_args.arguments["negative_prompt_embeds_mask"] = None
        call_args.arguments["true_cfg_scale"] = 1.0

        def combine_cfg_predictions(_module, _inputs, output):
            prediction = output[0]
            prediction_uncond, prediction_cond = get_cfg_group().all_gather(prediction, separate_tensors=True)
            prediction_cfg = prediction_uncond + true_cfg_scale * (prediction_cond - prediction_uncond)

            # Preserve Qwen-Image-Edit's true-CFG normalization.
            cond_norm = torch.norm(prediction_cond, dim=-1, keepdim=True)
            cfg_norm = torch.norm(prediction_cfg, dim=-1, keepdim=True)
            prediction_cfg = prediction_cfg * (cond_norm / cfg_norm)
            return (prediction_cfg, *output[1:])

        hook = self.transformer.register_forward_hook(combine_cfg_predictions)
        try:
            return parent_call(*call_args.args, **call_args.kwargs)
        finally:
            hook.remove()


class xFuserQwenImageEditPipeline(_TrueCfgParallelMixin, QwenImageEditPipeline):
    """Qwen-Image-Edit pipeline with distributed true-CFG branches."""


class xFuserQwenImageEditPipeFusionPipeline(QwenPipeFusionMixin, xFuserPipelineBaseWrapper):
    """PipeFusion pipeline for the original single-reference Qwen Image Edit."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        engine_config: EngineConfig,
        cache_args: dict | None = None,
        **kwargs,
    ):
        pipeline = QwenImageEditPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(pipeline, engine_config, cache_args or {})

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        negative_prompt=None,
        true_cfg_scale=4.0,
        height=None,
        width=None,
        num_inference_steps=50,
        sigmas=None,
        guidance_scale=None,
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
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        max_sequence_length=512,
        **kwargs,
    ):
        self._validate_qwen_callback(callback_on_step_end)
        image_size = _image_size(image)
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024,
            image_size[0] / image_size[1],
        )
        height = height or calculated_height
        width = width or calculated_width
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
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
        prompt_image = image
        if not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width).unsqueeze(2)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
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
                image=prompt_image,
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
        latents, image_latents = self.prepare_latents(
            image,
            batch_size,
            channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (
                    1,
                    calculated_height // self.vae_scale_factor // 2,
                    calculated_width // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size
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
                image_latents=image_latents,
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


if QwenImageEditPlusPipeline is not None:

    class xFuserQwenImageEditPlusPipeline(_TrueCfgParallelMixin, QwenImageEditPlusPipeline):
        """Qwen-Image-Edit-2509/2511 (Edit Plus) pipeline with distributed true-CFG branches."""

else:
    xFuserQwenImageEditPlusPipeline = None
