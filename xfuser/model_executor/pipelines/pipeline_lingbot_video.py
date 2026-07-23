from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def get_lingbot_video_pipeline_class():
    from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
    return LingBotVideoPipeline


class xFuserLingBotVideoPipeline:
    """Wrapper that adds xDiT's CFG parallelism to LingBotVideoPipeline.

    Uses xDiT's distributed infrastructure (get_cfg_group, etc.) instead of
    the built-in cfg_parallel_group parameter.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        LingBotVideoPipeline = get_lingbot_video_pipeline_class()
        kwargs.setdefault("trust_remote_code", True)
        pipe = LingBotVideoPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        pipe.__class__ = type(
            "xFuserLingBotVideoPipeline",
            (xFuserLingBotVideoPipeline, LingBotVideoPipeline),
            {},
        )
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        negative_prompt: Union[str, List[str]] = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        **kwargs,
    ):
        cfg_rank = get_classifier_free_guidance_rank()
        do_cfg_parallel = guidance_scale > 1.0 and get_classifier_free_guidance_world_size() == 2

        if do_cfg_parallel and "image" not in kwargs:
            # T2V CFG parallel: custom path with xDiT's all_gather
            return self._call_with_xdit_cfg_parallel(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_mask=negative_prompt_mask,
                output_type=output_type,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            # Delegate to the stock pipeline (T2V or TI2V).
            stock_bases = [b for b in type(self).__mro__
                          if b not in (xFuserLingBotVideoPipeline, type(self))
                          and hasattr(b, "__call__")
                          and "__call__" in b.__dict__]
            parent_cls = stock_bases[0] if stock_bases else get_lingbot_video_pipeline_class()
            return parent_cls.__call__(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_mask=negative_prompt_mask,
                output_type=output_type,
                return_dict=return_dict,
                **kwargs,
            )

    def _call_with_xdit_cfg_parallel(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        shift,
        generator,
        latents,
        prompt_embeds,
        prompt_mask,
        negative_prompt_embeds,
        negative_prompt_mask,
        output_type,
        return_dict,
        **kwargs,
    ):
        from lingbot_video.pipeline_lingbot_video import (
            LingBotVideoPipelineOutput,
            _transformer_timestep,
            _transformer_autocast,
            _module_dtype,
        )
        from lingbot_video.utils import compute_refiner_sigmas

        self.check_inputs(height, width, num_frames)
        device = self._execution_device
        cfg_rank = get_classifier_free_guidance_rank()

        # TI2V: preprocess image for VLM text encoder (both ranks need it)
        encode_kwargs = {}
        image = kwargs.get("image")
        if image is not None and hasattr(self, "_vlm_image"):
            pixel = self.preprocess_image(image, height, width)
            pixel = pixel.to(device=device, dtype=torch.float32)
            encode_kwargs["images"] = [self._vlm_image(pixel)]

        # Encode prompts per-rank: rank 0 = conditional, rank 1 = unconditional
        if cfg_rank == 0:
            if prompt_embeds is None:
                prompt_embeds, prompt_mask = self.encode_prompt(
                    prompt, device=device, **encode_kwargs)
            local_embeds = prompt_embeds.to(device)
            local_mask = prompt_mask.to(device)
        else:
            if negative_prompt_embeds is None:
                negative_prompt_embeds, negative_prompt_mask = self.encode_prompt(
                    negative_prompt, device=device, **encode_kwargs)
            local_embeds = negative_prompt_embeds.to(device)
            local_mask = negative_prompt_mask.to(device)

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)

        # TI2V: encode first frame and inject into latents
        if image is not None and hasattr(self, "encode_image_latent"):
            cond_latent = self.encode_image_latent(pixel, generator=generator)
            cond_latent = cond_latent.to(device=device, dtype=torch.float32)
            latents = self._apply_inpainting(latents, cond_latent)
        else:
            cond_latent = None

        t_thresh = kwargs.get("t_thresh", None)
        refiner_sigma_tail_steps = kwargs.get("refiner_sigma_tail_steps", None)
        sigmas = compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=refiner_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        else:
            self.scheduler.set_timesteps(int(sigmas.shape[0]), device=device, sigmas=sigmas, shift=1.0)

        transformer_dtype = _module_dtype(self.transformer)

        for i, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            local_model_input = local_embeds.to(transformer_dtype)

            with _transformer_autocast(device, transformer_dtype):
                noise_pred = self.transformer(
                    latents,
                    timestep_batch,
                    local_model_input,
                    encoder_attention_mask=local_mask,
                    return_dict=False,
                )[0].float()

            # All-gather across CFG ranks: rank 0 = cond, rank 1 = uncond
            noise_pred_cond, noise_pred_uncond = get_cfg_group().all_gather(
                noise_pred, separate_tensors=True
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, timestep, latents, return_dict=False, generator=generator
            )[0]
            if cond_latent is not None:
                latents = self._apply_inpainting(latents, cond_latent)

        if output_type == "latent":
            frames = latents
        elif output_type == "np":
            frames = self._decode_latents(latents)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return LingBotVideoPipelineOutput(frames=frames)
