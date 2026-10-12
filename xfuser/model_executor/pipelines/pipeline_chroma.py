import inspect
from typing import Any

import torch
from diffusers import ChromaPipeline

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


class xFuserChromaPipeline(ChromaPipeline):
    """Chroma pipeline that runs its two guidance branches on separate ranks.

    Chroma applies real classifier-free guidance as two transformer calls per
    step, one with the prompt and one with the negative prompt. With a CFG
    degree of 2, rank 0 of the CFG group runs the negative branch and rank 1
    the positive one; a forward hook gathers both predictions and combines
    them with the same formula as the diffusers pipeline before the scheduler
    step, so every rank steps identical latents.
    """

    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any):
        parent_call = super().__call__
        if get_classifier_free_guidance_world_size() == 1:
            return parent_call(*args, **kwargs)
        if get_classifier_free_guidance_world_size() != 2:
            raise ValueError(
                f"Chroma supports a CFG parallel degree of 1 or 2, got {get_classifier_free_guidance_world_size()}."
            )

        signature = inspect.signature(parent_call)
        call_args = signature.bind_partial(*args, **kwargs)
        arguments = call_args.arguments
        guidance_scale = arguments.get("guidance_scale", signature.parameters["guidance_scale"].default)
        if guidance_scale <= 1:
            # No negative branch to distribute; both ranks compute the same output.
            return parent_call(*args, **kwargs)
        if (
            arguments.get("ip_adapter_image") is not None
            or arguments.get("ip_adapter_image_embeds") is not None
            or arguments.get("negative_ip_adapter_image") is not None
            or arguments.get("negative_ip_adapter_image_embeds") is not None
        ):
            raise NotImplementedError("IP-Adapter inputs are not supported with CFG parallelism.")

        if get_classifier_free_guidance_rank() == 0:
            prompt = arguments.get("prompt")
            negative_prompt_embeds = arguments.get("negative_prompt_embeds")
            if negative_prompt_embeds is not None:
                arguments["prompt"] = None
                arguments["prompt_embeds"] = negative_prompt_embeds
                arguments["prompt_attention_mask"] = arguments.get("negative_prompt_attention_mask")
            else:
                # Mirrors ChromaPipeline.encode_prompt, which encodes "" when no
                # negative prompt is given.
                negative_prompt = arguments.get("negative_prompt") or ""
                if isinstance(negative_prompt, str):
                    if isinstance(prompt, list):
                        batch_size = len(prompt)
                    elif prompt is not None:
                        batch_size = 1
                    else:
                        batch_size = arguments["prompt_embeds"].shape[0]
                    negative_prompt = batch_size * [negative_prompt]
                arguments["prompt"] = negative_prompt
                arguments["prompt_embeds"] = None
                arguments["prompt_attention_mask"] = None

        # The parent now runs only this rank's branch; the hook combines them.
        arguments["negative_prompt"] = None
        arguments["negative_prompt_embeds"] = None
        arguments["negative_prompt_attention_mask"] = None
        arguments["guidance_scale"] = 1.0

        def combine_cfg_predictions(_module, _inputs, output):
            noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                output[0].contiguous(), separate_tensors=True
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return (noise_pred, *output[1:])

        hook = self.transformer.register_forward_hook(combine_cfg_predictions)
        try:
            return parent_call(*call_args.args, **call_args.kwargs)
        finally:
            hook.remove()
