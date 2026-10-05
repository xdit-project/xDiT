import inspect
from typing import Any

import torch
from diffusers import QwenImageEditPipeline

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


class xFuserQwenImageEditPipeline(QwenImageEditPipeline):
    """Qwen-Image-Edit pipeline with distributed true-CFG branches."""

    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any):
        if get_classifier_free_guidance_world_size() != 2:
            return super().__call__(*args, **kwargs)

        parent_call = super().__call__
        call_signature = inspect.signature(parent_call)
        call_args = call_signature.bind_partial(*args, **kwargs)
        true_cfg_scale = call_args.arguments.get(
            "true_cfg_scale", call_signature.parameters["true_cfg_scale"].default
        )
        has_negative_prompt = (
            call_args.arguments.get("negative_prompt") is not None
            or call_args.arguments.get("negative_prompt_embeds") is not None
        )
        if true_cfg_scale <= 1 or not has_negative_prompt:
            return parent_call(*args, **kwargs)

        if get_classifier_free_guidance_rank() == 0:
            call_args.arguments["prompt"] = call_args.arguments.get("negative_prompt")
            call_args.arguments["prompt_embeds"] = call_args.arguments.get(
                "negative_prompt_embeds"
            )
            call_args.arguments["prompt_embeds_mask"] = call_args.arguments.get(
                "negative_prompt_embeds_mask"
            )

        # The parent pipeline now executes one local branch. The hook below
        # combines both predictions before its scheduler step.
        call_args.arguments["negative_prompt"] = None
        call_args.arguments["negative_prompt_embeds"] = None
        call_args.arguments["negative_prompt_embeds_mask"] = None
        call_args.arguments["true_cfg_scale"] = 1.0

        def combine_cfg_predictions(_module, _inputs, output):
            prediction = output[0]
            prediction_uncond, prediction_cond = get_cfg_group().all_gather(
                prediction, separate_tensors=True
            )
            prediction_cfg = prediction_uncond + true_cfg_scale * (
                prediction_cond - prediction_uncond
            )

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
