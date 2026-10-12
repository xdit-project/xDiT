"""CogVideoX and ConsisID must run with classifier-free guidance switched off.

The xDiT ``__call__`` of these pipelines runs whenever any parallelism is on;
on a single rank they defer to the stock diffusers pipeline. The tests call it
directly, with the parallel state of one rank, and compare against diffusers.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from diffusers import (  # noqa: E402
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)

from xfuser.model_executor.pipelines import base_pipeline, pipeline_cogvideox  # noqa: E402
from xfuser.model_executor.pipelines.pipeline_cogvideox import xFuserCogVideoXPipeline  # noqa: E402
from xfuser.model_executor.schedulers import (  # noqa: E402
    base_scheduler,
    xFuserCogVideoXDDIMSchedulerWrapper,
    xFuserCogVideoXDPMSchedulerWrapper,
)

ConsisIDPipeline = getattr(diffusers, "ConsisIDPipeline", None)
ConsisIDTransformer3DModel = getattr(diffusers, "ConsisIDTransformer3DModel", None)

TEXT_LEN = 16
TEXT_DIM = 32


def _tiny_vae():
    torch.manual_seed(0)
    return AutoencoderKLCogVideoX(
        in_channels=3,
        out_channels=3,
        down_block_types=("CogVideoXDownBlock3D",) * 4,
        up_block_types=("CogVideoXUpBlock3D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=2,
        temporal_compression_ratio=4,
    ).eval()


_SINGLE_RANK = {
    "get_classifier_free_guidance_world_size": 1,
    "get_pipeline_parallel_world_size": 1,
    "get_sequence_parallel_world_size": 1,
    "get_sequence_parallel_rank": 0,
    "is_dp_last_group": True,
}


def _single_rank(pipeline_module):
    """Run ``pipeline_module`` as one rank with no parallelism (2x2 latents, 1 token per frame)."""
    state = SimpleNamespace(
        split_text_embed_in_sp=False,
        pp_patches_start_end_idx_global=[(0, 2)],
        pp_patches_token_start_end_idx_global=[(0, 1)],
        set_video_input_parameters=lambda **kwargs: None,
        set_patched_mode=lambda patch_mode: None,
    )
    stack = ExitStack()
    for module in (base_pipeline, base_scheduler, pipeline_module):
        for name, value in {**_SINGLE_RANK, "get_runtime_state": state}.items():
            if hasattr(module, name):
                stack.enter_context(patch.object(module, name, return_value=value))
    return stack


def _wrap(wrapper_cls, pipe, scheduler_wrapper_cls):
    """Build the xFuser wrapper around ``pipe`` without a distributed runtime."""
    wrapper = object.__new__(wrapper_cls)
    wrapper.module = pipe
    wrapper.module_type = type(pipe)
    wrapper.scheduler = scheduler_wrapper_cls(pipe.scheduler)
    return wrapper


def _call_wrapper(wrapper_cls, wrapper, **kwargs):
    # Skip the data-parallel and naive-forward decorators, which need a runtime
    # and would hand a single rank to the stock pipeline.
    call = inspect.unwrap(wrapper_cls.__call__)
    return call(wrapper, **kwargs)


def _embeds(batch=1):
    generator = torch.Generator().manual_seed(1)
    prompt = torch.randn(batch, TEXT_LEN, TEXT_DIM, generator=generator)
    negative = torch.randn(batch, TEXT_LEN, TEXT_DIM, generator=generator)
    return prompt, negative


def _cogvideox_pipe():
    torch.manual_seed(0)
    transformer = CogVideoXTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=TEXT_DIM,
        num_layers=1,
        sample_width=2,
        sample_height=2,
        sample_frames=9,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=TEXT_LEN,
    ).eval()
    return CogVideoXPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=_tiny_vae(),
        transformer=transformer,
        scheduler=CogVideoXDDIMScheduler(),
    )


def _run_cogvideox(guidance_scale):
    pipe = _cogvideox_pipe()
    prompt_embeds, negative_prompt_embeds = _embeds()
    latents = torch.randn(1, 3, 4, 2, 2, generator=torch.Generator().manual_seed(2))
    kwargs = dict(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds if guidance_scale > 1 else None,
        latents=latents,
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        max_sequence_length=TEXT_LEN,
        output_type="latent",
        return_dict=False,
    )
    expected = pipe(**kwargs)[0]

    wrapper = _wrap(xFuserCogVideoXPipeline, pipe, xFuserCogVideoXDDIMSchedulerWrapper)
    with _single_rank(pipeline_cogvideox):
        actual = _call_wrapper(xFuserCogVideoXPipeline, wrapper, **kwargs)[0]
    return actual, expected


@pytest.mark.parametrize("guidance_scale", [1.0, 6.0])
def test_cogvideox_matches_diffusers_with_and_without_guidance(guidance_scale):
    actual, expected = _run_cogvideox(guidance_scale)

    torch.testing.assert_close(actual, expected)


def _consisid_pipe():
    torch.manual_seed(0)
    transformer = ConsisIDTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=8,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=TEXT_DIM,
        num_layers=1,
        sample_width=2,
        sample_height=2,
        sample_frames=9,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=TEXT_LEN,
        use_rotary_positional_embeddings=True,
        use_learned_positional_embeddings=True,
        # Keep the face branch, but tiny.
        is_train_face=True,
        cross_attn_interval=1,
        LFE_id_dim=2,
        LFE_vit_dim=2,
        LFE_depth=1,
        LFE_output_dim=21,
        LFE_num_scale=1,
    ).eval()
    return ConsisIDPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=_tiny_vae(),
        transformer=transformer,
        scheduler=CogVideoXDPMScheduler(),
    )


@pytest.mark.skipif(ConsisIDPipeline is None, reason="diffusers has no ConsisID")
@pytest.mark.parametrize("guidance_scale", [1.0, 6.0])
def test_consisid_matches_diffusers_with_and_without_guidance(guidance_scale):
    pytest.importorskip("cv2", reason="ConsisIDPipeline requires OpenCV")
    from xfuser.model_executor.pipelines import pipeline_consisid

    pipe = _consisid_pipe()
    prompt_embeds, negative_prompt_embeds = _embeds()
    image = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    latents = torch.randn(1, 3, 4, 2, 2, generator=torch.Generator().manual_seed(2))

    def kwargs():
        return dict(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=(negative_prompt_embeds if guidance_scale > 1 else None),
            latents=latents,
            height=16,
            width=16,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=guidance_scale,
            max_sequence_length=TEXT_LEN,
            id_vit_hidden=[torch.ones(1, 2, 2)],
            id_cond=torch.ones(1, 2),
            # The image encode samples from the VAE posterior.
            generator=torch.Generator().manual_seed(4),
            output_type="latent",
            return_dict=False,
        )

    expected = pipe(**kwargs())[0]

    wrapper_cls = pipeline_consisid.xFuserConsisIDPipeline
    wrapper = _wrap(wrapper_cls, pipe, xFuserCogVideoXDPMSchedulerWrapper)
    with _single_rank(pipeline_consisid):
        actual = _call_wrapper(wrapper_cls, wrapper, **kwargs())[0]

    torch.testing.assert_close(actual, expected)
