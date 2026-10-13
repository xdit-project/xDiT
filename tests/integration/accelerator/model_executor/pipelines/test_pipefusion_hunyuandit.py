"""HunyuanDiT under PipeFusion must match diffusers across images per prompt and CFG.

HunyuanDiT's second-half PipeFusion stages receive skip connections in a buffer
sized for one image per prompt with CFG on, and its last stage returned the
latents instead of the noise prediction with CFG off. Two pipeline stages,
synchronous for every step so that PipeFusion is exact, run two images per prompt
with CFG and then one image without, and compare each call with diffusers.
"""

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import init_distributed_environment, is_pipeline_last_stage
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

STEPS = 2


def _tiny_hunyuandit():
    from diffusers import DDPMScheduler, HunyuanDiT2DModel, HunyuanDiTPipeline

    torch.manual_seed(0)
    transformer = HunyuanDiT2DModel(
        sample_size=16,
        num_layers=4,
        patch_size=2,
        attention_head_dim=8,
        num_attention_heads=3,
        in_channels=4,
        cross_attention_dim=32,
        cross_attention_dim_t5=32,
        pooled_projection_dim=16,
        hidden_size=24,
        activation_fn="gelu-approximate",
    )
    return HunyuanDiTPipeline(
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
        scheduler=DDPMScheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def _hunyuandit_kwargs(guidance_scale, num_images_per_prompt):
    generator = torch.Generator().manual_seed(1)
    # The text lengths are the ones HunyuanDiT's text poolers are built for. The
    # pipelines repeat precomputed embeddings per image but not their masks.
    masks = num_images_per_prompt
    return {
        "prompt_embeds": torch.randn(1, 77, 32, generator=generator),
        "prompt_attention_mask": torch.ones(masks, 77, dtype=torch.long),
        "negative_prompt_embeds": torch.randn(1, 77, 32, generator=generator),
        "negative_prompt_attention_mask": torch.ones(masks, 77, dtype=torch.long),
        "prompt_embeds_2": torch.randn(1, 256, 32, generator=generator),
        "prompt_attention_mask_2": torch.ones(masks, 256, dtype=torch.long),
        "negative_prompt_embeds_2": torch.randn(1, 256, 32, generator=generator),
        "negative_prompt_attention_mask_2": torch.ones(masks, 256, dtype=torch.long),
        "latents": torch.randn(num_images_per_prompt, 4, 16, 16, generator=generator),
        # DDPM adds fresh noise every step; draw it from the same seed on both sides.
        "generator": torch.Generator().manual_seed(0),
        "height": 128,
        "width": 128,
        "num_inference_steps": STEPS,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
        "use_resolution_binning": False,
        "output_type": "latent",
    }


# name -> (builder, call kwargs, wrapper module, wrapper class, (guidance_scale, images per prompt) per call)
CASES = {
    "hunyuandit_images_per_prompt_and_cfg": (
        _tiny_hunyuandit,
        _hunyuandit_kwargs,
        "pipeline_hunyuandit",
        "xFuserHunyuanDiTPipeline",
        [(5.0, 2), (1.0, 1)],
    ),
}


def _worker(rank, world_size, init_method, name):
    import importlib

    build, call_kwargs, module_name, wrapper_name, calls = CASES[name]
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    # Warm up for every step: synchronous PipeFusion is exact, so it must match diffusers.
    args = xFuserArgs(
        model="tiny",
        pipefusion_parallel_degree=world_size,
        num_pipeline_patch=2,
        warmup_steps=STEPS,
        attention_backend="sdpa",
    )
    engine_config, _ = args.create_config()
    engine_config.runtime_config.dtype = torch.float32

    def on_device(kwargs):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in kwargs.items()}

    reference = build().to(device)
    wrapper_cls = getattr(importlib.import_module(f"xfuser.model_executor.pipelines.{module_name}"), wrapper_name)
    wrapper = wrapper_cls(build().to(device), engine_config)
    for guidance_scale, num_images_per_prompt in calls:
        with torch.no_grad():
            expected = reference(**on_device(call_kwargs(guidance_scale, num_images_per_prompt)))[0]
        output = wrapper(**on_device(call_kwargs(guidance_scale, num_images_per_prompt)))
        if is_pipeline_last_stage():
            torch.testing.assert_close(output[0], expected, rtol=1e-4, atol=1e-4)
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("name", sorted(CASES))
def test_pipefusion_hunyuandit_matches_diffusers(accelerator_ranks, name):
    accelerator_ranks(_worker, world_size=2, timeout=300, init_filename=f"pipefusion-{name}", args=(name,))
