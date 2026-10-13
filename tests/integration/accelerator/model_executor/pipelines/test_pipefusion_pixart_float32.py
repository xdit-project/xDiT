"""PipeFusion must run a float32 PixArt on the cuDNN attention backend.

With every step synchronous, PipeFusion computes exactly what diffusers does, so
two pipeline stages must reproduce the single-device output. In float32 they
returned NaN because the default NVIDIA attention backend, cuDNN, cannot serve
float32 inputs.
"""

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import init_distributed_environment, is_pipeline_last_stage
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

STEPS = 2


def _tiny_pixart():
    from diffusers import DDIMScheduler, PixArtAlphaPipeline, PixArtTransformer2DModel

    torch.manual_seed(0)
    transformer = PixArtTransformer2DModel(
        sample_size=8,
        num_layers=2,
        patch_size=2,
        attention_head_dim=8,
        num_attention_heads=3,
        caption_channels=32,
        in_channels=4,
        cross_attention_dim=24,
        out_channels=8,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )
    return PixArtAlphaPipeline(
        tokenizer=None, text_encoder=None, vae=None, transformer=transformer, scheduler=DDIMScheduler()
    )


def _call_kwargs(device):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(1, 6, 32, generator=generator).to(device),
        "prompt_attention_mask": torch.ones(1, 6, dtype=torch.long, device=device),
        "negative_prompt_embeds": torch.randn(1, 6, 32, generator=generator).to(device),
        "negative_prompt_attention_mask": torch.ones(1, 6, dtype=torch.long, device=device),
        "negative_prompt": None,
        "latents": torch.randn(1, 4, 8, 8, generator=generator).to(device),
        "height": 64,
        "width": 64,
        "num_inference_steps": STEPS,
        "guidance_scale": 4.5,
        "use_resolution_binning": False,
        "output_type": "latent",
    }


def _worker(rank, world_size, init_method):
    from xfuser.model_executor.pipelines.pipeline_pixart_alpha import xFuserPixArtAlphaPipeline

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    # Warm up for every step: synchronous PipeFusion is exact, so it must match diffusers.
    # Pinned to cuDNN, the default on NVIDIA without FlashAttention, so the test does not
    # depend on which backends are installed.
    args = xFuserArgs(
        model="tiny",
        pipefusion_parallel_degree=world_size,
        num_pipeline_patch=2,
        warmup_steps=STEPS,
        attention_backend="cudnn",
    )
    engine_config, _ = args.create_config()
    engine_config.runtime_config.dtype = torch.float32

    pipeline = _tiny_pixart().to(device)
    with torch.no_grad():
        expected = pipeline(**_call_kwargs(device))[0]
    output = xFuserPixArtAlphaPipeline(pipeline, engine_config)(**_call_kwargs(device))

    if is_pipeline_last_stage():
        torch.testing.assert_close(output[0], expected, rtol=1e-4, atol=1e-4)
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.nvidia
@pytest.mark.multi_gpu
def test_pipefusion_float32_pixart_matches_diffusers(accelerator_ranks):
    accelerator_ranks(_worker, world_size=2, timeout=240, init_filename="pipefusion-pixart-fp32")
