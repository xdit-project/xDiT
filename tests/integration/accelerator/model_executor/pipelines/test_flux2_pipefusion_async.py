"""A tiny FLUX.2 model must complete a genuinely asynchronous PipeFusion step."""

import pytest
import numpy as np
import torch
from PIL import Image

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import init_distributed_environment, is_pipeline_last_stage
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel


def _tiny_klein():
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )

    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=256,
        axes_dims_rope=[4, 4, 4, 4],
        guidance_embeds=False,
    )
    vae = AutoencoderKLFlux2(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    # The RDNA test image replaces torch GroupNorm with AITER's implementation,
    # while DistVAE's adapter reads the standard module metadata.
    for module in vae.modules():
        if isinstance(module, torch.nn.GroupNorm) and not hasattr(module, "num_channels"):
            module.num_channels = module.weight.numel()
    return Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        is_distilled=True,
    )


def _worker(rank, world_size, init_method, parallel_vae=False, reference_image=False):
    from xfuser.model_executor.pipelines.pipeline_flux2_klein import xFuserFlux2KleinPipeline

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    args = xFuserArgs(
        model="tiny",
        pipefusion_parallel_degree=world_size,
        num_pipeline_patch=2,
        warmup_steps=1,
        attention_backend="sdpa",
        use_parallel_vae=parallel_vae,
    )
    engine_config, _ = args.create_config()
    engine_config.runtime_config.dtype = torch.float32
    pipeline = xFuserFlux2KleinPipeline(_tiny_klein().to(device), engine_config)
    prompt_embeds = torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(2)).to(device)

    def run(image=None):
        return pipeline(
            image=image,
            prompt_embeds=prompt_embeds,
            height=16,
            width=16,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=torch.Generator().manual_seed(1),
            output_type="np" if parallel_vae else "latent",
        )

    baseline = run() if reference_image else None
    image = (
        Image.fromarray(np.random.default_rng(3).integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
        if reference_image
        else None
    )
    output = run(image)

    if is_pipeline_last_stage():
        expected_shape = (1, 16, 16, 3) if parallel_vae else (1, 64, 4)
        assert output.images.shape == expected_shape
        assert torch.isfinite(torch.as_tensor(output.images)).all()
        if reference_image:
            assert not torch.allclose(output.images, baseline.images)
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.multi_gpu
def test_flux2_pipefusion_async_step_completes(accelerator_ranks):
    accelerator_ranks(
        _worker,
        world_size=2,
        timeout=240,
        init_filename="flux2-pipefusion-async",
    )


@pytest.mark.multi_gpu
def test_flux2_pipefusion_composes_with_parallel_vae(accelerator_ranks):
    accelerator_ranks(
        _worker,
        world_size=2,
        timeout=240,
        init_filename="flux2-pipefusion-parallel-vae",
        args=(True,),
    )


@pytest.mark.multi_gpu
def test_flux2_pipefusion_preserves_reference_image_conditioning(accelerator_ranks):
    accelerator_ranks(
        _worker,
        world_size=2,
        timeout=240,
        init_filename="flux2-pipefusion-reference-image",
        args=(False, True),
    )
