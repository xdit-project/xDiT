"""Legacy PipeFusion pipelines with their own async loop must complete asynchronous steps.

FLUX Control and Sana still drive PipeFusion's asynchronous phase themselves and
report each step through the base wrapper's legacy callback helper. Two pipeline
stages run one synchronous warmup step and two asynchronous steps on tiny models;
the last stage must return finite latents and the callback must see every step once.
"""

import importlib

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import init_distributed_environment, is_pipeline_last_stage
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

STEPS = 3


def _tiny_flux_control():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxTransformer2DModel

    torch.manual_seed(0)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=32,
        out_channels=16,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=True,
        axes_dims_rope=[4, 6, 6],
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.0,
        scaling_factor=1.0,
    )
    # The RDNA test image replaces torch GroupNorm with AITER's implementation,
    # while diffusers reads the standard module metadata.
    for module in vae.modules():
        if isinstance(module, torch.nn.GroupNorm) and not hasattr(module, "num_channels"):
            module.num_channels = module.weight.numel()
    pipeline = FluxControlPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )
    generator = torch.Generator().manual_seed(1)
    kwargs = {
        "control_image": torch.rand(1, 3, 16, 16, generator=generator),
        "prompt_embeds": torch.randn(1, 8, 32, generator=generator),
        "pooled_prompt_embeds": torch.randn(1, 16, generator=generator),
        "max_sequence_length": 8,
        "height": 16,
        "width": 16,
        "guidance_scale": 3.5,
    }
    return pipeline, kwargs, (1, 64, 16)


def _tiny_sana():
    from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel

    torch.manual_seed(0)
    transformer = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        sample_size=8,
        patch_size=1,
    )
    pipeline = SanaPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    generator = torch.Generator().manual_seed(1)
    kwargs = {
        "prompt_embeds": torch.randn(1, 6, 8, generator=generator),
        "prompt_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "negative_prompt": None,
        "latents": torch.randn(1, 4, 8, 8, generator=generator),
        "height": 256,
        "width": 256,
        "guidance_scale": 1.0,
        "use_resolution_binning": False,
    }
    return pipeline, kwargs, (1, 4, 8, 8)


# name -> (builder, wrapper module, wrapper class)
CASES = {
    "flux_control": (_tiny_flux_control, "pipeline_flux_control", "xFuserPipelineFluxControlPipeline"),
    "sana": (_tiny_sana, "pipeline_sana", "xFuserSanaPipeline"),
}


def _worker(rank, world_size, init_method, name):
    build, module_name, wrapper_name = CASES[name]
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    args = xFuserArgs(
        model="tiny",
        pipefusion_parallel_degree=world_size,
        num_pipeline_patch=2,
        warmup_steps=1,
        attention_backend="sdpa",
    )
    engine_config, _ = args.create_config()
    engine_config.runtime_config.dtype = torch.float32
    pipeline, kwargs, expected_shape = build()
    wrapper_cls = getattr(importlib.import_module(f"xfuser.model_executor.pipelines.{module_name}"), wrapper_name)
    wrapper = wrapper_cls(pipeline.to(device), engine_config)
    steps = []
    kwargs = {
        **{key: value.to(device) if torch.is_tensor(value) else value for key, value in kwargs.items()},
        "num_inference_steps": STEPS,
        "output_type": "latent",
        "callback_on_step_end": lambda pipe, step, t, callback_kwargs: steps.append(step) or callback_kwargs,
    }
    output = wrapper(**kwargs)
    if is_pipeline_last_stage():
        latents = output[0]
        assert tuple(latents.shape) == expected_shape
        assert torch.isfinite(latents).all()
        assert steps == list(range(STEPS))
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_pipefusion_async_steps_complete(accelerator_ranks, name):
    accelerator_ranks(_worker, world_size=2, timeout=300, init_filename=f"legacy-pipefusion-{name}", args=(name,))
