"""Prism's two-tower transformer gives the same velocities at every Ulysses degree.

A tiny randomly initialised MOVABridge runs once on a single rank and again sharded
across ranks, on both video experts. Neither the 45 video tokens nor the 13 audio
tokens split evenly across 2 or 4 ranks, which exercises the trailing padding that
attention must drop, and the 3-head audio space of the video-to-audio bridge does not
divide either degree, which exercises its zero-head padding. Nothing is downloaded.
"""

import pytest

pytestmark = pytest.mark.multi_gpu

# Head dim 24: Wan's 3D RoPE splits it into 8 + 8 + 8, which needs dim // 3 even.
_VIDEO = dict(
    dim=96,
    in_dim=8,
    ffn_dim=192,
    out_dim=4,
    text_dim=32,
    freq_dim=32,
    eps=1e-6,
    patch_size=(1, 2, 2),
    num_heads=4,
    num_layers=3,
)
_AUDIO = dict(
    dim=72,
    in_dim=8,
    ffn_dim=144,
    out_dim=8,
    text_dim=32,
    freq_dim=32,
    eps=1e-6,
    patch_size=(1,),
    num_heads=3,
    num_layers=2,
)
_BRIDGE = dict(
    visual_layers=3,
    audio_layers=2,
    visual_hidden_dim=96,
    audio_hidden_dim=72,
    audio_fps=50.0,
    head_dim=24,
    interaction_strategy="full",
    apply_cross_rope=True,
)


def _outputs(device):
    import torch

    from xfuser.model_executor.models.customized.prism.bridge import DualTowerConditionalBridge
    from xfuser.model_executor.models.customized.prism.mova import MOVABridge
    from xfuser.model_executor.models.customized.prism.wan_dit import WanAudioModel, WanModel

    torch.manual_seed(0)
    model = MOVABridge(
        WanModel(**_VIDEO),
        WanModel(**_VIDEO),
        WanAudioModel(**_AUDIO),
        DualTowerConditionalBridge(**_BRIDGE),
    )
    model = model.to(device).eval()

    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        visual_latents=torch.randn(1, 8, 3, 6, 10, generator=generator),
        audio_latents=torch.randn(1, 8, 13, generator=generator),
        context=torch.randn(1, 7, 32, generator=generator),
        audio_context=torch.randn(1, 7, 32, generator=generator),
        timestep=torch.tensor([700.0]),
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        return {
            expert: tuple(out.cpu() for out in model(**inputs, use_video_dit_2=expert == "low"))
            for expert in ("high", "low")
        }


def _prism_worker(rank, world_size, init_method, reference_path):
    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ulysses_degree=world_size)
    try:
        engine_config, _ = xFuserArgs(attention_backend="SDPA", ulysses_degree=world_size).create_config()
        initialize_runtime_state(engine_config=engine_config)
        get_runtime_state().set_attention_backend("SDPA")
        outputs = _outputs(torch.device(f"cuda:{rank}"))
        if world_size == 1:
            torch.save(outputs, reference_path)
            return
        expected = torch.load(reference_path)
        for expert in ("high", "low"):
            for actual, reference in zip(outputs[expert], expected[expert]):
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("ulysses_degree", [2, 4])
def test_ulysses_matches_a_single_rank(ulysses_degree, accelerator_ranks, tmp_path):
    reference_path = tmp_path / "single_rank.pt"
    accelerator_ranks(_prism_worker, world_size=1, init_filename="prism-u1", args=(reference_path,))
    accelerator_ranks(
        _prism_worker,
        world_size=ulysses_degree,
        init_filename=f"prism-u{ulysses_degree}",
        args=(reference_path,),
    )
