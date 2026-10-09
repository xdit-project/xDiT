"""DDPM steps a sequence-parallel latent as one device steps the whole latent.

DDPMScheduler adds generator noise at every step. Stepping each rank's rows on
its own drew the same noise patch on every rank, which drove HunyuanDiT
(Ulysses or ring) far from its single-device image.
"""

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import (
    get_runtime_state,
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
from xfuser.model_executor.schedulers import xFuserDDPMSchedulerWrapper

# HunyuanDiT-v1.2's scheduler.
SCHEDULER_CONFIG = dict(
    beta_start=0.00085,
    beta_end=0.018,
    beta_schedule="scaled_linear",
    prediction_type="v_prediction",
    variance_type="fixed_small",
    clip_sample=False,
    steps_offset=1,
    timestep_spacing="leading",
)
HEIGHT, WIDTH = 256, 192  # latent rows split across ranks
STEPS = 4


def _worker(rank, world_size, init_method, ulysses_degree, ring_degree):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    try:
        args = xFuserArgs()
        args.ulysses_degree = ulysses_degree
        args.ring_degree = ring_degree
        engine_config, _ = args.create_config()
        config = SimpleNamespace(patch_size=2, in_channels=4, num_attention_heads=2, attention_head_dim=8)
        pipeline = SimpleNamespace(transformer=SimpleNamespace(config=config), vae_scale_factor=8)
        initialize_runtime_state(pipeline=pipeline, engine_config=engine_config)
        get_runtime_state().set_input_parameters(height=HEIGHT, width=WIDTH, batch_size=1, num_inference_steps=STEPS)

        reference = DDPMScheduler(**SCHEDULER_CONFIG)
        parallel = xFuserDDPMSchedulerWrapper(DDPMScheduler(**SCHEDULER_CONFIG))
        reference.set_timesteps(STEPS, device=device)
        parallel.set_timesteps(STEPS, device=device)

        inputs = torch.Generator().manual_seed(0)
        shape = (1, 4, HEIGHT // 8, WIDTH // 8)
        full = torch.randn(shape, generator=inputs).to(device)
        rows = full.shape[-2] // world_size
        local = full[..., rank * rows : (rank + 1) * rows, :]
        reference_noise = torch.Generator(device).manual_seed(42)
        parallel_noise = torch.Generator(device).manual_seed(42)

        for t in reference.timesteps:
            model_output = torch.randn(shape, generator=inputs).to(device)
            full = reference.step(model_output, t, full, generator=reference_noise, return_dict=False)[0]
            local_output = model_output[..., rank * rows : (rank + 1) * rows, :]
            local = parallel.step(local_output, t, local, generator=parallel_noise).prev_sample

            torch.testing.assert_close(local, full[..., rank * rows : (rank + 1) * rows, :])
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("ulysses_degree, ring_degree", [(2, 1), (1, 2)], ids=["ulysses", "ring"])
def test_sequence_parallel_ddpm_step_matches_whole_latent_step(ulysses_degree, ring_degree, accelerator_ranks):
    accelerator_ranks(
        _worker,
        world_size=ulysses_degree * ring_degree,
        init_filename="ddpm-init",
        args=(ulysses_degree, ring_degree),
    )
