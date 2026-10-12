"""LTX-2 with interleaved RoPE matches the stock transformer under Ulysses.

The video-to-audio attention gathers the sharded video keys, and their RoPE
with them. Interleaved cos/sin are [B, S, D] while split ones are
[B, H, S, D // 2], so the gather has to follow the token dimension of
whichever layout the model uses. The released checkpoint uses split RoPE; the
interleaved layout is the diffusers default.
"""

import pytest
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

_HEADS = 6
_HEAD_DIM = 16


class _Pipeline:
    """The part of a pipeline the runtime state reads: its transformer."""

    def __init__(self, transformer):
        self.transformer = transformer


def _randn(generator, device, *shape):
    return torch.randn(*shape, generator=generator).to(device)


def _parity_worker(rank, world_size, init_method, rope_type):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    from xfuser.model_executor.models.transformers.transformer_ltx2 import (
        xFuserLTX2VideoTransformer3DWrapper,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size)
    try:
        args = xFuserArgs()
        args.ulysses_degree = world_size
        engine_config, _ = args.create_config()
        device = torch.device("cuda", rank)

        config = dict(
            in_channels=8,
            out_channels=8,
            num_attention_heads=_HEADS,
            attention_head_dim=_HEAD_DIM,
            cross_attention_dim=_HEADS * _HEAD_DIM,
            audio_in_channels=8,
            audio_out_channels=8,
            audio_num_attention_heads=_HEADS,
            audio_attention_head_dim=_HEAD_DIM // 2,
            audio_cross_attention_dim=_HEADS * _HEAD_DIM // 2,
            num_layers=2,
            caption_channels=16,
            rope_type=rope_type,
        )
        torch.manual_seed(0)
        reference = LTX2VideoTransformer3DModel(**config).to(device).eval()
        parallel = xFuserLTX2VideoTransformer3DWrapper(**config)
        parallel.load_state_dict(reference.state_dict())
        parallel = parallel.to(device).eval()
        initialize_runtime_state(pipeline=_Pipeline(parallel), engine_config=engine_config)
        get_runtime_state().set_attention_backend("SDPA")

        # 2 x 3 x 4 video tokens: divisible by the degree, so no padding.
        frames, height, width, audio_frames = 2, 3, 4, 5
        generator = torch.Generator().manual_seed(0)
        inputs = dict(
            hidden_states=_randn(generator, device, 1, frames * height * width, 8),
            audio_hidden_states=_randn(generator, device, 1, audio_frames, 8),
            encoder_hidden_states=_randn(generator, device, 1, 6, 16),
            audio_encoder_hidden_states=_randn(generator, device, 1, 6, 16),
            timestep=torch.tensor([500.0], device=device),
            num_frames=frames,
            height=height,
            width=width,
            audio_num_frames=audio_frames,
            return_dict=False,
        )
        with torch.no_grad():
            expected = reference(**inputs)
            actual = parallel(**inputs)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
def test_ulysses_matches_the_stock_transformer(accelerator_ranks, rope_type):
    accelerator_ranks(_parity_worker, world_size=2, args=(rope_type,))
