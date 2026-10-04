"""Sparse (SSTA) HunyuanVideo-1.5 under Ulysses matches the single-device model.

The sparse path keeps a symmetric [image, text] layout on every rank, so when
the image or the text token count does not divide the sequence parallel degree,
the wrapper zero-pads it before sharding. Neither kind of padding may take part
in the sparse attention: the single-device reference has none. Stock diffusers
has no sparse attention, so the reference is the same wrapper without sequence
parallelism.

flex_block_attn, the block-sparse kernel the SSTA backend calls, is a separate
package that only builds for Hopper GPUs. It is replaced here by the attention
it computes, dense attention under the block mask expanded to tokens, so the
test exercises the wrapper and the SSTA tiling and masking around the kernel.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

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


def _block_sparse_attention(q, k, v, block_m, block_n, block_mask):
    """What flex_block_attn_func computes: full attention within the selected
    (query block, key block) pairs and none outside them."""
    mask = block_mask.repeat_interleave(block_m, dim=-2).repeat_interleave(block_n, dim=-1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _install_block_sparse_kernel():
    module = types.ModuleType("flex_block_attn")
    module.flex_block_attn_func = _block_sparse_attention
    sys.modules["flex_block_attn"] = module


def _sparse_attention_kwargs():
    # The released sparse checkpoint's attn_param, at a tile small enough for
    # a tiny latent grid: 4-token blocks, so the text spans several blocks.
    return {
        "attn_mask_share_within_head": 0,
        "attn_pad_type": "zero",
        "attn_sparse_type": "ssta",
        "attn_use_text_mask": 1,
        "ssta_adaptive_pool": None,
        "ssta_lambda": 0.7,
        "ssta_sampling_type": "importance",
        "ssta_threshold": 0.0,
        "ssta_topk": 8,
        "tile_size": [1, 2, 2],
        "win_size": [[1, 1, 1]],
        "sparse_text_to_image": False,
    }


def _model_and_inputs(device, tokens):
    from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (
        xFuserHunyuanVideo15Transformer3DWrapper,
    )

    (frames, height, width), text_tokens = tokens
    torch.manual_seed(0)
    model = xFuserHunyuanVideo15Transformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        qk_norm="rms_norm",
        text_embed_dim=16,
        text_embed_2_dim=8,
        image_embed_dim=8,
        rope_axes_dim=(4, 6, 6),
        task_type="i2v",
        attention_kwargs=_sparse_attention_kwargs(),
    )
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    # Three conditioning streams whose lengths add up to text_tokens.
    text, text_2 = text_tokens - 5, 3
    inputs = dict(
        hidden_states=randn(1, 4, frames, height, width),
        timestep=torch.tensor([500.0], device=device),
        encoder_hidden_states=randn(1, text, 16),
        encoder_attention_mask=torch.ones(1, text, device=device),
        encoder_hidden_states_2=randn(1, text_2, 8),
        encoder_attention_mask_2=torch.ones(1, text_2, device=device),
        image_embeds=randn(1, 2, 8),
        return_dict=False,
    )
    return model.to(device).eval(), inputs


class _Pipeline:
    """The part of a pipeline the runtime state reads: its transformer."""

    def __init__(self, transformer):
        self.transformer = transformer


def _parallelize(data_parallel, ulysses):
    initialize_model_parallel(data_parallel_degree=data_parallel, ulysses_degree=ulysses)
    args = xFuserArgs()
    args.data_parallel_degree = data_parallel
    args.ulysses_degree = ulysses
    engine_config, _ = args.create_config()
    return engine_config


def _forward(device, tokens, engine_config):
    model, inputs = _model_and_inputs(device, tokens)
    initialize_runtime_state(pipeline=_Pipeline(model), engine_config=engine_config)
    get_runtime_state().set_attention_backend("FLEX_BLOCK_ATTN")
    with torch.no_grad():
        return model(**inputs)[0]


def _parity_worker(rank, world_size, init_method, tokens):
    _install_block_sparse_kernel()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    try:
        # Every rank first runs the whole sequence, without sequence
        # parallelism, as the reference.
        expected = _forward(device, tokens, _parallelize(data_parallel=world_size, ulysses=1))
        destroy_model_parallel()
        actual = _forward(device, tokens, _parallelize(data_parallel=1, ulysses=world_size))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


# (ulysses degree, ((frames, height, width), text tokens)). The text is
# 4-token blocks, so its sequence-parallel padding shares a block with prompt
# tokens unless the prompt fills its last block.
_CASES = [
    pytest.param(2, ((2, 4, 4), 12), id="divisible"),
    pytest.param(2, ((3, 3, 3), 12), id="padded-image"),
    pytest.param(2, ((2, 4, 4), 11), id="padded-text"),
    pytest.param(3, ((2, 4, 5), 10), id="padded-image-text-u3"),
]


@pytest.mark.multi_gpu
@pytest.mark.parametrize("ulysses, tokens", _CASES)
def test_sparse_ulysses_matches_single_device(accelerator_ranks, ulysses, tokens):
    accelerator_ranks(_parity_worker, world_size=ulysses, args=(tokens,))
