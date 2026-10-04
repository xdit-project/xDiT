"""HunyuanVideo-1.5 masks each prompt's padding out of the attention keys.

The stock transformer pads the prompts of a batch to one length and masks the
padded tokens out of the keys per sample. Every case compares the xDiT wrapper
against the stock diffusers transformer with the same tiny random weights, on
one device and under Ulysses, for a batch of prompts of different lengths and
for a single prompt. With the SDPA backend the wrapper keeps the stock text
layout; with a backend that ignores masks (SDPA's memory-efficient kernel
here) it drops unused text columns and masks only prompts of different
lengths.
"""

import logging

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

_TEXT_DIM = 16
_TEXT_2_DIM = 8
_IMAGE_DIM = 8
_IMAGE_TOKENS = 2


def _models(task_type):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (
        xFuserHunyuanVideo15Transformer3DWrapper,
    )

    config = dict(
        in_channels=4,
        out_channels=4,
        num_attention_heads=6,
        attention_head_dim=16,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        qk_norm="rms_norm",
        text_embed_dim=_TEXT_DIM,
        text_embed_2_dim=_TEXT_2_DIM,
        image_embed_dim=_IMAGE_DIM,
        rope_axes_dim=(4, 6, 6),
        task_type=task_type,
    )
    return HunyuanVideo15Transformer3DModel(**config), xFuserHunyuanVideo15Transformer3DWrapper(**config)


def _mask(lengths, size, device):
    mask = torch.zeros(len(lengths), size, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return mask


def _inputs(device, grid, text_lengths, text_2_lengths, task_type):
    """Inputs for a batch with one sample per entry of ``text_lengths``."""
    frames, height, width = grid
    batch_size = len(text_lengths)
    text, text_2 = max(text_lengths) + 2, max(text_2_lengths) + 1
    generator = torch.Generator().manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    # Text-to-video passes all-zero image embeddings, which the model masks out.
    image_embeds = randn(batch_size, _IMAGE_TOKENS, _IMAGE_DIM)
    if task_type == "t2v":
        image_embeds = torch.zeros_like(image_embeds)
    return dict(
        hidden_states=randn(batch_size, 4, frames, height, width),
        timestep=torch.full((batch_size,), 500.0, device=device),
        encoder_hidden_states=randn(batch_size, text, _TEXT_DIM),
        encoder_attention_mask=_mask(text_lengths, text, device),
        encoder_hidden_states_2=randn(batch_size, text_2, _TEXT_2_DIM),
        encoder_attention_mask_2=_mask(text_2_lengths, text_2, device),
        image_embeds=image_embeds,
        return_dict=False,
    )


class _Pipeline:
    """The part of a pipeline the runtime state reads: its transformer."""

    def __init__(self, transformer):
        self.transformer = transformer


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _run(rank, world_size, init_method, ring, case, backend):
    """Return (stock output, wrapper output, USP log messages) on this rank."""
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    ulysses = world_size // ring
    initialize_model_parallel(ring_degree=ring, ulysses_degree=ulysses)
    records = _Records()
    logging.getLogger("xfuser.model_executor.layers.usp").addHandler(records)
    try:
        args = xFuserArgs()
        args.ulysses_degree = ulysses
        args.ring_degree = ring
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)

        device = torch.device("cuda", rank)
        grid, text_lengths, text_2_lengths, task_type = case
        torch.manual_seed(0)
        reference, wrapper = _models(task_type)
        reference = reference.to(device).eval()
        wrapper.load_state_dict(reference.state_dict())
        wrapper = wrapper.to(device).eval()
        initialize_runtime_state(pipeline=_Pipeline(wrapper), engine_config=engine_config)
        get_runtime_state().set_attention_backend(backend)

        inputs = _inputs(device, grid, text_lengths, text_2_lengths, task_type)
        with torch.no_grad():
            expected = reference(**inputs)[0]
            actual = wrapper(**inputs)[0]
        return expected, actual, records.messages
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _parity_worker(rank, world_size, init_method, case, backend):
    expected, actual, _ = _run(rank, world_size, init_method, 1, case, backend)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


# (latent grid, text lengths, byT5 lengths, task), one length per sample. The
# wrapper keeps the text tokens that some sample uses, so the longest prompt
# sets the text length and the shorter one is padded. Image-to-video adds two
# image-embedding tokens to every prompt.
_BATCHED = (
    pytest.param(((2, 3, 4), (7, 4), (3, 1), "t2v"), id="t2v"),
    pytest.param(((2, 3, 4), (7, 4), (3, 1), "i2v"), id="i2v"),
)
_SINGLE = (pytest.param(((2, 3, 4), (5,), (2,), "t2v"), id="t2v-single"),)
# Under Ulysses 2: 24 image tokens and 10 (12 for i2v) text tokens shard
# evenly; 27 image tokens are padded, so the text is replicated as joint keys;
# 11 text tokens do not shard evenly, so they are replicated too.
_ULYSSES = (
    pytest.param(((2, 3, 4), (7, 4), (3, 1), "t2v"), id="t2v-sharded-text"),
    pytest.param(((2, 3, 4), (7, 4), (3, 1), "i2v"), id="i2v-sharded-text"),
    pytest.param(((3, 3, 3), (7, 4), (3, 1), "t2v"), id="t2v-padded-image"),
    pytest.param(((2, 3, 4), (8, 4), (3, 1), "t2v"), id="t2v-replicated-text"),
    pytest.param(((2, 3, 4), (6,), (2,), "t2v"), id="t2v-single"),
)


_BACKENDS = ("SDPA", "SDPA_EFFICIENT")


@pytest.mark.parametrize("case", _BATCHED + _SINGLE)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_single_device_matches_stock(accelerator_ranks, case, backend):
    accelerator_ranks(_parity_worker, world_size=1, args=(case, backend))


@pytest.mark.multi_gpu
@pytest.mark.parametrize("case", _ULYSSES)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_ulysses_matches_stock(accelerator_ranks, case, backend):
    accelerator_ranks(_parity_worker, world_size=2, args=(case, backend))


def _ring_worker(rank, world_size, init_method):
    # 27 image and 5 text tokens per rank: the memory-efficient kernel merges
    # ring steps only for a multiple of 32 queries.
    case = ((2, 3, 9), (7, 4), (3, 1), "t2v")
    # SDPA proper has no ring path; its memory-efficient kernel does.
    _, actual, messages = _run(rank, world_size, init_method, world_size, case, "SDPA_EFFICIENT")
    assert torch.isfinite(actual).all()
    warnings = [m for m in messages if "Ring attention cannot apply the padding mask" in m]
    assert len(warnings) == 1, messages


@pytest.mark.multi_gpu
def test_ring_with_prompts_of_different_lengths_warns_once(accelerator_ranks):
    accelerator_ranks(_ring_worker, world_size=2)
