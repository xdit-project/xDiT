"""Sequence-parallel transformer wrappers match the single-device model for any token count.

When the token count does not divide the sequence parallel degree, the wrappers
zero-pad the sequence before sharding it, and a batch of prompts of different
lengths pads the shorter prompts. Every case here compares a parallel wrapper
against the stock diffusers transformer with the same tiny random weights, so a
padded token that leaks into the attention keys shows up as a mismatch.
"""

import functools
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

_HEADS = 6
_HEAD_DIM = 16


def _randn(generator, device, *shape):
    return torch.randn(*shape, generator=generator).to(device)


def _wan(device, tokens):
    from diffusers import WanTransformer3DModel

    from xfuser.model_executor.models.transformers.transformer_wan import (
        xFuserWanTransformer3DWrapper,
    )

    frames, height, width = tokens
    config = dict(
        patch_size=(1, 2, 2),
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=32,
    )
    reference = WanTransformer3DModel(**config)
    # The Wan runners always pass a layout dict, even for dense backends.
    parallel = xFuserWanTransformer3DWrapper(**config, attention_kwargs={"thw": None})
    generator = torch.Generator().manual_seed(0)
    inputs = dict(
        hidden_states=_randn(generator, device, 1, 4, frames, 2 * height, 2 * width),
        timestep=torch.tensor([500.0], device=device),
        encoder_hidden_states=_randn(generator, device, 1, 8, 16),
        return_dict=False,
    )
    return reference, parallel, inputs, inputs


def _hunyuan_video15(device, tokens):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (
        xFuserHunyuanVideo15Transformer3DWrapper,
    )

    (frames, height, width), text_tokens = tokens
    config = dict(
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
    )
    reference = HunyuanVideo15Transformer3DModel(**config)
    parallel = xFuserHunyuanVideo15Transformer3DWrapper(**config)
    generator = torch.Generator().manual_seed(0)
    # Three conditioning streams whose lengths add up to text_tokens.
    text, text_2 = text_tokens - 5, 3
    inputs = dict(
        hidden_states=_randn(generator, device, 1, 4, frames, height, width),
        timestep=torch.tensor([500.0], device=device),
        encoder_hidden_states=_randn(generator, device, 1, text, 16),
        encoder_attention_mask=torch.ones(1, text, device=device),
        encoder_hidden_states_2=_randn(generator, device, 1, text_2, 8),
        encoder_attention_mask_2=torch.ones(1, text_2, device=device),
        image_embeds=_randn(generator, device, 1, 2, 8),
        return_dict=False,
    )
    return reference, parallel, inputs, inputs


def _ltx2(device, tokens, rope_type="split"):
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )

    from xfuser.model_executor.models.transformers.transformer_ltx2 import (
        xFuserLTX2VideoTransformer3DWrapper,
    )

    frames, height, width = tokens
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
        # "split" as in the released LTX-2 checkpoint.
        rope_type=rope_type,
    )
    reference = LTX2VideoTransformer3DModel(**config)
    parallel = xFuserLTX2VideoTransformer3DWrapper(**config)
    generator = torch.Generator().manual_seed(0)
    audio_frames = 5
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
    return reference, parallel, inputs, inputs


def _qwen_image(device, tokens):
    from diffusers import QwenImageTransformer2DModel

    from xfuser.model_executor.models.transformers.transformer_qwen import (
        xFuserQwenImageTransformerWrapper,
    )

    # text_tokens is a prompt length, or one per batch item, padded to the
    # longest and masked as the pipeline does.
    (height, width), text_tokens = tokens
    prompt_lengths = text_tokens if isinstance(text_tokens, tuple) else (text_tokens,)
    batch, text_len = len(prompt_lengths), max(prompt_lengths)
    config = dict(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=_HEAD_DIM,
        num_attention_heads=_HEADS,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    reference = QwenImageTransformer2DModel(**config)
    parallel = xFuserQwenImageTransformerWrapper(**config)
    generator = torch.Generator().manual_seed(0)
    inputs = dict(
        hidden_states=_randn(generator, device, batch, height * width, 16),
        encoder_hidden_states=_randn(generator, device, batch, text_len, 16),
        timestep=torch.full((batch,), 0.5, device=device),
        img_shapes=[[(1, height, width)]] * batch,
        return_dict=False,
    )
    if batch > 1:
        lengths = torch.tensor(prompt_lengths)[:, None]
        inputs["encoder_hidden_states_mask"] = (torch.arange(text_len) < lengths).to(device)
    return reference, parallel, inputs, inputs


def _flux2(device, tokens):
    from diffusers import Flux2Transformer2DModel

    from xfuser.model_executor.models.transformers.transformer_flux2 import (
        xFuserFlux2Transformer2DWrapper,
    )

    image_tokens, text_tokens = tokens
    config = dict(
        patch_size=1,
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=_HEAD_DIM,
        num_attention_heads=_HEADS,
        joint_attention_dim=16,
        timestep_guidance_channels=16,
        mlp_ratio=2.0,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    reference = Flux2Transformer2DModel(**config)
    parallel = xFuserFlux2Transformer2DWrapper(**config)
    generator = torch.Generator().manual_seed(0)
    img_ids = torch.zeros(1, image_tokens, 4)
    img_ids[..., 2] = torch.arange(image_tokens) // 4
    img_ids[..., 3] = torch.arange(image_tokens) % 4
    txt_ids = torch.zeros(1, text_tokens, 4)
    txt_ids[..., 3] = torch.arange(text_tokens)
    inputs = dict(
        hidden_states=_randn(generator, device, 1, image_tokens, 8),
        encoder_hidden_states=_randn(generator, device, 1, text_tokens, 16),
        timestep=torch.tensor([0.5], device=device),
        img_ids=img_ids.to(device),
        txt_ids=txt_ids.to(device),
        return_dict=False,
    )
    return reference, parallel, inputs, inputs


def _z_image(device, tokens):
    from diffusers import ZImageTransformer2DModel

    from xfuser.model_executor.models.transformers.transformer_z_image import (
        xFuserZImageTransformer2DWrapper,
    )

    # text_tokens is a caption length, or one per batch item.
    (height, width), text_tokens = tokens
    caption_lengths = text_tokens if isinstance(text_tokens, tuple) else (text_tokens,)
    config = dict(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=_HEADS * _HEAD_DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=_HEADS,
        n_kv_heads=_HEADS,
        cap_feat_dim=16,
        axes_dims=[4, 6, 6],
        # Long enough for the image positions after a 96-token caption.
        axes_lens=[128, 64, 64],
    )
    reference = ZImageTransformer2DModel(**config)
    parallel = xFuserZImageTransformer2DWrapper(**config)
    generator = torch.Generator().manual_seed(0)
    inputs = dict(
        x=[_randn(generator, device, 4, 1, 2 * height, 2 * width) for _ in caption_lengths],
        t=torch.full((len(caption_lengths),), 0.5, device=device),
        cap_feats=[_randn(generator, device, length, 16) for length in caption_lengths],
        return_dict=False,
    )
    return reference, parallel, inputs, inputs


def _cosmos3(device, tokens):
    from diffusers.models.transformers.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
    )

    from xfuser.model_executor.models.transformers.transformer_cosmos3 import (
        get_cosmos3_transformer_wrapper_class,
    )

    (frames, height, width), und_len, kv_heads = tokens
    config = dict(
        head_dim=_HEAD_DIM,
        hidden_size=_HEADS * _HEAD_DIM,
        intermediate_size=64,
        latent_channel=4,
        latent_patch_size=2,
        num_attention_heads=_HEADS,
        num_hidden_layers=2,
        num_key_value_heads=kv_heads,
        patch_latent_dim=16,
        rope_axes_dim=(4, 2, 2),
        vocab_size=64,
    )
    reference = Cosmos3OmniTransformer(**config)
    parallel = Cosmos3OmniTransformer(**config)
    parallel.__class__ = get_cosmos3_transformer_wrapper_class()
    parallel._install_xfuser_processors()

    generator = torch.Generator().manual_seed(0)
    gen_len = frames * height * width
    sequence_length = und_len + gen_len
    position_ids = torch.arange(sequence_length).expand(3, -1).clone()
    inputs = dict(
        input_ids=torch.randint(0, 64, (und_len,), generator=generator).to(device),
        text_indexes=torch.arange(und_len, device=device),
        position_ids=position_ids.to(device),
        und_len=und_len,
        sequence_length=sequence_length,
        vision_tokens=[_randn(generator, device, 1, 4, frames, 2 * height, 2 * width)],
        vision_token_shapes=[(frames, height, width)],
        vision_sequence_indexes=torch.arange(und_len, sequence_length, device=device),
        vision_mse_loss_indexes=torch.arange(und_len, sequence_length, device=device),
        vision_timesteps=torch.full((gen_len,), 500.0, device=device),
        vision_noisy_frame_indexes=[torch.arange(frames, device=device)],
    )
    return reference, parallel, dict(inputs, return_dict=False), dict(inputs, return_dict=False)


_MODELS = {
    "wan": _wan,
    "hunyuan_video15": _hunyuan_video15,
    "ltx2": _ltx2,
    "ltx2_interleaved": functools.partial(_ltx2, rope_type="interleaved"),
    "qwen_image": _qwen_image,
    "flux2": _flux2,
    "z_image": _z_image,
    "cosmos3": _cosmos3,
}


def _flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    tensors = []
    for item in output:
        if item is not None:
            tensors.extend(_flatten(item))
    return tensors


class _Pipeline:
    """The part of a pipeline the runtime state reads: its transformer."""

    def __init__(self, transformer):
        self.transformer = transformer


def _init(rank, world_size, init_method, ulysses, ring):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ring_degree=ring, ulysses_degree=ulysses)
    args = xFuserArgs()
    args.ulysses_degree = ulysses
    args.ring_degree = ring
    engine_config, _ = args.create_config()
    initialize_runtime_state(engine_config=engine_config)
    return engine_config


def _build(model, tokens, device, engine_config, ring):
    torch.manual_seed(0)
    reference, parallel, reference_inputs, parallel_inputs = _MODELS[model](device, tokens)
    reference = reference.to(device).eval()
    parallel.load_state_dict(reference.state_dict())
    parallel = parallel.to(device).eval()
    # The runtime state a pipeline would set up, which the wrappers step.
    initialize_runtime_state(pipeline=_Pipeline(parallel), engine_config=engine_config)
    # SDPA proper has no ring path; its memory-efficient kernel does.
    get_runtime_state().set_attention_backend("SDPA_EFFICIENT" if ring > 1 else "SDPA")
    return reference, parallel, reference_inputs, parallel_inputs


def _parity_worker(rank, world_size, init_method, model, ulysses, ring, tokens):
    engine_config = _init(rank, world_size, init_method, ulysses, ring)
    try:
        device = torch.device("cuda", rank)
        reference, parallel, reference_inputs, parallel_inputs = _build(model, tokens, device, engine_config, ring)
        with torch.no_grad():
            expected = _flatten(reference(**reference_inputs))
            actual = _flatten(parallel(**parallel_inputs))
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


# (model, ulysses degree, sizes). Sizes are each model's latent grid (and text
# length where it is sharded too); see the builders above.
_PARITY_CASES = [
    # Wan: tokens = frames * height * width.
    pytest.param("wan", 2, (2, 3, 3), id="wan-divisible"),
    pytest.param("wan", 2, (3, 3, 3), id="wan-padded"),
    pytest.param("wan", 3, (2, 2, 5), id="wan-padded-u3"),
    # HunyuanVideo-1.5: ((frames, height, width), text tokens).
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), 10), id="hunyuan_video15-divisible"),
    pytest.param("hunyuan_video15", 2, ((3, 3, 3), 10), id="hunyuan_video15-padded-image"),
    pytest.param("hunyuan_video15", 2, ((3, 3, 3), 11), id="hunyuan_video15-padded-image-text"),
    pytest.param("hunyuan_video15", 3, ((2, 3, 4), 12), id="hunyuan_video15-divisible-u3"),
    # LTX-2: video tokens = frames * height * width; audio and text replicated.
    pytest.param("ltx2", 2, (2, 3, 4), id="ltx2-divisible"),
    pytest.param("ltx2", 2, (3, 3, 3), id="ltx2-padded"),
    # Interleaved RoPE keeps cos/sin as [B, S, D] rather than [B, H, S, D].
    pytest.param("ltx2_interleaved", 2, (3, 3, 3), id="ltx2-interleaved-padded"),
    # Qwen-Image: ((height, width) in packed tokens, text tokens).
    pytest.param("qwen_image", 2, ((4, 6), 8), id="qwen_image-divisible"),
    pytest.param("qwen_image", 2, ((5, 5), 8), id="qwen_image-padded-image"),
    pytest.param("qwen_image", 2, ((4, 6), 7), id="qwen_image-replicated-text"),
    pytest.param("qwen_image", 3, ((5, 5), 7), id="qwen_image-padded-image-replicated-text"),
    # A batch of two prompts of different lengths, padded and masked.
    pytest.param("qwen_image", 2, ((4, 6), (8, 5)), id="qwen_image-batched"),
    # Prompts of equal length: the mask is all True and masks nothing.
    pytest.param("qwen_image", 2, ((4, 6), (8, 8)), id="qwen_image-batched-equal-lengths"),
    pytest.param("qwen_image", 2, ((5, 5), (8, 5)), id="qwen_image-batched-padded-image"),
    pytest.param("qwen_image", 2, ((4, 6), (7, 3)), id="qwen_image-batched-replicated-text"),
    pytest.param("qwen_image", 3, ((5, 5), (9, 4)), id="qwen_image-batched-padded-image-u3"),
    pytest.param("qwen_image", 3, ((5, 5), (7, 4)), id="qwen_image-batched-padded-image-replicated-text-u3"),
    # FLUX.2: (image tokens, text tokens); its wrapper shards the text evenly.
    pytest.param("flux2", 2, (16, 8), id="flux2-divisible"),
    pytest.param("flux2", 2, (15, 8), id="flux2-padded"),
    pytest.param("flux2", 3, (16, 9), id="flux2-padded-u3"),
    # Z-Image pads every stream to a multiple of 32 tokens, so only degrees
    # that do not divide 32 need sequence-parallel padding.
    pytest.param("z_image", 2, ((4, 4), 10), id="z_image-divisible"),
    pytest.param("z_image", 3, ((4, 4), 10), id="z_image-padded-u3"),
    # Two captions of different lengths, 32 and 64 tokens once padded to a
    # multiple of 32, or 32 and 96: the caption and unified streams are masked,
    # and at Ulysses 3 one or both of them are padded as well.
    pytest.param("z_image", 2, ((4, 4), (10, 40)), id="z_image-batched"),
    pytest.param("z_image", 3, ((4, 4), (10, 40)), id="z_image-batched-padded-caption-u3"),
    pytest.param("z_image", 3, ((4, 4), (10, 70)), id="z_image-batched-padded-unified-u3"),
    # Cosmos3: ((frames, height, width) gen tokens, und tokens, KV heads). The
    # und tokens are replicated on every rank; grouped-query attention as in
    # the released checkpoints.
    pytest.param("cosmos3", 2, ((2, 2, 3), 5, 2), id="cosmos3-divisible"),
    pytest.param("cosmos3", 2, ((3, 1, 3), 5, 2), id="cosmos3-padded"),
    pytest.param("cosmos3", 3, ((2, 2, 2), 5, 3), id="cosmos3-padded-u3"),
    # One KV head per query head, which the parallel attention also served
    # before grouped-query support: isolates the replicated und keys.
    pytest.param("cosmos3", 2, ((2, 2, 3), 5, _HEADS), id="cosmos3-divisible-mha"),
]


@pytest.mark.multi_gpu
@pytest.mark.parametrize("model, ulysses, tokens", _PARITY_CASES)
def test_ulysses_matches_single_device(accelerator_ranks, model, ulysses, tokens):
    accelerator_ranks(
        _parity_worker,
        world_size=ulysses,
        args=(model, ulysses, 1, tokens),
    )


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _ring_padding_worker(rank, world_size, init_method):
    engine_config = _init(rank, world_size, init_method, ulysses=1, ring=world_size)
    records = _Records()
    logging.getLogger("xfuser.model_executor.layers.usp").addHandler(records)
    try:
        device = torch.device("cuda", rank)
        # 7 * 3 * 3 = 63 tokens pad to 64: one padded key, and 32 queries per
        # rank, which the memory-efficient kernel needs to merge ring steps.
        _, parallel, _, inputs = _build("wan", (7, 3, 3), device, engine_config, world_size)
        with torch.no_grad():
            first = parallel(**inputs)[0]
            second = parallel(**inputs)[0]
        assert first.shape == inputs["hidden_states"].shape
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second)
        # Logged once, and only by rank 0.
        warnings = [m for m in records.messages if "Ring attention cannot exclude" in m]
        assert len(warnings) == (1 if rank == 0 else 0), records.messages
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
def test_ring_with_padding_warns_once_and_runs(accelerator_ranks):
    accelerator_ranks(_ring_padding_worker, world_size=2)


def _ring_key_padding_mask_worker(rank, world_size, init_method):
    engine_config = _init(rank, world_size, init_method, ulysses=1, ring=world_size)
    records = _Records()
    logging.getLogger("xfuser.model_executor.layers.usp").addHandler(records)
    try:
        device = torch.device("cuda", rank)
        # Captions of 64 and 32 tokens: masked caption and unified streams, no
        # sequence-parallel padding, and at least 32 queries per rank.
        _, parallel, _, inputs = _build("z_image", ((8, 8), (40, 20)), device, engine_config, world_size)
        with torch.no_grad():
            first = parallel(**inputs)[0]
            second = parallel(**inputs)[0]
        assert [x.shape for x in first] == [x.shape for x in inputs["x"]]
        assert all(torch.isfinite(x).all() for x in first)
        torch.testing.assert_close(first, second)
        # Logged once, and only by rank 0.
        warnings = [m for m in records.messages if "Ring attention cannot apply the padding mask" in m]
        assert len(warnings) == (1 if rank == 0 else 0), records.messages
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
def test_ring_with_batched_prompts_warns_once_and_runs(accelerator_ranks):
    accelerator_ranks(_ring_key_padding_mask_worker, world_size=2)
