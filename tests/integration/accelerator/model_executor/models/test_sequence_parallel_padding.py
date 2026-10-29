"""Sequence-parallel transformer wrappers match the single-device model for any token count.

When the token count does not divide the sequence parallel degree, the wrappers
zero-pad the sequence before sharding it, and a batch of prompts of different
lengths pads the shorter prompts. Every case here compares a parallel wrapper
against the stock diffusers transformer with the same tiny random weights, so a
padded token that leaks into the attention keys shows up as a mismatch.

These need NCCL and xDiT's own process groups, which the wrappers read (`get_sp_group`);
the gloo tests wire only USP by hand. USP's trimming with joint tokens, Ulysses and ring
is therefore covered on CPU in `tests/integration/cpu/.../test_usp_sequence_parallel_pad.py`,
and each wrapper's declared length and sharding in the unit tests beside the wrappers.
"""

import functools

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


def _hunyuan_video15(device, tokens):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (
        xFuserHunyuanVideo15Transformer3DWrapper,
    )

    # text_tokens is a prompt length, or one per batch item; the shorter prompts
    # are padded and masked as the pipeline does.
    (frames, height, width), text_tokens = tokens
    prompt_lengths = text_tokens if isinstance(text_tokens, tuple) else (text_tokens,)
    batch = len(prompt_lengths)
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
    # Three conditioning streams whose valid lengths add up to the prompt length.
    text, text_2 = max(prompt_lengths) - 5, 3
    text_lengths = torch.tensor(prompt_lengths)[:, None] - 5
    inputs = dict(
        hidden_states=_randn(generator, device, batch, 4, frames, height, width),
        timestep=torch.full((batch,), 500.0, device=device),
        encoder_hidden_states=_randn(generator, device, batch, text, 16),
        encoder_attention_mask=(torch.arange(text) < text_lengths).float().to(device),
        encoder_hidden_states_2=_randn(generator, device, batch, text_2, 8),
        encoder_attention_mask_2=torch.ones(batch, text_2, device=device),
        image_embeds=_randn(generator, device, batch, 2, 8),
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

    (frames, height, width), und_len = tokens
    config = dict(
        head_dim=_HEAD_DIM,
        hidden_size=_HEADS * _HEAD_DIM,
        intermediate_size=64,
        latent_channel=4,
        latent_patch_size=2,
        num_attention_heads=_HEADS,
        num_hidden_layers=2,
        num_key_value_heads=_HEADS,
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


def _cast(value, dtype):
    """Floating tensors of a (nested) input in ``dtype``; masks and indices keep theirs."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(item, dtype) for item in value)
    return {key: _cast(item, dtype) for key, item in value.items()} if isinstance(value, dict) else value


def _build(model, tokens, device, engine_config, ring, backend=None, dtype=None):
    torch.manual_seed(0)
    reference, parallel, reference_inputs, parallel_inputs = _MODELS[model](device, tokens)
    if dtype is not None:
        reference, parallel = reference.to(dtype), parallel.to(dtype)
        reference_inputs, parallel_inputs = _cast(reference_inputs, dtype), _cast(parallel_inputs, dtype)
    reference = reference.to(device).eval()
    # AITER's RMSNorm registers a `_ones_weight` buffer that the Diffusers
    # reference does not have; nothing else may differ.
    incompatible = parallel.load_state_dict(reference.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.endswith("._ones_weight") for key in incompatible.missing_keys)
    parallel = parallel.to(device).eval()
    # The runtime state a pipeline would set up, which the wrappers step.
    initialize_runtime_state(pipeline=_Pipeline(parallel), engine_config=engine_config)
    # SDPA proper has no ring path (ring=NEVER); its memory-efficient kernel does.
    get_runtime_state().set_attention_backend(backend or ("SDPA_EFFICIENT" if ring > 1 else "SDPA"))
    return reference, parallel, reference_inputs, parallel_inputs


def _parity_worker(rank, world_size, init_method, model, ulysses, ring, tokens, backend=None, dtype=None):
    engine_config = _init(rank, world_size, init_method, ulysses, ring)
    try:
        device = torch.device("cuda", rank)
        reference, parallel, reference_inputs, parallel_inputs = _build(
            model, tokens, device, engine_config, ring, backend, dtype
        )
        checked = []

        def check_padding(module, args, output):
            if model == "hunyuan_video15":
                tensors = [
                    (output[0], parallel._sp_pad_image["local_rows"], 1),
                    (output[1], parallel._sp_pad_text["local_rows"], 1),
                ]
            elif model == "cosmos3":
                tensors = [(output[1], parallel._sp_pad["local_rows"], 0)]
            elif model.startswith("ltx2"):
                tensors = [(output[0], parallel._sp_pad["local_rows"], 1)]
            else:
                image = output[1] if isinstance(output, tuple) else output
                tensors = [(image, parallel._sp_pad["local_rows"], 1)]
            for tensor, rows, dim in tensors:
                if rows:
                    assert tensor.narrow(dim, tensor.shape[dim] - rows, rows).eq(0).all()
                checked.append(rows)

        blocks = parallel.layers if model in ("z_image", "cosmos3") else parallel.transformer_blocks
        for block in blocks:
            block.register_forward_hook(check_padding)
        if model == "flux2":
            for block in parallel.single_transformer_blocks:
                block.register_forward_hook(check_padding)
        elif model == "z_image":
            for block in (*parallel.noise_refiner, *parallel.context_refiner):
                block.register_forward_hook(check_padding)
        with torch.no_grad():
            expected = _flatten(reference(**reference_inputs))
            actual = _flatten(parallel(**parallel_inputs))
        assert len(actual) == len(expected)
        assert checked, "the model must execute the blocks guarded against padding drift"
        # bfloat16 runs on a flash backend, whose kernel rounds differently from the reference's.
        tolerance = dict(rtol=1e-4, atol=1e-4) if dtype is None else dict(rtol=5e-2, atol=5e-2)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, **tolerance)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


# (model, ulysses degree, sizes). Sizes are each model's latent grid (and text
# length where it is sharded too); see the builders above.
_PARITY_CASES = [
    # HunyuanVideo-1.5: ((frames, height, width), text tokens).
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), 10), id="hunyuan_video15-divisible"),
    pytest.param("hunyuan_video15", 2, ((3, 3, 3), 10), id="hunyuan_video15-padded-image"),
    pytest.param("hunyuan_video15", 2, ((3, 3, 3), 11), id="hunyuan_video15-padded-image-text"),
    pytest.param("hunyuan_video15", 3, ((2, 3, 4), 12), id="hunyuan_video15-divisible-u3"),
    # Two prompts of different lengths: the shorter one is padded and masked. 10
    # text tokens shard evenly; 11 do not, and a padded image leaves the text
    # joint, so each of those keeps the prompt as joint keys behind the image.
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), (10, 7)), id="hunyuan_video15-batched"),
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), (10, 10)), id="hunyuan_video15-batched-equal-lengths"),
    pytest.param("hunyuan_video15", 2, ((3, 3, 3), (10, 7)), id="hunyuan_video15-batched-padded-image"),
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), (11, 7)), id="hunyuan_video15-batched-replicated-text"),
    pytest.param("hunyuan_video15", 3, ((2, 3, 4), (12, 8)), id="hunyuan_video15-batched-u3"),
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
    # FLUX.2: (image tokens, text tokens). A text that does not divide the degree is replicated.
    pytest.param("flux2", 2, (16, 8), id="flux2-divisible"),
    pytest.param("flux2", 2, (15, 8), id="flux2-padded"),
    pytest.param("flux2", 3, (16, 9), id="flux2-padded-u3"),
    pytest.param("flux2", 2, (16, 7), id="flux2-replicated-text"),
    pytest.param("flux2", 3, (16, 8), id="flux2-padded-replicated-text-u3"),
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
    # Cosmos3: ((frames, height, width) gen tokens, und tokens). The und tokens
    # are replicated on every rank. One KV head per query head: the SDPA kernels
    # here are called without `enable_gqa`.
    pytest.param("cosmos3", 2, ((2, 2, 3), 5), id="cosmos3-divisible"),
    pytest.param("cosmos3", 2, ((3, 1, 3), 5), id="cosmos3-padded"),
    pytest.param("cosmos3", 3, ((2, 2, 2), 5), id="cosmos3-padded-u3"),
]


@pytest.mark.multi_gpu
@pytest.mark.parametrize("model, ulysses, tokens", _PARITY_CASES)
def test_ulysses_matches_single_device(accelerator_ranks, model, ulysses, tokens):
    accelerator_ranks(
        _parity_worker,
        world_size=ulysses,
        args=(model, ulysses, 1, tokens),
    )


_RING_CASES = [
    pytest.param("hunyuan_video15", 1, 2, ((7, 3, 3), 10), id="hunyuan-padded-video-sharded-text"),
    pytest.param("hunyuan_video15", 1, 2, ((7, 3, 3), 11), id="hunyuan-padded-video-joint-text"),
    pytest.param("ltx2", 1, 2, (7, 3, 3), id="ltx2-padded"),
    pytest.param("ltx2_interleaved", 1, 2, (7, 3, 3), id="ltx2-interleaved-padded"),
    pytest.param("qwen_image", 1, 2, ((7, 9), 8), id="qwen-padded-image-sharded-text"),
    pytest.param("qwen_image", 1, 2, ((7, 9), 7), id="qwen-padded-image-joint-text"),
    pytest.param("flux2", 1, 2, (63, 8), id="flux2-padded"),
    pytest.param("flux2", 1, 2, (63, 7), id="flux2-padded-replicated-text"),
    pytest.param("cosmos3", 1, 2, ((7, 3, 3), 5), id="cosmos3-padded-joint-keys"),
    # Z-Image's 32-token model alignment needs three SP ranks to require an extra pad.
    pytest.param("z_image", 1, 3, ((4, 4), 10), id="zimage-padded-ring3"),
    pytest.param("hunyuan_video15", 2, 2, ((7, 3, 3), 11), id="hunyuan-hybrid-joint-text"),
    pytest.param("qwen_image", 2, 2, ((7, 9), 7), id="qwen-hybrid-joint-text"),
    pytest.param("cosmos3", 2, 2, ((7, 3, 3), 5), id="cosmos3-hybrid-joint-keys"),
]


@pytest.mark.multi_gpu
@pytest.mark.parametrize("model, ulysses, ring, tokens", _RING_CASES)
def test_ring_matches_single_device(accelerator_ranks, model, ulysses, ring, tokens):
    accelerator_ranks(
        _parity_worker,
        world_size=ulysses * ring,
        args=(model, ulysses, ring, tokens),
    )


# Batches of prompts of different lengths, on the flash backends (bfloat16, as they require): the mask reaches
# them as packed keys. Ulysses 3 pads the sequence, and the replicated prompts join as joint keys.
_FLASH_CASES = [
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), (10, 7)), id="hunyuan_video15-batched"),
    pytest.param("hunyuan_video15", 2, ((2, 3, 4), (11, 7)), id="hunyuan_video15-batched-joint-rear"),
    pytest.param("hunyuan_video15", 3, ((2, 3, 4), (12, 8)), id="hunyuan_video15-batched-u3"),
    pytest.param("qwen_image", 2, ((4, 6), (8, 5)), id="qwen_image-batched"),
    pytest.param("qwen_image", 2, ((4, 6), (7, 3)), id="qwen_image-batched-joint-front"),
    pytest.param("qwen_image", 3, ((5, 5), (9, 4)), id="qwen_image-batched-padded-image-u3"),
    pytest.param("qwen_image", 3, ((5, 5), (7, 4)), id="qwen_image-batched-joint-front-padded-image-u3"),
    pytest.param("z_image", 2, ((4, 4), (10, 40)), id="z_image-batched"),
    pytest.param("z_image", 3, ((4, 4), (10, 40)), id="z_image-batched-padded-caption-u3"),
    pytest.param("z_image", 3, ((4, 4), (10, 70)), id="z_image-batched-padded-unified-u3"),
]


@pytest.mark.multi_gpu
@pytest.mark.parametrize("backend", ["FLASH", "FLASH_3", "FLASH_4"])
@pytest.mark.parametrize("model, ulysses, tokens", _FLASH_CASES)
def test_batched_prompts_on_a_flash_backend_match_single_device(accelerator_ranks, backend, model, ulysses, tokens):
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    unavailable = registry.get(AttentionBackendType[backend]).unavailable()
    if unavailable is not None:
        pytest.skip(f"{backend}: {unavailable}")
    accelerator_ranks(
        _parity_worker,
        world_size=ulysses,
        args=(model, ulysses, 1, tokens, backend, torch.bfloat16),
    )
