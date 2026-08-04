"""Which DistVAE adapter, if any, can shard a given diffusers VAE, and how to size it.

DistVAE shards a VAE by rebuilding it out of sharded convolutions, norms and upsampling, so an
adapter only fits a decoder or encoder assembled from the blocks it was written against. Which
adapter fits which VAE class, and the numbers an adapter has to be told about the VAE, are knowledge
about those two libraries and nothing else. Keeping it here means a runner model adds support by
declaring it rather than by carrying its own copy of the wiring.
"""

import importlib
from typing import Optional, Tuple

import torch.nn as nn

DECODER_MODULE = "distvae.modules.adapters.vae.decoder_adapters"
ENCODER_MODULE = "distvae.modules.adapters.vae.encoder_adapters"
# Adapters by name rather than import, so an installed DistVAE predating one of them fails naming
# the adapter it lacks instead of on importing this module.
TWO_D = "DecoderAdapter"
WAN = "WanDecoderAdapter"
QWEN_IMAGE = "QwenImageDecoderAdapter"
HUNYUAN_VIDEO = "HunyuanVideoDecoderAdapter"
HUNYUAN_VIDEO_15 = "HunyuanVideo15DecoderAdapter"
LTX2_VIDEO = "LTX2VideoDecoderAdapter"
WAN_ENCODER = "WanEncoderAdapter"

# Each family's decoder is recognised by the classes its up blocks and mid block are built from,
# named here rather than imported: a VAE that arrived after the installed diffusers leaves its
# entry empty and matches nothing, instead of failing this module's import for every other VAE.
_FAMILIES = (
    (WAN, "autoencoder_kl_wan", ("WanUpBlock", "WanResidualUpBlock"), ("WanMidBlock",)),
    (QWEN_IMAGE, "autoencoder_kl_qwenimage", ("QwenImageUpBlock",), ("QwenImageMidBlock",)),
    (
        HUNYUAN_VIDEO,
        "autoencoder_kl_hunyuan_video",
        ("HunyuanVideoUpBlock3D",),
        ("HunyuanVideoMidBlock3D",),
    ),
    (
        HUNYUAN_VIDEO_15,
        "autoencoder_kl_hunyuanvideo15",
        ("HunyuanVideo15UpBlock3D",),
        ("HunyuanVideo15MidBlock",),
    ),
    (LTX2_VIDEO, "autoencoder_kl_ltx2", ("LTX2VideoUpBlock3d",), ("LTX2VideoMidBlock3d",)),
)


def _blocks(module: str, names: Tuple[str, ...]) -> Tuple[type, ...]:
    """Those of these block classes the installed diffusers has"""
    try:
        found = importlib.import_module(f"diffusers.models.autoencoders.{module}")
    except ImportError:
        return ()
    return tuple(
        block
        for block in (getattr(found, name, None) for name in names)
        if isinstance(block, type)
    )


def decoder_adapter_name(vae) -> Optional[str]:
    """The DistVAE adapter that fits this VAE's decoder, None when none does"""
    # The adapters assert this themselves, from inside a half-built replacement decoder. Asking
    # first keeps an unsupported VAE from reaching that point, and lets a model be told it is
    # unsupported rather than shown an assertion from a library it did not name.
    decoder = getattr(vae, "decoder", None)
    up_blocks = tuple(getattr(decoder, "up_blocks", None) or ())
    if not up_blocks:
        return None

    from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D

    if all(isinstance(block, UpDecoderBlock2D) for block in up_blocks) and isinstance(
        getattr(decoder, "conv_norm_out", None), nn.GroupNorm
    ):
        return TWO_D

    mid_block = getattr(decoder, "mid_block", None)
    for name, module, up_names, mid_names in _FAMILIES:
        up_types = _blocks(module, up_names)
        mid_types = _blocks(module, mid_names)
        # The mid block is checked too because these families fork one another closely enough
        # that up blocks alone would not tell two of them apart.
        if up_types and mid_types and all(isinstance(b, up_types) for b in up_blocks):
            if isinstance(mid_block, mid_types):
                return None if name == LTX2_VIDEO and _injects_noise(decoder) else name
    return None


def _injects_noise(decoder) -> bool:
    """Whether an LTX-2 decoder adds noise inside its residual blocks"""
    # DistVAE cannot shard one that does: each rank would draw noise for its own rows, and the
    # ranks together would not reconstruct what one rank draws. It refuses from inside a decoder
    # it has already half-replaced, so this asks first. No released LTX-2 checkpoint turns it on.
    return any(
        getattr(block, "per_channel_scale1", None) is not None
        or getattr(block, "per_channel_scale2", None) is not None
        for block in decoder.modules()
    )


def _patch_size(vae) -> Optional[int]:
    """The VAE's own patching factor, where it patches on top of its conv stack"""
    # A single factor is Wan's spelling and the only one either adapter can act on. Flux 2 spells
    # the pixel unshuffle at its boundary `(2, 2)`, which is not that and is not something an
    # adapter takes, so anything other than one number reads as no patching.
    patch_size = getattr(vae.config, "patch_size", None)
    return patch_size if isinstance(patch_size, int) and patch_size > 1 else None


def encoder_scale_factor(vae) -> int:
    """The encoder's own spatial downsampling, which is what the encoder adapter shards by"""
    # A VAE that patches folds that factor into its spatial ratio, and the adapter needs the conv
    # stack's share of it alone: Cosmos 3's 16 is 8 from the encoder and 2 from patching.
    factor = getattr(vae.config, "scale_factor_spatial", None) or 8
    patch_size = _patch_size(vae)
    return factor // patch_size if patch_size else factor


def _adapter(module: str, name: str, vae) -> type:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"The installed DistVAE does not provide {name}, which this VAE "
            f"({type(vae).__name__}) needs. Try installing the latest DistVAE from "
            f"https://github.com/xdit-project/DistVAE."
        ) from e


def parallelize_decoder(vae, vae_group) -> str:
    """Replace this VAE's decoder with a sharded one, returning the adapter that did it"""
    name = decoder_adapter_name(vae)
    if name is None:
        raise ValueError(
            f"--use_parallel_vae cannot shard this VAE ({type(vae).__name__}): DistVAE has no "
            f"adapter for its decoder blocks. Use --vae_tile_size to lower VAE decode memory "
            f"instead."
        )
    decoder = _adapter(DECODER_MODULE, name, vae)(vae.decoder, vae_group=vae_group)
    patch_size = _patch_size(vae)
    # The adapter crops its output by the ratio it upsamples, and its patchify assumes no patching
    # because Wan does none. A VAE that patches upsamples by that much again.
    if patch_size and hasattr(decoder, "patchify"):
        decoder.patchify.scale_factor = patch_size
    vae.decoder = decoder.to(vae.device)
    return name


def parallelize_encoder(vae, vae_group) -> str:
    """Replace this VAE's encoder with a sharded one, returning the adapter that did it"""
    # DistVAE ships one encoder adapter, written for Wan's blocks, so the VAEs it can encode with
    # are the ones it can decode with.
    if decoder_adapter_name(vae) != WAN:
        raise ValueError(
            f"Parallel VAE encoding is not available for this VAE ({type(vae).__name__}): DistVAE "
            f"only has {WAN_ENCODER}."
        )
    adapter = _adapter(ENCODER_MODULE, WAN_ENCODER, vae)
    vae.encoder = adapter(
        vae.encoder, vae_group=vae_group, vae_scale_factor=encoder_scale_factor(vae)
    ).to(vae.device)
    return WAN_ENCODER
