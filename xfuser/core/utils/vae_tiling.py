"""What the installed diffusers can tile or slice, and how wide a window it can decode.

Everything here is knowledge about diffusers VAEs: which tiling attributes a class carries, how
they relate, and which releases have them. Nothing reads xDiT config or touches a pipeline, so
the policy around these numbers lives with the runner instead.
"""

import math
from typing import Callable, Optional, Tuple

import diffusers
import torch

# The tiling window as diffusers spells it, across the shapes its VAEs use: a latent/pixel pair
# (AutoencoderKL and friends), a pixel window plus a stride (Wan, Qwen-Image, the video VAEs), and
# either of those keyed by height and width. Frame tiling is left out on purpose, being unrelated
# to a spatial tile edge.
PIXEL_ATTRS = (
    "tile_sample_min_size",
    "tile_sample_min_height",
    "tile_sample_min_width",
)
LATENT_ATTRS = (
    "tile_latent_min_size",
    "tile_latent_min_height",
    "tile_latent_min_width",
)
STRIDE_ATTRS = ("tile_sample_stride_height", "tile_sample_stride_width")
SCALED_ATTRS = LATENT_ATTRS + STRIDE_ATTRS
OVERLAP_ATTRS = (
    "tile_overlap_factor",
    "tile_overlap_factor_height",
    "tile_overlap_factor_width",
)
# Which overlap fraction governs which latent window. A VAE that carries one unkeyed fraction
# applies it to both axes.
OVERLAP_AXES = {
    "tile_latent_min_height": "tile_overlap_factor_height",
    "tile_latent_min_width": "tile_overlap_factor_width",
    "tile_latent_min_size": "tile_overlap_factor",
}


def require_vae_support(vae, feature: str, flag: str) -> None:
    """Raise unless the installed diffusers really implements `feature` for this VAE"""
    # Diffusers hands every autoencoder the enable_tiling and enable_slicing methods through a
    # shared mixin, implemented or not, so their presence proves nothing. The state flag the mixin
    # itself checks does. Both features also arrived class by class over several releases, Wan's in
    # 0.34, one past the floor setup.py asks for.
    if not hasattr(vae, f"use_{feature}"):
        raise ValueError(
            f"{flag} is not supported by this VAE ({type(vae).__name__}) in the installed "
            f"diffusers {diffusers.__version__}."
        )


def is_tile_padding_error(error: BaseException) -> bool:
    """Whether a decode failure is the padding error a too-narrow tile window causes"""
    # Torch raises this from a pad deep inside the decoder, where a tile arrives thinner than the
    # convolution's own padding: "Padding size should be less than the corresponding input
    # dimension, but got: padding (1, 1) at dimension 4 of input [1, 8, 3, 4, 1]". Text is all
    # there is to key on, and a rewording upstream only costs the hint, since anything unmatched
    # reaches the caller as the decoder wrote it.
    return "padding size should be less than" in str(error).lower()


def tile_window(vae) -> Optional[int]:
    """The VAE's pixel-space tile edge, None if one number cannot describe it"""
    windows = [
        value
        for attr in PIXEL_ATTRS
        if isinstance(value := getattr(vae, attr, None), int) and value > 0
    ]
    if not windows:
        return None
    # A VAE that sizes height and width apart, as CogVideoX does at 240x360, has no single edge to
    # set: moving both to one number would leave the latent window on one axis describing a
    # different region than the pixel window above it.
    if len(set(windows)) > 1:
        return None
    return windows[0]


def _tile_defaults(vae) -> dict:
    """Every tiling attribute the VAE carries, as the reference to rescale from"""
    defaults = {}
    for attr in PIXEL_ATTRS + SCALED_ATTRS + OVERLAP_ATTRS:
        value = getattr(vae, attr, None)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            defaults[attr] = value
    return defaults


def spatial_ratio(vae) -> Optional[int]:
    """Pixels per latent pixel, where the VAE says so; config first, since reading a config key
    off the module is deprecated"""
    for source in (getattr(vae, "config", None), vae):
        ratio = (
            getattr(source, "spatial_compression_ratio", None)
            if source is not None
            else None
        )
        if isinstance(ratio, int) and ratio > 0:
            return ratio
    return None


def _is_whole(value: float) -> bool:
    """Whole within float error, so 30 x (1 - 1/3) counts as 20 and not 20.000000000000004"""
    return abs(value - round(value)) < 1e-9


def tile_plan(vae, pixels: int) -> Optional[dict]:
    """Every tiling attribute rescaled to a `pixels` window, or None if it can't land whole"""
    # One knob, applied by scaling the whole set by the same factor, which keeps the pixel and
    # latent windows describing the same region and keeps each VAE's own tile overlap.
    window = tile_window(vae)
    if window is None:
        return None
    defaults = _tile_defaults(vae)
    plan = {attr: pixels for attr in PIXEL_ATTRS if attr in defaults}
    for attr in SCALED_ATTRS:
        if attr not in defaults:
            continue
        scaled = pixels * defaults[attr] / window
        if scaled < 1 or not _is_whole(scaled):
            return None
        plan[attr] = round(scaled)
    # Decoders that store an overlap fraction rather than a stride derive the stride by truncating
    # latent x (1 - overlap) while cropping tiles on a separately truncated pixel width. Unless
    # that product lands whole the two disagree and the assembled image comes out the wrong size,
    # with nothing downstream to catch it.
    for latent_attr, factor_attr in OVERLAP_AXES.items():
        latent = plan.get(latent_attr)
        factor = defaults.get(factor_attr, defaults.get("tile_overlap_factor"))
        if latent is None or not isinstance(factor, float) or factor >= 1.0:
            continue
        if not _is_whole(latent * (1.0 - factor)):
            return None
    # A stride below one latent pixel divides down to a zero step, which raises out of range()
    # inside diffusers rather than producing anything.
    ratio = spatial_ratio(vae)
    strides = [plan[attr] for attr in STRIDE_ATTRS if attr in plan]
    if ratio is not None and min([pixels] + strides) < ratio:
        return None
    return plan


def apply_tile_plan(vae, plan: dict) -> None:
    """Set a planned window on the VAE"""
    # Newer VAE classes also take these through enable_tiling(), but only some of them, with a
    # different signature each, and the body is a plain assignment either way.
    for attr, value in plan.items():
        setattr(vae, attr, value)


def latent_rows(vae, plan: dict) -> Optional[int]:
    """How many latent rows a planned tile holds, None where the VAE does not say"""
    latents = [plan[attr] for attr in LATENT_ATTRS if attr in plan]
    if latents:
        return min(latents)
    ratio = spatial_ratio(vae)
    pixels = [plan[attr] for attr in PIXEL_ATTRS if attr in plan]
    if ratio is None or not pixels:
        return None
    return min(pixels) // ratio


def snap_tile_window(vae, pixels: int) -> Tuple[Optional[int], Optional[dict]]:
    """The largest workable window at or below `pixels`, and the attributes that set it"""
    for candidate in range(pixels, 0, -1):
        plan = tile_plan(vae, candidate)
        if plan is not None:
            return candidate, plan
    return None, None


def smallest_tile_window(
    vae, floor: int, ceiling: int, min_latent_rows: int = 1
) -> Optional[int]:
    """The first window from `floor` up that works and holds `min_latent_rows` latent rows, so a
    refusal can name a size that would be accepted
    """
    for pixels in range(floor, ceiling + 1):
        plan = tile_plan(vae, pixels)
        if plan is None:
            continue
        rows = latent_rows(vae, plan)
        if rows is None or rows >= min_latent_rows:
            return pixels
    return None


def tile_latent_area(vae) -> Optional[int]:
    """The latent area of the VAE's current tile, None where it has no square latent window"""
    # Read either side of a narrowing to get both areas the batch budget is derived from: before,
    # it is the area the VAE was built to decode in one call; after, the area the caller asked for.
    size = getattr(vae, "tile_latent_min_size", None)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return None
    return size * size


def tile_batch_budget(default_area: Optional[int], tile_area: Optional[int]) -> Optional[int]:
    """The latent area one decoder call should carry, given the VAE's window and the narrowed one

    Halve --vae_tile_size and this halves: the budget is the geometric mean of the two areas,
    which is the product of their two edges, so it is linear in the edge the caller asked for.
    Equivalently, a batched decode at window W costs what an unbatched decode at the geometric
    mean of W and the VAE's own window costs. At the VAE's own window the two are equal, the
    budget is one tile, and a run that asked for nothing decodes exactly as it always did.

    Two costs pull against each other, which is why the budget is neither of the obvious ones.
    Activation memory follows the area a call carries, so holding that area at the VAE's own
    window - the fastest choice - would hand back every byte a narrower window was set to save.
    A round of collectives costs the same whatever the call carries, so batching a fixed number
    of tiles - the thriftiest choice - would give a narrow window back the per-tile collective
    tax that makes it slow. Scaling with the edge moves both ways at once.
    """
    if not default_area or not tile_area:
        return None
    return max(tile_area, math.isqrt(default_area * tile_area))


def tiles_by_overlap_factor(vae) -> bool:
    """Whether this VAE tiles with the loop `batched_tiled_decode` reimplements"""
    # AutoencoderKL and AutoencoderKLFlux2 walk one square latent window at a stride derived from
    # an overlap fraction. Wan, Qwen-Image and the video VAEs walk a stride they store outright,
    # over a loop with different blending and a frame axis, and keep their own tiled_decode.
    if any(getattr(vae, attr, None) for attr in STRIDE_ATTRS):
        return False
    return (
        isinstance(getattr(vae, "tile_latent_min_size", None), int)
        and isinstance(getattr(vae, "tile_sample_min_size", None), int)
        and isinstance(getattr(vae, "tile_overlap_factor", None), float)
        and callable(getattr(vae, "blend_v", None))
        and callable(getattr(vae, "blend_h", None))
    )


def batched_tiled_decode(vae, budget_elems: int) -> Optional[Callable]:
    """A tiled_decode for `vae` that decodes same-shaped tiles in one call, None where it can't

    Upstream decodes one tile per decoder call, and under --use_parallel_vae every one of those
    calls pays a cost that does not shrink with the tile: a Patchify, a halo exchange per
    convolution, a reduction per norm, and a DePatchify to gather the result. Narrowing the window
    to save memory therefore multiplies a fixed cost, which is what makes a small tile size so
    much slower than the VAE's own. Tiles are independent and, away from the right and bottom
    edges, identically shaped, so they can be stacked on the batch dimension and share all of it.

    `budget_elems` caps the latent area one call may carry; `tile_batch_budget` is what the runner
    sizes it with. At 0, or wherever the budget only fits one tile, this decodes a tile at a time
    and does exactly what upstream does.
    """
    if not tiles_by_overlap_factor(vae):
        return None

    from diffusers.models.autoencoders.vae import DecoderOutput

    # Some classes hold the flag and a None conv, others only the conv; both spellings mean the
    # same thing, and a class carrying neither has no post-quant step.
    use_post_quant_conv = getattr(getattr(vae, "config", None), "use_post_quant_conv", None)
    if use_post_quant_conv is None:
        use_post_quant_conv = getattr(vae, "post_quant_conv", None) is not None

    def tiled_decode(z, return_dict: bool = True):
        overlap_size = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
        blend_extent = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
        row_limit = vae.tile_sample_min_size - blend_extent

        tiles = []
        for i in range(0, z.shape[2], overlap_size):
            for j in range(0, z.shape[3], overlap_size):
                tile = z[
                    :,
                    :,
                    i : i + vae.tile_latent_min_size,
                    j : j + vae.tile_latent_min_size,
                ]
                if use_post_quant_conv:
                    tile = vae.post_quant_conv(tile)
                tiles.append(tile)

        by_shape = {}
        for index, tile in enumerate(tiles):
            by_shape.setdefault(tuple(tile.shape), []).append(index)

        decoded = [None] * len(tiles)
        for shape, indices in by_shape.items():
            # Budget by area rather than by tile count. Activation memory follows the area being
            # decoded, so one count would batch the widest tiles into an allocation the VAE was
            # never sized for while barely helping the narrow ones. Edge tiles are clipped by the
            # latent bounds and so group larger than the full-shape ones for the same area.
            area = max(1, shape[0] * shape[-2] * shape[-1])
            per_call = max(1, budget_elems // area) if budget_elems > 0 else 1
            for start in range(0, len(indices), per_call):
                group = indices[start : start + per_call]
                # One tile is handed over as it stands, so a budget that fits one tile leaves the
                # decoder the same tensor upstream would have given it.
                batched = (
                    tiles[group[0]]
                    if len(group) == 1
                    else torch.cat([tiles[k] for k in group], dim=0)
                )
                out = vae.decoder(batched)
                stride = out.shape[0] // len(group)
                for n, k in enumerate(group):
                    decoded[k] = out[n * stride : (n + 1) * stride]

        columns = len(range(0, z.shape[3], overlap_size))
        rows = [decoded[k : k + columns] for k in range(0, len(decoded), columns)]

        # Upstream's assembly, unchanged. blend_v and blend_h write into the tile they are given,
        # so each tile is blended against neighbours that were themselves already blended, and the
        # scan order that produces is part of the result.
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = vae.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = vae.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    return tiled_decode
