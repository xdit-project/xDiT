"""What the installed diffusers can tile or slice, and how wide a window it can decode.

Everything here is knowledge about diffusers VAEs: which tiling attributes a class carries, how
they relate, and which releases have them. Nothing reads xDiT config or touches a pipeline, so
the policy around these numbers lives with the runner instead.
"""

import functools
import math
from typing import Callable, List, NamedTuple, Optional, Tuple

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


SATURATING_LATENT_AREA = 1024
"""The latent area at which one tile already gives the device enough to do

Below this a decoder call is bound by what it costs to make rather than by the arithmetic in it -
kernel launches, and a grid too small to occupy the device - and stacking tiles into the call
spreads that over all of them. Above it the call is bound by the arithmetic, which batching cannot
make cheaper and in practice makes dearer, since the wider shapes pick worse convolution kernels.

Measured on gfx1201 (see the values in the note below): decoding tiles together rather than one at
a time costs 2.58x per tile at a latent area of 16384, 1.69x at 4096, 1.13x at 1024, then pays
0.50x at 256, 0.22x at 64 and 0.09x at 16. The turn is between 1024 and 256 and it is sharp.

Where the device saturates is a property of the device, so this is the one number here that is
worth re-deriving on new hardware; `distvae_bench --tile-shape-costs` prints the curve above in a
few minutes. A larger device turns later, so this value only leaves a win unclaimed there rather
than causing a loss; a smaller one turns earlier, and can lose the ~15% seen just above the turn.
"""


def tile_batch_budget(default_area: Optional[int], tile_area: Optional[int]) -> Optional[int]:
    """The latent area one decoder call should carry, or None to give it a tile and no more

    Stacking tiles into one call buys one thing: it pays what a call costs to make once instead of
    once per tile. That is worth having only while a tile is too small to keep the device busy on
    its own, which is what `SATURATING_LATENT_AREA` marks. A tile at or above it is decoded alone,
    as upstream does and as the VAE was built to be asked - and since a grid's edge tiles are
    clipped by the latent bounds, this is also what stops them batching where the full-size ones
    beside them would not, which would leave two ranks holding the same area making very
    different calls.

    Below the turn, halve --vae_tile_size and this halves: the budget is the geometric mean of the
    two areas, which is the product of their two edges, so it is linear in the edge the caller
    asked for. Activation memory follows the area a call carries, so a budget held at the VAE's
    own window would hand back every byte a narrower window was set to save, while batching a
    fixed number of tiles would give a narrow window back the per-call cost that makes it slow.
    Scaling with the edge moves both ways at once: at a 128px window it takes about nine tenths of
    the available speed-up for a seventh of the memory a full window's batch would have cost.
    """
    if not default_area or not tile_area:
        return None
    if tile_area >= SATURATING_LATENT_AREA:
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


class _StrideLoop(NamedTuple):
    """Where the two stride-walked tiling loops differ from one another"""

    patches: bool  # decodes into a pixel unshuffle, and unpatchifies the assembled sample
    clamps: bool  # holds the assembled sample in [-1, 1]
    first_chunk: bool  # tells the decoder which frame starts the tile


# The VAEs whose stride-walked loop is reimplemented below, by name, because no attribute says
# which loop body a class has. Hunyuan, LTX-2 and CogVideoX carry the same stride attributes and
# the same frame cache and walk them differently, tiling over frames as well, which this does not.
_STRIDE_LOOPS = {
    "AutoencoderKLWan": _StrideLoop(patches=True, clamps=True, first_chunk=True),
    "AutoencoderKLQwenImage": _StrideLoop(patches=False, clamps=False, first_chunk=False),
}


def tiles_by_stored_stride(vae) -> bool:
    """Whether this VAE tiles with the frame-cached loop `strided_tiled_decode` reimplements"""
    if type(vae).__name__ not in _STRIDE_LOOPS:
        return False
    # The class is the loop, but the pieces it walks are still checked, so that a VAE refactored
    # out from under this fails the question rather than the decode.
    return all(
        isinstance(getattr(vae, attr, None), int)
        for attr in (
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
            "spatial_compression_ratio",
        )
    ) and all(
        callable(getattr(vae, attr, None))
        for attr in ("blend_v", "blend_h", "clear_cache", "post_quant_conv", "decoder")
    )


def supports_tile_parallel(vae) -> bool:
    """Whether this VAE's tiling loop is one of the ones reimplemented here

    Deciding which rank makes which decoder call means owning the loop that makes them, so this
    is what a caller asks before planning to decode a VAE's tiles apart from one another.
    """
    return tiles_by_overlap_factor(vae) or tiles_by_stored_stride(vae)


def tiled_decode_for(
    vae,
    budget_elems: int,
    dispatch: Optional[Callable] = None,
    assemble: Optional[Callable] = None,
) -> Optional[Callable]:
    """The tiled_decode to install on this VAE, None where its loop is not one reimplemented here"""
    batched = batched_tiled_decode(vae, budget_elems, dispatch, assemble)
    if batched is not None:
        return batched
    # The stride-walked loop is reimplemented for one reason, which is to hand its tiles round;
    # left to decode them all here it would only be diffusers' own loop with a second author.
    if dispatch is None and assemble is None:
        return None
    return strided_tiled_decode(vae, dispatch, assemble)


def _latent_areas(down, across, window, bounds) -> List[int]:
    """The latent area each tile of the grid covers, in the order the loop walks

    What a tile costs to decode follows the latent it is cut from, and the tiles on the last row
    and the last column are cut short by the bounds. Which of them are short is the same on every
    rank, being read off the grid rather than off a decoded tile.
    """
    deep, wide = window if isinstance(window, tuple) else (window, window)
    return [
        (min(top + deep, bounds[0]) - top) * (min(left + wide, bounds[1]) - left)
        for top in down
        for left in across
    ]


def batched_tiled_decode(
    vae,
    budget_elems: int,
    dispatch: Optional[Callable] = None,
    assemble: Optional[Callable] = None,
) -> Optional[Callable]:
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

    `dispatch` decides who makes those calls, and defaults to this rank making all of them in
    order. `vae_tile_parallel` supplies one that deals them out to a group instead, which is the
    other way to spend the independence between tiles and the one that removes the per-tile cost
    above rather than dividing it.

    `assemble` goes further and divides the blending too, by giving each rank a run of
    neighbouring tiles to decode and stitch by itself. Where it declines - too few tiles to give
    every rank one, or tiles too small to blend against a neighbour's edge alone - the decode
    falls back to `dispatch`, which divides the decoder calls and leaves the blending everywhere.
    """
    if not tiles_by_overlap_factor(vae):
        return None

    from diffusers.models.autoencoders.vae import DecoderOutput

    from xfuser.core.utils import vae_tile_parallel

    # Some classes hold the flag and a None conv, others only the conv; both spellings mean the
    # same thing, and a class carrying neither has no post-quant step.
    use_post_quant_conv = getattr(getattr(vae, "config", None), "use_post_quant_conv", None)
    if use_post_quant_conv is None:
        use_post_quant_conv = getattr(vae, "post_quant_conv", None) is not None

    def tiled_decode(z, return_dict: bool = True):
        overlap_size = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
        blend_extent = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
        row_limit = vae.tile_sample_min_size - blend_extent

        down = range(0, z.shape[2], overlap_size)
        across = range(0, z.shape[3], overlap_size)

        def latent_at(i, j):
            tile = z[
                :,
                :,
                down[i] : down[i] + vae.tile_latent_min_size,
                across[j] : across[j] + vae.tile_latent_min_size,
            ]
            return vae.post_quant_conv(tile) if use_post_quant_conv else tile

        def decode_with(share):
            def decode(where):
                tiles = {at: latent_at(*at) for at in where}

                by_shape = {}
                for at, tile in tiles.items():
                    by_shape.setdefault(tuple(tile.shape), []).append(at)

                batches, calls = [], []
                for shape, group in by_shape.items():
                    # Budget by area rather than by tile count. Activation memory follows the
                    # area being decoded, so one count would batch the widest tiles into an
                    # allocation the VAE was never sized for while barely helping the narrow
                    # ones. Edge tiles are clipped by the latent bounds and so group larger than
                    # the full-shape ones for the same area.
                    area = max(1, shape[0] * shape[-2] * shape[-1])
                    per_call = max(1, budget_elems // area) if budget_elems > 0 else 1
                    for start in range(0, len(group), per_call):
                        batch = group[start : start + per_call]
                        # One tile is handed over as it stands, so a budget that fits one tile
                        # leaves the decoder the same tensor upstream would have given it.
                        batched = (
                            tiles[batch[0]]
                            if len(batch) == 1
                            else torch.cat([tiles[at] for at in batch], dim=0)
                        )
                        batches.append(batch)
                        # Every call is built before any is made, so that a dispatcher can see
                        # them all and hand them round. Each holds a latent tile, which is the
                        # small side of the decode; what they return is not held any longer than
                        # it was before.
                        calls.append(functools.partial(vae.decoder, batched))

                made = {}
                for batch, out in zip(batches, share(calls)):
                    stride = out.shape[0] // len(batch)
                    for n, at in enumerate(batch):
                        made[at] = out[n * stride : (n + 1) * stride]
                return made

            return decode

        blend = vae_tile_parallel.Blend(
            down=vae.blend_v,
            across=vae.blend_h,
            deep_down=blend_extent,
            deep_across=blend_extent,
            crop=lambda tile: tile[:, :, :row_limit, :row_limit],
            tile_down=vae.tile_sample_min_size,
            tile_across=vae.tile_sample_min_size,
        )
        dec = None
        if assemble is not None:
            # A run decodes its own tiles, so the calls stay here rather than going round again.
            dec = assemble(
                len(down),
                len(across),
                decode_with(vae_tile_parallel.in_order),
                blend,
                _latent_areas(down, across, vae.tile_latent_min_size, z.shape[2:]),
            )
        if dec is None:
            share = dispatch if dispatch is not None else vae_tile_parallel.in_order
            dec = vae_tile_parallel.assemble_here(
                len(down), len(across), decode_with(share), blend
            )

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    return tiled_decode


def strided_tiled_decode(
    vae, dispatch: Optional[Callable] = None, assemble: Optional[Callable] = None
) -> Optional[Callable]:
    """A tiled_decode for the video VAEs that walk a stride they store, None where it can't

    Upstream's loop, with the tiles built as calls rather than made where they are built, so that
    `dispatch` can hand them round a group. A tile here is a frame loop threading the VAE's own
    feature cache, which is cleared at the start of each one, so a tile is independent of every
    other tile in the way the frames inside it are not.

    Batching, which the overlap-fraction family gets from `batched_tiled_decode`, is not offered:
    that exists to spread one round of collectives over several tiles, and a decode that hands
    whole tiles out has no round to spread.
    """
    if not tiles_by_stored_stride(vae):
        return None

    from diffusers.models.autoencoders.vae import DecoderOutput

    from xfuser.core.utils import vae_tile_parallel

    loop = _STRIDE_LOOPS[type(vae).__name__]
    patch_size = getattr(vae.config, "patch_size", None) if loop.patches else None

    def tiled_decode(z, return_dict: bool = True):
        _, _, num_frames, height, width = z.shape
        ratio = vae.spatial_compression_ratio
        sample_height = height * ratio
        sample_width = width * ratio
        latent_min_height = vae.tile_sample_min_height // ratio
        latent_min_width = vae.tile_sample_min_width // ratio
        latent_stride_height = vae.tile_sample_stride_height // ratio
        latent_stride_width = vae.tile_sample_stride_width // ratio
        sample_stride_height = vae.tile_sample_stride_height
        sample_stride_width = vae.tile_sample_stride_width
        if patch_size is not None:
            sample_height //= patch_size
            sample_width //= patch_size
            sample_stride_height //= patch_size
            sample_stride_width //= patch_size
            blend_height = vae.tile_sample_min_height // patch_size - sample_stride_height
            blend_width = vae.tile_sample_min_width // patch_size - sample_stride_width
        else:
            blend_height = vae.tile_sample_min_height - sample_stride_height
            blend_width = vae.tile_sample_min_width - sample_stride_width

        down = range(0, height, latent_stride_height)
        across = range(0, width, latent_stride_width)

        def tile_at(i, j):
            def decode():
                # The cache is per tile and threaded through the frames of one, which is why the
                # frames cannot be handed round but the tiles can.
                vae.clear_cache()
                frames = []
                for k in range(num_frames):
                    vae._conv_idx = [0]
                    tile = z[
                        :,
                        :,
                        k : k + 1,
                        down[i] : down[i] + latent_min_height,
                        across[j] : across[j] + latent_min_width,
                    ]
                    tile = vae.post_quant_conv(tile)
                    extra = {"first_chunk": k == 0} if loop.first_chunk else {}
                    frames.append(
                        vae.decoder(
                            tile, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **extra
                        )
                    )
                return torch.cat(frames, dim=2)

            return decode

        def decode_with(share):
            def decode(where):
                made = share([tile_at(*at) for at in where])
                vae.clear_cache()
                return dict(zip(where, made))

            return decode

        blend = vae_tile_parallel.Blend(
            down=vae.blend_v,
            across=vae.blend_h,
            deep_down=blend_height,
            deep_across=blend_width,
            crop=lambda tile: tile[:, :, :, :sample_stride_height, :sample_stride_width],
            tile_down=(
                vae.tile_sample_min_height // patch_size
                if patch_size is not None
                else vae.tile_sample_min_height
            ),
            tile_across=(
                vae.tile_sample_min_width // patch_size
                if patch_size is not None
                else vae.tile_sample_min_width
            ),
        )
        dec = None
        if assemble is not None:
            dec = assemble(
                len(down),
                len(across),
                decode_with(vae_tile_parallel.in_order),
                blend,
                _latent_areas(
                    down,
                    across,
                    (latent_min_height, latent_min_width),
                    (height, width),
                ),
            )
        if dec is None:
            share = dispatch if dispatch is not None else vae_tile_parallel.in_order
            dec = vae_tile_parallel.assemble_here(
                len(down), len(across), decode_with(share), blend
            )
        dec = dec[:, :, :, :sample_height, :sample_width]

        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            dec = unpatchify(dec, patch_size=patch_size)
        if loop.clamps:
            dec = torch.clamp(dec, min=-1.0, max=1.0)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    return tiled_decode
