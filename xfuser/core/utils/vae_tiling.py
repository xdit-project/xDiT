"""What the installed diffusers can tile or slice, and how wide a window it can decode.

Everything here is knowledge about diffusers VAEs: which tiling attributes a class carries, how
they relate, and which releases have them. Nothing reads xDiT config or touches a pipeline, so
the policy around these numbers lives with the runner instead.
"""

import functools
import inspect
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


def latent_rows(vae, plan: Optional[dict] = None) -> Optional[int]:
    """How many latent rows a tile holds, under `plan` or as the VAE stands, None where it
    does not say
    """
    # Without a plan the VAE's own attributes are the plan, which is how a caller asks about a
    # window that no flag set - a VAE tiling at its own default, or one a model turned on at
    # load.
    if plan is None:
        plan = _tile_defaults(vae)
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


NARROWEST_USEFUL_FRACTION = 2
"""How far below its own window a VAE's tile window is worth narrowing: to a half

A guard rather than a recommendation. Narrowing the window trades output fidelity for memory, and
the trade stops paying well before the window runs out: peak memory falls until the tile's
activations no longer dominate the weights and the full-resolution sample, and then it flattens,
while the seams and the time keep growing. Measured on flux2 at 1024x1024 (whose VAE ships a
1024px window), sweeping only the window - peak memory 1017 MB at 512px, 469 MB at 256px, then
544 MB at 128px and 417 MB at 64px, so it has flattened by a quarter of the window; time turns at
the same place, 382.7 ms at 512px against 497.2 ms at 256px and 679.2 ms at 64px; and divergence
from an untiled decode climbs the whole way, 19.6% max at 512px, 42.0% at 256px, 103.5% at 64px.

Half rather than the quarter where the curves actually turn, because this only has to catch
someone reaching well past the point of return. A half is already a 19.6% divergence on the one
VAE swept, which is the most a window should be asked to give up, and the windows anyone picks
deliberately sit above it.

Expressed as a fraction of the VAE's own window rather than a pixel count, because that window is
the tile size the VAE was built around and a half of it means the same thing across VAEs; the
pixel count where flux2 flattens says nothing about a VAE that ships a 512px window. Only flux2
has been swept, so the fraction is the generalisation and it is the part to re-check.
"""


def narrowest_useful_window(vae) -> Optional[int]:
    """The narrowest window worth setting on this VAE, None where it has no single window"""
    window = tile_window(vae)
    if window is None:
        return None
    return max(1, window // NARROWEST_USEFUL_FRACTION)


def overlap_windows(vae) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """The latent and pixel tile windows as (down, across) pairs, None where the VAE has neither

    Two spellings for the same thing. AutoencoderKL and FLUX.2 carry one square edge; HunyuanVideo
    1.5 carries an edge per axis. A square edge is the same number on both axes, so reading both
    into a pair lets one loop walk either.
    """
    square = getattr(vae, "tile_latent_min_size", None)
    if isinstance(square, int):
        pixels = getattr(vae, "tile_sample_min_size", None)
        return ((square, square), (pixels, pixels)) if isinstance(pixels, int) else None
    keyed = [
        getattr(vae, attr, None)
        for attr in (
            "tile_latent_min_height",
            "tile_latent_min_width",
            "tile_sample_min_height",
            "tile_sample_min_width",
        )
    ]
    if not all(isinstance(value, int) for value in keyed):
        return None
    return (keyed[0], keyed[1]), (keyed[2], keyed[3])


def tiles_by_overlap_factor(vae) -> bool:
    """Whether this VAE tiles with the loop `overlap_tiled_decode` reimplements"""
    # AutoencoderKL, AutoencoderKLFlux2 and HunyuanVideo 1.5 walk a latent window at a stride
    # derived from an overlap fraction. Wan, Qwen-Image and the other video VAEs walk a stride
    # they store outright, over a loop with different blending, and keep their own tiled_decode.
    if any(getattr(vae, attr, None) for attr in STRIDE_ATTRS):
        return False
    if overlap_windows(vae) is None:
        return False
    # The ONE unkeyed fraction is what separates this loop from CogVideoX's, which keys the
    # fraction by axis as well as the window and tiles its frames inside this loop rather than
    # above it. Both blends are named because the loop calls them rather than blending itself.
    return (
        isinstance(getattr(vae, "tile_overlap_factor", None), float)
        and callable(getattr(vae, "blend_v", None))
        and callable(getattr(vae, "blend_h", None))
    )


WINDOW_ATTRS_FOR_STRIDE = ("tile_sample_min_height", "tile_sample_min_width")
"""The window each stride in STRIDE_ATTRS steps across, in the same order"""


def tile_overlap(vae) -> Optional[Tuple[float, float]]:
    """How much of each tile repeats its neighbour, as (down, across) fractions of the window

    The two families spell the step between tiles differently: one stores the overlap as a
    fraction and derives the stride, the other stores the stride in pixels and derives the
    overlap. This reads whichever the VAE carries and answers in fractions either way, so a
    caller can ask what a VAE is set to without knowing which family it belongs to. None where
    it carries neither.
    """
    strides = [getattr(vae, attr, None) for attr in STRIDE_ATTRS]
    windows = [getattr(vae, attr, None) for attr in WINDOW_ATTRS_FOR_STRIDE]
    if all(isinstance(value, int) and value > 0 for value in strides + windows):
        down, across = (
            1.0 - stride / window for stride, window in zip(strides, windows)
        )
        return (down, across)
    factor = getattr(vae, "tile_overlap_factor", None)
    if isinstance(factor, float):
        return (factor, factor)
    return None


def _stride_granularity(vae) -> Optional[int]:
    """The multiple a pixel stride must land on for the stride-walked loop to stay self-consistent

    That loop divides the stride it stores twice: by the compression ratio, to step the latent
    grid, and - where the family decodes into a pixel unshuffle - by the patch size, to place the
    crop. Both are integer divisions, so a stride that is not a multiple of each truncates in one
    of them and the grid and the crop stop describing the same region.
    """
    ratio = spatial_ratio(vae)
    if ratio is None:
        return None
    loop = _STRIDE_LOOPS.get(type(vae).__name__)
    patch = getattr(vae.config, "patch_size", None) if loop and loop.patches else None
    if isinstance(patch, int) and patch > 1:
        return math.lcm(ratio, patch)
    return ratio


def _overlap_lands(latent: int, pixel: int, factor: float) -> bool:
    """Whether the overlap-fraction loop's own arithmetic agrees with itself on this axis

    The loop derives the latent step by truncating `latent x (1 - factor)`, and crops each
    decoded tile to `pixel - int(pixel x factor)`. Unless the second is the first in pixels, the
    tiles step by one amount and are cropped by another, and the assembled image comes out a
    different size than the decode was asked for - with nothing downstream to catch it.

    Checked by recomputing what the loop will compute, rather than by reasoning about the
    algebra, because the factor is a float and the two truncations do not have to fall the same
    way on both sides of it.
    """
    stride = int(latent * (1.0 - factor))
    if stride < 1:
        return False
    ratio, remainder = divmod(pixel, latent)
    return remainder == 0 and pixel - int(pixel * factor) == stride * ratio


def tile_overlap_plan(vae, overlap: float) -> Optional[dict]:
    """Every attribute setting the step between tiles, at `overlap`, or None if it cannot land

    The window says how large a tile is; this says how far apart their origins sit. They are two
    levers and not one. At a fixed window a tiled decode covers (window/stride)^2 times the
    latent it was cut from, so the stride is what decides how much of the decode is redundant,
    while the window is what decides how much memory one tile costs. Widening the stride is
    therefore the lever that buys back the time tiling spends, and it costs seams rather than
    memory - the opposite trade to narrowing the window.

    Never steps wider than asked. Where the exact stride would leave one of the loop's integer
    divisions truncating, the step narrows until it lands whole, so what results overlaps by at
    least what was requested.

    Returns attributes rather than setting them, so `apply_tile_plan` stays the one place a
    window or a stride is written, and so a caller can find out whether an overlap is reachable
    without half-applying it.
    """
    if tiles_by_stored_stride(vae):
        step = _stride_granularity(vae)
        if step is None:
            return None
        plan = {}
        for stride_attr, window_attr in zip(STRIDE_ATTRS, WINDOW_ATTRS_FOR_STRIDE):
            window = getattr(vae, window_attr)
            stride = int(window * (1.0 - overlap)) // step * step
            if stride < step:
                return None
            plan[stride_attr] = min(stride, window)
        return plan

    if not tiles_by_overlap_factor(vae):
        return None
    windows = overlap_windows(vae)
    if windows is None:
        return None
    (latent_down, latent_across), (pixel_down, pixel_across) = windows
    axes = ((latent_down, pixel_down), (latent_across, pixel_across))
    # One fraction governs both axes, so a step that lands whole down the rows still has to land
    # whole across the columns; a VAE windowing the two differently rules out fractions that
    # either axis alone would accept. Walked from the requested step downward, which narrows the
    # step and so widens the overlap - the direction that keeps a wrong guess conservative.
    for stride in range(min(int(latent_down * (1.0 - overlap)), latent_down), 0, -1):
        factor = 1.0 - stride / latent_down
        if not 0.0 <= factor < 1.0:
            continue
        if all(_overlap_lands(latent, pixel, factor) for latent, pixel in axes):
            return {
                attr: factor
                for attr in OVERLAP_ATTRS
                if isinstance(getattr(vae, attr, None), float)
            }
    return None


def widest_tile_overlap(vae) -> Optional[float]:
    """The most overlap this VAE can step by, so a refusal can name one that would be accepted

    Reachability is one-sided: less overlap is a wider step, and a wider step is never the one
    that fails, so walking down from a refused overlap finds where it turns. To a hundredth,
    which is finer than this is set by hand.
    """
    for hundredths in range(99, -1, -1):
        overlap = hundredths / 100
        if tile_overlap_plan(vae, overlap) is not None:
            return overlap
    return None


def _returns_decoder_output(vae) -> bool:
    """Whether this class's own tiled_decode hands back a DecoderOutput rather than a tensor

    The replacement is installed over `tiled_decode` and called by the VAE's own `_decode`, so it
    has to hand back what that caller already expects. Most classes take a `return_dict` and wrap;
    HunyuanVideo 1.5 takes no such argument, returns the tensor, and its `_decode` passes that
    straight to `decode` - which would wrap a DecoderOutput inside another one.

    Read off the class rather than the instance, so that installing twice cannot end up reading
    the first install's signature instead of the original.
    """
    own = getattr(type(vae), "tiled_decode", None)
    if own is None:
        return True
    try:
        return "return_dict" in inspect.signature(own).parameters
    except (TypeError, ValueError):
        return True


class _StrideLoop(NamedTuple):
    """Where the stride-walked tiling loops differ from one another"""

    patches: bool  # decodes into a pixel unshuffle, and unpatchifies the assembled sample
    clamps: bool  # holds the assembled sample in [-1, 1]
    first_chunk: bool  # tells the decoder which frame starts the tile
    frame_cache: bool  # decodes a tile frame by frame, threading the VAE's own feature cache
    post_quant: bool  # puts a tile through post_quant_conv before decoding it
    conditioned: bool  # carries a timestep embedding and a causality flag into the decoder


# The VAEs whose stride-walked loop is reimplemented below, by name, because no attribute says
# which loop body a class has. All four walk the same grid and blend it the same way, and differ
# only in what a tile costs to turn into a decoder call.
#
# HunyuanVideo and LTX-2 keep no feature cache, so a tile is one decoder call over all of its
# frames rather than a loop over them. Both also tile their frames a level up, in a temporal loop
# that calls this one per chunk of them, so what is handed round here is the tiles of one chunk;
# LTX-2 ships with that loop off, and HunyuanVideo with it on.
#
# Still out: CogVideoX tiles over frames inside this loop rather than above it, so its tiles are
# not independent of one another the way every family here is. HunyuanVideo 1.5 was listed here
# too until it turned out to belong to the other family - it walks an overlap fraction, not a
# stride, and `overlap_tiled_decode` now covers it.
_STRIDE_LOOPS = {
    "AutoencoderKLWan": _StrideLoop(
        patches=True,
        clamps=True,
        first_chunk=True,
        frame_cache=True,
        post_quant=True,
        conditioned=False,
    ),
    "AutoencoderKLQwenImage": _StrideLoop(
        patches=False,
        clamps=False,
        first_chunk=False,
        frame_cache=True,
        post_quant=True,
        conditioned=False,
    ),
    "AutoencoderKLHunyuanVideo": _StrideLoop(
        patches=False,
        clamps=False,
        first_chunk=False,
        frame_cache=False,
        post_quant=True,
        conditioned=False,
    ),
    "AutoencoderKLLTX2Video": _StrideLoop(
        patches=False,
        clamps=False,
        first_chunk=False,
        frame_cache=False,
        post_quant=False,
        conditioned=True,
    ),
}


def tiles_by_stored_stride(vae) -> bool:
    """Whether this VAE tiles with the stride-walked loop `strided_tiled_decode` reimplements"""
    loop = _STRIDE_LOOPS.get(type(vae).__name__)
    if loop is None:
        return False
    # The class is the loop, but the pieces it walks are still checked, so that a VAE refactored
    # out from under this fails the question rather than the decode.
    parts = ["blend_v", "blend_h", "decoder"]
    if loop.post_quant:
        parts.append("post_quant_conv")
    if loop.frame_cache:
        parts.append("clear_cache")
    return all(
        isinstance(getattr(vae, attr, None), int)
        for attr in (
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
            "spatial_compression_ratio",
        )
    ) and all(callable(getattr(vae, attr, None)) for attr in parts)


def supports_tile_parallel(vae) -> bool:
    """Whether this VAE's tiling loop is one of the ones reimplemented here

    Deciding which rank makes which decoder call means owning the loop that makes them, so this
    is what a caller asks before planning to decode a VAE's tiles apart from one another.
    """
    return tiles_by_overlap_factor(vae) or tiles_by_stored_stride(vae)


def tiled_decode_for(
    vae,
    dispatch: Optional[Callable] = None,
    assemble: Optional[Callable] = None,
) -> Optional[Callable]:
    """The tiled_decode to install on this VAE, None where its loop is not one reimplemented here"""
    overlapping = overlap_tiled_decode(vae, dispatch, assemble)
    if overlapping is not None:
        return overlapping
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


def overlap_tiled_decode(
    vae,
    dispatch: Optional[Callable] = None,
    assemble: Optional[Callable] = None,
) -> Optional[Callable]:
    """A tiled_decode for the overlap-fraction family, None where the VAE is not one of them

    One tile per decoder call, as upstream does. Tiles are independent and, away from the right
    and bottom edges, identically shaped, so they could instead be stacked on the batch dimension
    to pay a call's fixed cost once for several of them - and that was done here until 2026-08-06.
    It was removed rather than tuned: stacking only pays below roughly a thousand latent elements
    a tile, and every window that small is one where peak memory has already flattened, the decode
    has got slower, and the sample has diverged past use. `narrowest_useful_window` keeps the
    window above that instead, which leaves nothing for a batch to win.

    `dispatch` decides who makes those calls, and defaults to this rank making all of them in
    order. `vae_tile_parallel` supplies one that deals them out to a group instead, which is how
    the independence between tiles is actually spent.

    `assemble` goes further and divides the blending too, by giving each rank a run of
    neighbouring tiles to decode and stitch by itself. Where it declines - too few tiles to give
    every rank one, or tiles too small to blend against a neighbour's edge alone - the decode
    falls back to `dispatch`, which divides the decoder calls and leaves the blending everywhere.

    Three classes share this loop and spell it differently. HunyuanVideo 1.5 sizes its window per
    axis rather than as one square edge, carries a frame axis, and hands back a bare tensor where
    the others hand back a DecoderOutput. None of that reaches the loop: the window is read as a
    pair either way, height and width are always the last two dimensions so `...` indexes them
    whatever sits in front, and the return shape is matched to the method being replaced.
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

    def decode_tiles(z):
        (latent_down, latent_across), (pixel_down, pixel_across) = overlap_windows(vae)
        factor = vae.tile_overlap_factor
        stride_down = int(latent_down * (1 - factor))
        stride_across = int(latent_across * (1 - factor))
        blend_down = int(pixel_down * factor)
        blend_across = int(pixel_across * factor)
        limit_down = pixel_down - blend_down
        limit_across = pixel_across - blend_across

        down = range(0, z.shape[-2], stride_down)
        across = range(0, z.shape[-1], stride_across)

        def latent_at(i, j):
            tile = z[
                ...,
                down[i] : down[i] + latent_down,
                across[j] : across[j] + latent_across,
            ]
            return vae.post_quant_conv(tile) if use_post_quant_conv else tile

        def decode_with(share):
            def decode(where):
                # Every call is built before any is made, so that a dispatcher can see them all
                # and hand them round. Each holds a latent tile, which is the small side of the
                # decode; what they return is not held any longer than it was before.
                at_order = list(where)
                calls = [functools.partial(vae.decoder, latent_at(*at)) for at in at_order]
                return dict(zip(at_order, share(calls)))

            return decode

        blend = vae_tile_parallel.Blend(
            down=vae.blend_v,
            across=vae.blend_h,
            deep_down=blend_down,
            deep_across=blend_across,
            crop=lambda tile: tile[..., :limit_down, :limit_across],
            tile_down=pixel_down,
            tile_across=pixel_across,
        )
        if assemble is not None:
            # A run decodes its own tiles, so the calls stay here rather than going round again.
            dec = assemble(
                len(down),
                len(across),
                decode_with(vae_tile_parallel.in_order),
                blend,
                _latent_areas(
                    down, across, (latent_down, latent_across), z.shape[-2:]
                ),
            )
            if dec is not None:
                return dec
        share = dispatch if dispatch is not None else vae_tile_parallel.in_order
        return vae_tile_parallel.assemble_here(
            len(down), len(across), decode_with(share), blend
        )

    def tiled_decode(z, return_dict: bool = True):
        dec = decode_tiles(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def bare_tiled_decode(z):
        return decode_tiles(z)

    return tiled_decode if _returns_decoder_output(vae) else bare_tiled_decode


def strided_tiled_decode(
    vae, dispatch: Optional[Callable] = None, assemble: Optional[Callable] = None
) -> Optional[Callable]:
    """A tiled_decode for the video VAEs that walk a stride they store, None where it can't

    Upstream's loop, with the tiles built as calls rather than made where they are built, so that
    `dispatch` can hand them round a group. Where the family keeps a feature cache a tile is a
    frame loop threading it, cleared at the start of each tile, so a tile is independent of every
    other tile in the way the frames inside it are not; where it keeps none, a tile is one call.

    A tile per call, as in the other family: what made stacking them look worthwhile there did
    not survive being measured, and `narrowest_useful_window` now keeps the window out of the
    only range where it would have paid.
    """
    if not tiles_by_stored_stride(vae):
        return None

    from diffusers.models.autoencoders.vae import DecoderOutput

    from xfuser.core.utils import vae_tile_parallel

    loop = _STRIDE_LOOPS[type(vae).__name__]
    patch_size = getattr(vae.config, "patch_size", None) if loop.patches else None

    # `temb` and `causal` are LTX-2's, which conditions its decoder on them and passes them
    # through its own tiled_decode to reach it. The families that do not take them never send
    # them, so they sit at the default and this stays one signature for all four loops.
    def tiled_decode(z, temb=None, causal=None, return_dict: bool = True):
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
            def cut(frames=slice(None)):
                return z[
                    :,
                    :,
                    frames,
                    down[i] : down[i] + latent_min_height,
                    across[j] : across[j] + latent_min_width,
                ]

            def decode_frame_by_frame():
                # The cache is per tile and threaded through the frames of one, which is why the
                # frames cannot be handed round but the tiles can.
                vae.clear_cache()
                frames = []
                for k in range(num_frames):
                    vae._conv_idx = [0]
                    tile = vae.post_quant_conv(cut(slice(k, k + 1)))
                    extra = {"first_chunk": k == 0} if loop.first_chunk else {}
                    frames.append(
                        vae.decoder(
                            tile, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **extra
                        )
                    )
                return torch.cat(frames, dim=2)

            def decode_at_once():
                tile = vae.post_quant_conv(cut()) if loop.post_quant else cut()
                if loop.conditioned:
                    return vae.decoder(tile, temb, causal=causal)
                return vae.decoder(tile)

            return decode_frame_by_frame if loop.frame_cache else decode_at_once

        def decode_with(share):
            def decode(where):
                made = share([tile_at(*at) for at in where])
                if loop.frame_cache:
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
