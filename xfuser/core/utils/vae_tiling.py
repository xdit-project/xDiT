"""What the installed diffusers can tile or slice, and how wide a window it can decode.

Everything here is knowledge about diffusers VAEs: which tiling attributes a class carries, how
they relate, and which releases have them. Nothing reads xDiT config or touches a pipeline, so
the policy around these numbers lives with the runner instead.
"""

from typing import Optional, Tuple

import diffusers

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
