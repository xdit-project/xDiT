"""What one run decides to tell its VAE about tiling, and the order the telling has to happen in.

vae_tiling knows what a diffusers VAE *can* be told; this decides what to tell it, and refuses the
combinations that do not work. The two are separate because the order here is the fragile part and
deserves to be readable in one place: the overlap is a fraction of the window, so sizing the window
rescales the stride and the overlap has to be set after; the check that a tile still holds a latent
row per rank can only read numbers once both are settled; and the tiled decode can only be
installed once the window whose tiles are being dealt out is final.

Nothing here holds a pipeline or a runner. Everything a decision needs arrives in a TilingRequest,
so the whole of this can be exercised against a bare VAE.
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from xfuser.core.utils import vae_tile_parallel, vae_tiling
from xfuser.core.utils.runner_utils import log


@dataclass(frozen=True)
class TilingRequest:
    """One run's asks about its VAE decode, in the terms the steps below decide with

    Assembled by the runner from its config and capabilities so that the decisions can be read,
    and tested, without one.
    """

    model_name: str
    #: --vae_tile_size, the pixel edge of one tile: what a tile costs to hold.
    tile_size: Optional[int] = None
    #: --vae_tile_overlap, the fraction tiles share: how much of the decode is spent twice.
    tile_overlap: Optional[float] = None
    #: --enable_tiling / --enable_slicing.
    tiling: bool = False
    slicing: bool = False
    #: Whether the model declares it can tile at all, for advice given after a failure.
    model_tiles: bool = True
    #: Whether this run splits each tile's latent rows across a group, and across how many.
    splits_tiles: bool = False
    ranks: int = 1

    @property
    def flag(self) -> Optional[str]:
        """The flag asking this run to tile its VAE decode, None where none does"""
        # Either knob is a request to tile, so either turns tiling on by itself. Sizing the
        # window would otherwise mean passing --enable_tiling as well, which tiles every stage
        # to reach the one that ran out of memory.
        for asked, flag in (
            (self.tiling, "--enable_tiling"),
            (self.tile_size is not None, "--vae_tile_size"),
            (self.tile_overlap is not None, "--vae_tile_overlap"),
        ):
            if asked:
                return flag
        return None


def tiles(vae, request: TilingRequest) -> bool:
    """Whether this VAE's decode will be cut into tiles"""
    # Off the VAE as well as off the request, because a model can arrive already tiling: LTX
    # 2.3 turns it on for its stage-2 VAE at load, where nothing on the command line says so.
    #
    # Asked in the two places that have to agree - where the tiled decode is installed, and
    # where the parallel VAE chooses between dealing whole tiles out and sharding the rows
    # inside each one - so it is answered once. Disagreeing is not a slower decode but a hung
    # one: reading the config alone left a tiling VAE on the sharding path, where a tile
    # holding fewer latent rows than the group leaves the surplus ranks indexing off the end
    # of a split the rest are waiting in.
    return request.flag is not None or getattr(vae, "use_tiling", False)


def configure(vae, request: TilingRequest) -> None:
    """Settle one VAE's slicing and tiling, in the one order these steps can be taken in"""
    if request.slicing:
        vae_tiling.require_vae_support(vae, "slicing", "--enable_slicing")
        log(f"Enabling VAE slicing on {type(vae).__name__}...")
        vae.enable_slicing()

    window = None
    if tiles(vae, request):
        # Read before anything narrows it, since a refusal below has to be able to say what
        # this VAE's own window was and search the sizes between the two.
        own_window = vae_tiling.tile_window(vae)
        if request.flag is not None:
            vae_tiling.require_vae_support(vae, "tiling", request.flag)
            log(f"Enabling VAE tiling on {type(vae).__name__}...")
            vae.enable_tiling()
        window = apply_window(vae, request)
        # After the window, never before: the overlap is a fraction of the window, and sizing
        # the window rescales the stride to keep whatever overlap the VAE had.
        apply_overlap(vae, request)
        check_against_split(vae, request, own_window)
        install_tiled_decode(vae)
    # Installed either way, so a decode that OOMs with tiling off still says so.
    install_decode_guard(vae, request, window)


def apply_window(vae, request: TilingRequest) -> Optional[int]:
    """The window set on this VAE, None where the VAE keeps its own"""
    # The default window tracks the VAE's training resolution and never shrinks for an
    # above-training-res decode, so a single tile can outgrow free VRAM. This shrinks it.
    requested = request.tile_size
    if requested is None:
        return None
    window = vae_tiling.tile_window(vae)
    if window is None:
        raise ValueError(
            f"Model {request.model_name} does not support --vae_tile_size: its VAE "
            f"({type(vae).__name__}) has no single pixel-space tile window that one size sets."
        )
    if requested > window:
        log(f"--vae_tile_size {requested} is larger than this VAE's {window}px tile window, "
            f"which would raise peak memory rather than lower it; leaving it at {window}px.")
        return None
    # Narrowing buys memory only until the tile stops being what the decode is holding, and
    # past that it costs seams and time for nothing. Clamped rather than refused, because a
    # window this narrow is someone reaching for memory that is no longer there to save.
    narrowest = vae_tiling.narrowest_useful_window(vae)
    if narrowest is not None and requested < narrowest:
        log(f"--vae_tile_size {requested} is below half of this VAE's {window}px tile "
            f"window, where peak memory has stopped falling and only the seams and the time "
            f"keep growing; using {narrowest}px instead.")
        requested = narrowest
    pixels, plan = vae_tiling.snap_tile_window(vae, requested)
    if plan is None:
        smallest = vae_tiling.smallest_tile_window(vae, requested, window)
        raise ValueError(
            f"--vae_tile_size {requested} is not a window this VAE ({type(vae).__name__}) "
            f"can tile with" + (f"; the smallest that works is {smallest}px."
                                if smallest else
                                f", and neither is any size up to its own {window}px window.")
        )
    if pixels != requested:
        log(f"--vae_tile_size {requested} is not a window this VAE can tile with exactly; "
            f"using the next one down at {pixels}px.")
    vae_tiling.apply_tile_plan(vae, plan)
    log(f"VAE tile window set to {pixels}px "
        f"({', '.join(f'{a}={v}' for a, v in sorted(plan.items()))})")
    return pixels


def apply_overlap(vae, request: TilingRequest) -> None:
    """Step this VAE's tile grid at the requested overlap, where one was requested"""
    # The window and the overlap divide different things. The window decides what one tile
    # costs to hold; the overlap decides how much of the decode is spent twice, since tiles
    # overlapping by f cover 1/(1-f)^2 times the latent they were cut from. A VAE ships the
    # overlap its own training resolution wanted, and at a large decode that redundancy is
    # what tiling costs in time - so this is the knob that gives it back.
    requested = request.tile_overlap
    if requested is None:
        return
    if vae_tiling.tile_overlap(vae) is None:
        raise ValueError(
            f"Model {request.model_name} does not support --vae_tile_overlap: its VAE "
            f"({type(vae).__name__}) does not say how far apart it steps its tiles, so there "
            f"is nothing here to set."
        )
    plan = vae_tiling.tile_overlap_plan(vae, requested)
    if plan is None:
        widest = vae_tiling.widest_tile_overlap(vae)
        raise ValueError(
            f"--vae_tile_overlap {requested} is not a step this VAE "
            f"({type(vae).__name__}) can take" +
            (f"; the most overlap it can step by is {widest:g}."
             if widest is not None else
             ", and neither is any overlap: this VAE's tiling loop is not one xFuser knows "
             "how to step.")
        )
    vae_tiling.apply_tile_plan(vae, plan)
    # Read back off the VAE rather than off the plan: the two families store different things
    # - one the fraction, one the stride it implies - and only what the VAE ends up holding
    # says what the decode will actually do.
    landed = vae_tiling.tile_overlap(vae)
    shown = overlap_shown(landed)
    if landed is not None and any(abs(f - requested) > 1e-9 for f in landed):
        log(f"--vae_tile_overlap {requested:g} is not a step this VAE can take exactly; "
            f"using the next one up at {shown}.")
    log(f"VAE tile overlap set to {shown} "
        f"({', '.join(f'{a}={v:g}' for a, v in sorted(plan.items()))})")


def overlap_shown(overlap: Optional[Tuple[float, float]]) -> str:
    """An overlap as one number, or as two where the VAE steps its axes differently"""
    if overlap is None:
        return "unknown"
    down, across = overlap
    if abs(down - across) < 1e-9:
        return f"{down:g}"
    return f"{down:g} down, {across:g} across"


def check_against_split(vae, request: TilingRequest, own_window: Optional[int]) -> None:
    """Refuse a tile holding fewer latent rows than the ranks that will split them"""
    # Tiling and sharding divide the same axis: diffusers hands the decoder one tile, and
    # DistVAE then splits that tile's latent rows across the VAE group. Under a row per rank
    # it splits into fewer patches than there are ranks, and the surplus ranks index off the
    # end of the split rather than reporting anything - so this hangs the group rather than
    # failing it, and is worth refusing up front.
    #
    # Read off the VAE once the window and the stride are settled, rather than off the plan
    # a flag applied, because what is dangerous is the composition and not the flag: a VAE
    # tiling at its own default window reaches it with no plan to check.
    if not request.splits_tiles:
        return
    # Unless the tiles are what the ranks divide, in which case nothing divides a tile and a
    # window narrower than the group is no longer anybody's problem.
    if vae_tile_parallel.group_of(vae) is not None:
        return
    ranks = request.ranks
    rows = vae_tiling.latent_rows(vae)
    if ranks < 2 or rows is None or rows >= ranks:
        return
    window = vae_tiling.tile_window(vae)
    smallest = (
        vae_tiling.smallest_tile_window(vae, window, own_window, min_latent_rows=ranks)
        if window is not None and own_window is not None
        else None
    )
    raise ValueError(
        f"A {window}px VAE tile window leaves {rows} latent rows for the {ranks} ranks "
        f"--use_parallel_vae splits each tile across" +
        (f"; the smallest window with a row per rank is --vae_tile_size {smallest}."
         if smallest else
         f", and no window up to this VAE's own {own_window}px gives them one each. Decode "
         f"without tiling, or across fewer VAE ranks.")
    )


def install_tiled_decode(vae) -> None:
    """Decode a tiled VAE's tiles apart across a group, a tile to a call"""
    # There is one thing worth doing with the independence between tiles, which is to give
    # them to different ranks. Stacking several into one call was the other, and it only ever
    # paid at window sizes apply_window now refuses, so with no group there is nothing to
    # install and upstream's own loop stands.
    group = vae_tile_parallel.group_of(vae)
    if group is None:
        return
    dispatch, assemble = vae_tile_parallel.sharing(group)
    installed = vae_tiling.tiled_decode_for(vae, dispatch, assemble)
    if installed is None:
        return
    vae.tiled_decode = installed
    log(f"VAE tiled decode on {type(vae).__name__}: a tile per call, divided across "
        f"{torch.distributed.get_world_size(group)} ranks, a run of neighbouring tiles each "
        f"to decode and blend where the grid has the tiles to spare them.")


def install_decode_guard(vae, request: TilingRequest, window: Optional[int] = None) -> None:
    """Point a failed VAE decode at the knob that fixes it. Success path untouched."""
    # The window is recorded on the VAE rather than closed over, so that installing again
    # over an already-guarded decode is a matter of updating what the guard reports rather
    # than nesting a second guard holding the older window. A runner's initialize() can run
    # more than once in a process, and a stack of guards would name whichever window was set
    # first.
    vae._xfuser_guarded_tile_window = window
    if getattr(vae, "_xfuser_decode_guarded", False):
        return
    original_decode = vae.decode

    @functools.wraps(original_decode)
    def decode_guard(*args, **kwargs):
        try:
            return original_decode(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(f"{oom_hint(vae, request)}\n{e}") from e
        except RuntimeError as e:
            # Two things have to hold before the window gets the blame: this run narrowed it,
            # and the decoder failed the way a narrow window makes it fail. A dtype or device
            # error is failing for reasons of its own, as is a VAE still at its own window.
            guarded = getattr(vae, "_xfuser_guarded_tile_window", None)
            if guarded is None or not vae_tiling.is_tile_padding_error(e):
                raise
            # Whether a window leaves a tile the decoder cannot pad depends on the output size,
            # so this cannot be caught when the window is set; name the window, being the part
            # the caller can change, and keep the decoder's own words underneath.
            raise RuntimeError(
                f"VAE tiled decode failed at the {guarded}px tile window set by "
                f"--vae_tile_size: at this output size the window leaves a tile too thin for "
                f"the decoder to pad. A larger window can fail where a smaller one works, so "
                f"try another --vae_tile_size, or drop it to decode at this VAE's own "
                f"window.\n{e}"
            ) from e

    vae.decode = decode_guard
    vae._xfuser_decode_guarded = True


def oom_hint(vae, request: TilingRequest) -> str:
    """What to change after a VAE decode has run out of memory"""
    # Read from the VAE, since a model can arrive with tiling on and no flag set.
    if not getattr(vae, "use_tiling", False):
        if request.model_tiles:
            return ("VAE decode ran out of memory with tiling disabled. Re-run with "
                    "--enable_tiling to decode in tiles.")
        return (f"VAE decode ran out of memory, and model {request.model_name} does not "
                "support VAE tiling.")

    window = vae_tiling.tile_window(vae)
    if window is None:
        return ("VAE tiled decode ran out of memory. This model's VAE "
                f"({type(vae).__name__}) has no single tile window to size, so "
                "--vae_tile_size does not apply.")
    # Halve the window through the same snap the override uses, so the guard can never name a
    # size that the next run would turn around and refuse.
    target, _ = vae_tiling.snap_tile_window(vae, max(window // 2, 1))
    if target is None:
        return (f"VAE tiled decode ran out of memory at a {window}px tile window, the smallest "
                "this VAE can tile with.")
    # One step, and say what to look at, rather than leaving halving to look like a free knob
    # that can be turned again. It cannot: the VAE stops being what peaks after a step or two,
    # and from there a narrower window costs image quality and buys no memory at all.
    return (f"VAE tiled decode ran out of memory at a {window}px tile window. Shrink it with "
            f"--vae_tile_size {target}, then re-run. Take one step at a time and compare peak "
            f"VRAM: if it barely moves, the VAE is no longer what peaks and a narrower window "
            f"will cost image quality without saving memory.")
