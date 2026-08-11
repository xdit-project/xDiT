"""VAE discovery, configuration, and runtime orchestration for runner models."""

import functools
from typing import List, Optional, Tuple

import torch

from xfuser.compat import load_distvae_parallel_context, load_distvae_vae
from xfuser.core.distributed.parallel_state import (
    get_vae_parallel_group,
    get_vae_parallel_world_size,
)
from xfuser.core.utils.runner_utils import (
    convert_model_convs_to_channels_last,
    log,
)
from xfuser.envs import PACKAGES_CHECKER, restore_torch_group_norm_for_distvae


vae_parallel, vae_tile_parallel, vae_tiling = load_distvae_vae()
ParallelContext = load_distvae_parallel_context()

packages_info = PACKAGES_CHECKER.get_packages_info()


def validate_vae_config(config, capabilities, settings, package_info=None) -> None:
    """Validate VAE-specific runner configuration without runner state."""
    available = packages_info if package_info is None else package_info
    if config.use_parallel_vae:
        if not available.get("has_distvae", False):
            raise ValueError(
                "DistVAE is not installed. Please install it before using parallel VAE."
            )
        if restore_torch_group_norm_for_distvae():
            log(
                "AITER GroupNorm cannot be sharded. Reverting to torch GroupNorm so that "
                "--use_parallel_vae can recognise the norms it has to replace."
            )

    height = getattr(config, "vae_tile_size_height", None)
    width = getattr(config, "vae_tile_size_width", None)
    height_asked = height is not None
    width_asked = width is not None
    if height_asked != width_asked:
        raise ValueError(
            "--vae_tile_size_height and --vae_tile_size_width must be provided together."
        )
    for value, flag in (
        (height, "--vae_tile_size_height"),
        (width, "--vae_tile_size_width"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{flag} must be positive, got {value}.")
    if height_asked and not capabilities.enable_tiling:
        raise ValueError(
            "--vae_tile_size_height and --vae_tile_size_width decode the VAE in tiles, "
            f"which model {settings.model_name} does not support."
        )

    overlap_height = getattr(config, "vae_tile_overlap_height", None)
    overlap_width = getattr(config, "vae_tile_overlap_width", None)
    overlap_height_asked = overlap_height is not None
    overlap_width_asked = overlap_width is not None
    if overlap_height_asked != overlap_width_asked:
        raise ValueError(
            "--vae_tile_overlap_height and --vae_tile_overlap_width must be provided "
            "together."
        )
    for value, flag in (
        (overlap_height, "--vae_tile_overlap_height"),
        (overlap_width, "--vae_tile_overlap_width"),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{flag} must be non-negative, got {value}.")
    if overlap_height_asked and not capabilities.enable_tiling:
        raise ValueError(
            "--vae_tile_overlap_height and --vae_tile_overlap_width set the exact "
            "output-pixel overlap of a tiled VAE decode, "
            f"which model {settings.model_name} does not support."
        )


class VAEManager:
    """Owns all VAE-specific policy used by runner-model lifecycles."""

    def __init__(self, config, capabilities, settings) -> None:
        self.config = config
        self.capabilities = capabilities
        self.settings = settings

    def decoding_vaes(self, pipe, second_pipe=None) -> List:
        """Return every unique VAE decoded by the supplied pipeline stages."""
        vaes = []
        for candidate in (pipe, second_pipe):
            vae = getattr(candidate, "vae", None)
            if vae is not None and not any(vae is seen for seen in vaes):
                vaes.append(vae)
        return vaes

    def enable_options(self, vaes) -> None:
        """Apply slicing, tiling, exact plans, and decode wrappers."""
        tiling_flag = self._tiling_flag()
        for vae in vaes:
            if self.config.enable_slicing:
                vae_tiling.require_vae_support(
                    vae, "slicing", "--enable_slicing"
                )
                log(f"Enabling VAE slicing on {type(vae).__name__}...")
                vae.enable_slicing()

            applied_shape = None
            if self._tiles(vae):
                native_shape = vae_tiling.tile_shape(vae)
                if tiling_flag is not None:
                    vae_tiling.require_vae_support(vae, "tiling", tiling_flag)
                    log(f"Enabling VAE tiling on {type(vae).__name__}...")
                    vae.enable_tiling()
                applied_shape = self._apply_vae_tile_shape(vae)
                self._apply_vae_tile_overlap(vae)
                self._check_tiles_against_parallel_vae(vae, native_shape)
                self._install_vae_tiled_decode(vae)
            self._install_vae_decode_guard(vae, applied_shape)

    def _tiling_flag(self) -> Optional[str]:
        """Return the first flag asking this run to tile its VAE decode."""
        for asked, flag in (
            (self.config.enable_tiling, "--enable_tiling"),
            (
                getattr(self.config, "vae_tile_size_height", None) is not None,
                "--vae_tile_size_height",
            ),
            (
                getattr(self.config, "vae_tile_size_width", None) is not None,
                "--vae_tile_size_width",
            ),
            (
                getattr(self.config, "vae_tile_overlap_height", None) is not None,
                "--vae_tile_overlap_height",
            ),
            (
                getattr(self.config, "vae_tile_overlap_width", None) is not None,
                "--vae_tile_overlap_width",
            ),
        ):
            if asked:
                return flag
        return None

    def _tiles(self, vae) -> bool:
        """Return whether this VAE's decode will be cut into tiles."""
        return self._tiling_flag() is not None or getattr(vae, "use_tiling", False)

    def setup_parallel_vae(self, vaes) -> None:
        """Shard VAE decode, and capability-selected encode, across the VAE group."""
        coordinator = get_vae_parallel_group()
        vae_group = coordinator.device_group
        tile_context = ParallelContext(
            group=vae_group,
            rank=coordinator.rank_in_group,
            world_size=coordinator.world_size,
            patch_dim=-2,
            global_ranks=tuple(coordinator.ranks),
        )
        log(
            f"VAE parallel group: world_size={coordinator.world_size}, "
            f"rank={coordinator.rank_in_group}",
            debug=True,
        )
        for vae in vaes:
            if self._tiles(vae) and vae_tiling.supports_tile_parallel(vae):
                vae_tile_parallel.mark(vae, tile_context)
                log(
                    "Parallel VAE will deal whole tiles out to the group on "
                    f"{type(vae).__name__}, rather than shard the rows within each tile."
                )
            else:
                adapter = vae_parallel.parallelize_decoder(vae, vae_group)
                log(
                    f"Parallel VAE decoder enabled on {type(vae).__name__} via {adapter}."
                )
            if self.capabilities.use_parallel_vae_encoder:
                adapter = vae_parallel.parallelize_encoder(vae, vae_group)
                log(
                    f"Parallel VAE encoder enabled on {type(vae).__name__} via {adapter}."
                )

    def convert_to_channels_last(self, vaes) -> None:
        """Convert every supplied decoding VAE to channels-last exactly once."""
        for vae in vaes:
            self._convert_one_vae_to_channels_last(vae)

    def _convert_one_vae_to_channels_last(self, vae) -> None:
        if getattr(vae, "_xfuser_decode_channels_last", False):
            return
        convert_model_convs_to_channels_last(vae)

        original_decode = vae.decode
        memory_format = (
            torch.channels_last
            if self.settings.model_output_type == "image"
            else torch.channels_last_3d
        )

        @functools.wraps(original_decode)
        def decode_wrapper(*args, **kwargs):
            if args:
                args = list(args)
                args[0] = args[0].to(memory_format=memory_format)
                args = tuple(args)
            elif "z" in kwargs:
                kwargs["z"] = kwargs["z"].to(memory_format=memory_format)
            return original_decode(*args, **kwargs)

        vae.decode = decode_wrapper
        vae._xfuser_decode_channels_last = True

    def _requested_vae_tile_shape(self) -> Optional[Tuple[int, int]]:
        height = getattr(self.config, "vae_tile_size_height", None)
        width = getattr(self.config, "vae_tile_size_width", None)
        if height is None or width is None:
            return None
        return height, width

    def _requested_vae_tile_overlap(self) -> Optional[Tuple[int, int]]:
        height = getattr(self.config, "vae_tile_overlap_height", None)
        width = getattr(self.config, "vae_tile_overlap_width", None)
        if height is None or width is None:
            return None
        return height, width

    def _apply_vae_tile_shape(self, vae) -> Optional[Tuple[int, int]]:
        """Apply and return the exact requested shape, or retain the native window."""
        shape = self._requested_vae_tile_shape()
        if shape is None:
            return None
        height, width = shape
        plan = vae_tiling.tile_shape_plan(vae, height, width)
        if plan is None:
            native = vae_tiling.tile_shape(vae)
            native_shown = (
                f" Its native window is {native[0]}x{native[1]}."
                if native is not None
                else ""
            )
            raise ValueError(
                f"--vae_tile_size_height {height} with --vae_tile_size_width {width} "
                f"is not a shape this VAE ({type(vae).__name__}) can tile exactly."
                f"{native_shown} Choose dimensions compatible with the VAE's latent "
                "scale and tile stride."
            )
        vae_tiling.apply_tile_plan(vae, plan)
        log(
            f"VAE tile window set to {height}x{width}px "
            f"({', '.join(f'{a}={v}' for a, v in sorted(plan.items()))})"
        )
        return shape

    def _apply_vae_tile_overlap(self, vae) -> None:
        """Apply exact per-axis output-pixel tile overlap when requested."""
        requested = self._requested_vae_tile_overlap()
        if requested is None:
            return
        overlap_height, overlap_width = requested
        plan = vae_tiling.tile_overlap_plan(
            vae,
            overlap_height,
            overlap_width,
            sample_shape=(self.config.height, self.config.width),
        )
        if plan is None:
            shape = vae_tiling.tile_shape(vae)
            shape_shown = (
                f"{shape[0]}x{shape[1]} pixels"
                if shape is not None
                else "an unknown pixel shape"
            )
            raise ValueError(
                f"The requested VAE tile overlap of {overlap_height}x{overlap_width} pixels "
                f"is not exact for this VAE ({type(vae).__name__}) at its current tile shape "
                f"of {shape_shown}. Height and width are output pixels; use 0 for an inactive "
                "strip axis."
            )
        vae_tiling.apply_tile_plan(vae, plan)
        landed = vae_tiling.tile_overlap(vae)
        shown = (
            f"{landed[0]}x{landed[1]}px"
            if landed is not None
            else f"{overlap_height}x{overlap_width}px"
        )
        log(
            f"VAE tile overlap set to {shown} "
            f"({', '.join(f'{a}={v:g}' for a, v in sorted(plan.items()))})"
        )

    def _check_tiles_against_parallel_vae(
        self, vae, native_shape: Optional[Tuple[int, int]]
    ) -> None:
        """Refuse a tile holding fewer latent rows than row-sharding ranks."""
        if not (
            self.config.use_parallel_vae and self.capabilities.use_parallel_vae
        ):
            return
        if vae_tile_parallel.context_of(vae) is not None:
            return
        ranks = get_vae_parallel_world_size()
        rows = vae_tiling.latent_rows(vae)
        if ranks < 2 or rows is None or rows >= ranks:
            return
        shape = vae_tiling.tile_shape(vae)
        smallest = self._minimum_square_vae_tile_shape(
            vae, shape, native_shape, ranks
        )
        shown = (
            f"{shape[0]}x{shape[1]}px" if shape is not None else "unknown-size"
        )
        native_shown = (
            f"{native_shape[0]}x{native_shape[1]}px"
            if native_shape is not None
            else "unknown"
        )
        raise ValueError(
            f"A {shown} VAE tile window leaves {rows} latent rows for the {ranks} ranks "
            f"--use_parallel_vae splits each tile across"
            + (
                f"; the smallest square window with a row per rank is "
                f"--vae_tile_size_height {smallest} --vae_tile_size_width {smallest}."
                if smallest
                else f", and no shape up to this VAE's native {native_shown} window gives them one "
                f"each. Decode without tiling, increase --vae_tile_size_height and "
                f"--vae_tile_size_width, or use fewer VAE ranks."
            )
        )

    @staticmethod
    def _minimum_square_vae_tile_shape(
        vae,
        shape: Optional[Tuple[int, int]],
        native_shape: Optional[Tuple[int, int]],
        min_latent_rows: int,
    ) -> Optional[int]:
        """Find the first exact square plan between current and native shapes."""
        if shape is None or native_shape is None:
            return None
        floor = max(shape)
        ceiling = min(native_shape)
        if floor > ceiling:
            return None
        for candidate in range(floor, ceiling + 1):
            plan = vae_tiling.tile_shape_plan(vae, candidate, candidate)
            if plan is None:
                continue
            rows = vae_tiling.latent_rows(vae, plan)
            if rows is not None and rows >= min_latent_rows:
                return candidate
        return None

    def _install_vae_tiled_decode(self, vae) -> None:
        """Install local or group-dispatched tiled decode when DistVAE provides one."""
        context = vae_tile_parallel.context_of(vae)
        if context is None:
            if (
                self._requested_vae_tile_shape() is None
                and self._requested_vae_tile_overlap() is None
            ):
                return
            installed = vae_tiling.tiled_decode_for(vae)
        else:
            dispatch, assemble = vae_tile_parallel.sharing(context)
            installed = vae_tiling.tiled_decode_for(vae, dispatch, assemble)
        if installed is None:
            return
        vae.tiled_decode = installed
        if context is None:
            log(
                f"VAE tiled decode on {type(vae).__name__}: using DistVAE's local "
                "overlap loop."
            )
            return
        log(
            f"VAE tiled decode on {type(vae).__name__}: a tile per call, divided across "
            f"{context.world_size} ranks, a run of neighbouring tiles each "
            "to decode and blend where the grid has the tiles to spare them."
        )

    def _install_vae_decode_guard(
        self, vae, tile_shape: Optional[Tuple[int, int]] = None
    ) -> None:
        """Add actionable OOM and narrow-tile diagnostics to a VAE decode."""
        vae._xfuser_guarded_tile_shape = tile_shape
        if getattr(vae, "_xfuser_decode_guarded", False):
            return
        original_decode = vae.decode

        @functools.wraps(original_decode)
        def decode_guard(*args, **kwargs):
            try:
                return original_decode(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(
                    f"{self._vae_decode_oom_hint(vae)}\n{e}"
                ) from e
            except RuntimeError as e:
                shape = getattr(vae, "_xfuser_guarded_tile_shape", None)
                if shape is None or not vae_tiling.is_tile_padding_error(e):
                    raise
                shown = f"{shape[0]}x{shape[1]}"
                raise RuntimeError(
                    f"VAE tiled decode failed at the {shown}px tile window set by "
                    "--vae_tile_size_height and --vae_tile_size_width: at this output size the "
                    "window leaves a tile too thin for the decoder to pad. A larger window can "
                    "fail where a smaller one works, so try another exact height and width, or "
                    f"drop both flags to decode at this VAE's native window.\n{e}"
                ) from e

        vae.decode = decode_guard
        vae._xfuser_decode_guarded = True

    def _vae_decode_oom_hint(self, vae) -> str:
        if not getattr(vae, "use_tiling", False):
            if self.capabilities.enable_tiling:
                return (
                    "VAE decode ran out of memory with tiling disabled. Re-run with "
                    "--enable_tiling to decode in tiles."
                )
            return (
                f"VAE decode ran out of memory, and model {self.settings.model_name} does not "
                "support VAE tiling."
            )

        shape = vae_tiling.tile_shape(vae)
        if shape is None:
            return (
                "VAE tiled decode ran out of memory. This model's VAE "
                f"({type(vae).__name__}) does not expose a pixel-space tile shape to resize."
            )
        height, width = shape
        return (
            f"VAE tiled decode ran out of memory at a {height}x{width}px tile window. Shrink it "
            "by choosing another exact pair with --vae_tile_size_height and "
            "--vae_tile_size_width, then re-run and compare peak VRAM."
        )
