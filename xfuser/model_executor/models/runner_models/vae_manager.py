"""VAE discovery, configuration, and runtime orchestration for runner models."""

import functools
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from distvae.vae import ParallelContext, VAERowSplitError, latent_rows, mark
from distvae.vae import parallel as vae_parallel
from distvae.vae import tile_parallel as vae_tile_parallel
from distvae.vae import tiling as vae_tiling

from xfuser.core.distributed.parallel_state import (
    get_vae_parallel_group,
    get_vae_parallel_world_size,
)
from xfuser.core.utils.runner_utils import (
    convert_model_convs_to_channels_last,
    log,
)
from xfuser.envs import restore_torch_group_norm_for_distvae


def _validate_vae_tile_pair(config, capabilities, settings, kind, minimum, constraint) -> None:
    names = tuple(f"vae_tile_{kind}_{axis}" for axis in ("height", "width"))
    flags = tuple(f"--{name}" for name in names)
    values = tuple(getattr(config, name, None) for name in names)
    asked = tuple(value is not None for value in values)
    joined_flags = " and ".join(flags)

    if asked[0] != asked[1]:
        raise ValueError(f"{joined_flags} must be provided together.")
    for value, flag in zip(values, flags):
        if value is not None and value < minimum:
            raise ValueError(f"{flag} must be {constraint}, got {value}.")
    if not asked[0]:
        return
    if not config.enable_tiling:
        raise ValueError(f"{joined_flags} require --enable_tiling.")
    if not capabilities.enable_tiling:
        raise ValueError(
            f"{joined_flags} configure tiled VAE decoding, which model {settings.model_name} does not support."
        )


def attention_rows(attn, hidden_states: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """``attn(hidden_states).narrow(2, start, length)`` for a diffusers self-attention with
    ``AttnProcessor2_0``, computing only the queries of those rows.

    The group norm, keys and values still need every row; the attention, the query and the
    output projection, which grow with the number of queries, run on ``length`` rows only.
    """
    batch_size, channel, height, width = hidden_states.shape
    residual = hidden_states.narrow(2, start, length)
    tokens = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    if attn.group_norm is not None:
        tokens = attn.group_norm(tokens.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(tokens[:, start * width : (start + length) * width])
    key = attn.to_k(tokens)
    value = attn.to_v(tokens)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
    out = attn.to_out[1](attn.to_out[0](out))
    out = out.transpose(-1, -2).reshape(batch_size, channel, length, width)
    if attn.residual_connection:
        out = out + residual
    return out / attn.rescale_output_factor


class RowSplitAttnProcessor:
    """``AttnProcessor2_0`` for a self-attention that every rank of the VAE group runs on the
    whole image: each rank computes an equal band of query rows, and an all-gather rebuilds the
    output. Calls it cannot split this way (cross-attention, masks, a height that does not
    divide by the group size) run whole.
    """

    def __init__(self, parallel_context: ParallelContext):
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.parallel_context = parallel_context
        self.whole = AttnProcessor2_0()

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs
    ):
        world_size = self.parallel_context.world_size
        splittable = (
            hidden_states.ndim == 4
            and hidden_states.shape[2] % world_size == 0
            and encoder_hidden_states is None
            and attention_mask is None
            and temb is None
            and not args
            and not kwargs
        )
        if not splittable:
            return self.whole(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        rows = hidden_states.shape[2] // world_size
        local = attention_rows(attn, hidden_states, self.parallel_context.rank * rows, rows).contiguous()
        # Gloo requires the gather output concatenated along dim 0, not stacked.
        bands = torch.empty((world_size * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(bands, local, group=self.parallel_context.group)
        return bands.view(world_size, *local.shape).permute(1, 2, 0, 3, 4).reshape(hidden_states.shape)


def _splittable_attention(attn) -> bool:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    return (
        type(attn) is Attention
        and type(attn.processor) is AttnProcessor2_0
        and attn.spatial_norm is None
        and not attn.is_cross_attention
    )


def split_replicated_vae_attention(vae) -> int:
    """Split the mid-block attention that DistVAE's 2D decoder runs whole on every rank.

    ``DecoderAdapter`` shards from the up blocks on; its mid block, attention included, decodes
    the whole latent on every rank. At 2048² that is a 65k-token attention per rank.
    """
    from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

    adapter = getattr(vae, "decoder", None)
    if not isinstance(adapter, DecoderAdapter) or adapter.parallel_context.world_size == 1:
        return 0
    mid_block = getattr(adapter.decoder, "mid_block", None)
    resnets = getattr(mid_block, "resnets", None) or ()
    # Adapted resnets would mean the mid block runs on this rank's rows only.
    if any(not type(resnet).__module__.startswith("diffusers.") for resnet in resnets):
        return 0
    split = 0
    for attn in getattr(mid_block, "attentions", None) or ():
        if attn is not None and _splittable_attention(attn):
            attn.set_processor(RowSplitAttnProcessor(adapter.parallel_context))
            split += 1
    return split


def validate_vae_config(config, capabilities, settings) -> None:
    """Validate VAE-specific runner configuration without runner state."""
    if config.use_parallel_vae:
        if restore_torch_group_norm_for_distvae():
            log(
                "AITER GroupNorm cannot be sharded. Restoring torch GroupNorm so DistVAE can "
                "identify and shard the GroupNorm layers."
            )

    _validate_vae_tile_pair(config, capabilities, settings, "size", 1, "positive")
    _validate_vae_tile_pair(config, capabilities, settings, "overlap", 0, "non-negative")


class VAEManager:
    """Owns all VAE-specific policy used by runner-model lifecycles."""

    def __init__(self, config, capabilities, settings) -> None:
        self.config = config
        self.capabilities = capabilities
        self.settings = settings
        self._overlap_sample_shapes = {}

    def decoding_vaes(self, pipes) -> List:
        """Return every unique VAE decoded by the supplied pipeline stages."""
        vaes = []
        for candidate in pipes:
            vae = getattr(candidate, "vae", None)
            if vae is not None and not any(vae is seen for seen in vaes):
                vaes.append(vae)
        return vaes

    def enable_options(self, vaes) -> None:
        """Apply slicing, tiling, exact plans, and decode wrappers."""
        tiling_flag = self._tiling_flag()
        for vae in vaes:
            if self.config.enable_slicing:
                vae_tiling.require_vae_support(vae, "slicing", "--enable_slicing")
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
                self._check_tiles_against_parallel_vae(vae, native_shape)
                self._install_vae_tiled_decode(vae)
            self._install_vae_decode_guard(vae, applied_shape)

    def prepare_run(self, vaes, input_args) -> None:
        """Apply shape-dependent VAE options for the current invocation."""
        if self._requested_vae_tile_overlap() is None:
            return
        sample_shape = (input_args["height"], input_args["width"])
        for vae in vaes:
            key = id(vae)
            if self._overlap_sample_shapes.get(key) == sample_shape:
                continue
            self._apply_vae_tile_overlap(vae, sample_shape)
            self._overlap_sample_shapes[key] = sample_shape

    def _tiling_flag(self) -> Optional[str]:
        """Return the flag asking this run to tile its VAE decode."""
        return "--enable_tiling" if self.config.enable_tiling else None

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
            f"VAE parallel group: world_size={coordinator.world_size}, rank={coordinator.rank_in_group}",
            debug=True,
        )
        try:
            from diffusers.models import AutoencoderKLMiniMaxH3
        except ImportError:
            AutoencoderKLMiniMaxH3 = None
        try:
            from diffusers.models import AutoencoderKLMiniMaxH3Image
        except ImportError:
            AutoencoderKLMiniMaxH3Image = None

        for vae in vaes:
            if AutoencoderKLMiniMaxH3 is not None and isinstance(vae, AutoencoderKLMiniMaxH3):
                from xfuser.model_executor.models.runner_models.minimax_h3 import (
                    install_minimax_h3_vae_tile_parallel,
                )

                install_minimax_h3_vae_tile_parallel(vae)
                continue
            if AutoencoderKLMiniMaxH3Image is not None and isinstance(vae, AutoencoderKLMiniMaxH3Image):
                from xfuser.model_executor.models.runner_models.minimax_h3_image import (
                    install_minimax_h3_image_tiled_decode,
                )

                install_minimax_h3_image_tiled_decode(vae, coordinator)
                continue
            if self._tiles(vae) and vae_tiling.supports_tile_parallel(vae):
                mark(vae, tile_context)
                log(
                    "Parallel VAE will assign complete tiles of "
                    f"{type(vae).__name__} to ranks instead of sharding rows within each tile."
                )
            else:
                adapter = vae_parallel.parallelize_decoder(vae, vae_group)
                log(f"Parallel VAE decoder enabled on {type(vae).__name__} via {adapter}.")
                split = split_replicated_vae_attention(vae) if self.capabilities.split_parallel_vae_attention else 0
                if split:
                    log(f"Parallel VAE decoder: {split} replicated mid-block attention(s) split by query rows.")
            if self.capabilities.use_parallel_vae_encoder:
                adapter = vae_parallel.parallelize_encoder(vae, vae_group)
                log(f"Parallel VAE encoder enabled on {type(vae).__name__} via {adapter}.")

    def convert_to_channels_last(self, vaes) -> None:
        """Convert every supplied decoding VAE to channels-last exactly once."""
        for vae in vaes:
            self._convert_one_vae_to_channels_last(vae)

    def _convert_one_vae_to_channels_last(self, vae) -> None:
        if getattr(vae, "_xfuser_decode_channels_last", False):
            return
        convert_model_convs_to_channels_last(vae)

        original_decode = vae.decode
        memory_format = torch.channels_last if self.settings.model_output_type == "image" else torch.channels_last_3d

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
        """Apply and return the exact requested shape, or retain the default tile shape."""
        shape = self._requested_vae_tile_shape()
        if shape is None:
            return None
        height, width = shape
        plan = vae_tiling.tile_shape_plan(vae, height, width)
        if plan is None:
            native = vae_tiling.tile_shape(vae)
            native_shown = f" The VAE's default tile shape is {native[0]}x{native[1]}." if native is not None else ""
            raise ValueError(
                f"--vae_tile_size_height {height} with --vae_tile_size_width {width} "
                f"is not a shape this VAE ({type(vae).__name__}) can tile exactly."
                f"{native_shown} Choose dimensions compatible with the VAE's latent "
                "scale and tile stride."
            )
        vae_tiling.apply_tile_plan(vae, plan)
        log(f"VAE tile window set to {height}x{width}px ({', '.join(f'{a}={v}' for a, v in sorted(plan.items()))})")
        return shape

    def _apply_vae_tile_overlap(self, vae, sample_shape: Tuple[int, int]) -> None:
        """Apply exact per-axis output-pixel tile overlap when requested."""
        requested = self._requested_vae_tile_overlap()
        if requested is None:
            return
        overlap_height, overlap_width = requested
        plan = vae_tiling.tile_overlap_plan(
            vae,
            overlap_height,
            overlap_width,
            sample_shape=sample_shape,
        )
        if plan is None:
            shape = vae_tiling.tile_shape(vae)
            shape_shown = f"{shape[0]}x{shape[1]} pixels" if shape is not None else "an unknown pixel shape"
            raise ValueError(
                f"The requested VAE tile overlap of {overlap_height}x{overlap_width} pixels "
                f"is not exact for this VAE ({type(vae).__name__}) at its current tile shape "
                f"of {shape_shown}. Height and width are output pixels; use 0 for an inactive "
                "strip axis."
            )
        vae_tiling.apply_tile_plan(vae, plan)
        landed = vae_tiling.tile_overlap(vae)
        shown = f"{landed[0]}x{landed[1]}px" if landed is not None else f"{overlap_height}x{overlap_width}px"
        log(f"VAE tile overlap set to {shown} ({', '.join(f'{a}={v:g}' for a, v in sorted(plan.items()))})")

    def _check_tiles_against_parallel_vae(self, vae, native_shape: Optional[Tuple[int, int]]) -> None:
        """Refuse a tile holding fewer latent rows than row-sharding ranks."""
        if not (self.config.use_parallel_vae and self.capabilities.use_parallel_vae):
            return
        if vae_tile_parallel.context_of(vae) is not None:
            return
        ranks = get_vae_parallel_world_size()
        rows = latent_rows(vae)
        if ranks < 2 or rows is None or rows >= ranks:
            return
        shape = vae_tiling.tile_shape(vae)
        smallest = self._minimum_vae_tile_shape(vae, shape, native_shape, ranks)
        shown = f"{shape[0]}x{shape[1]}px" if shape is not None else "unknown-size"
        native_shown = f"{native_shape[0]}x{native_shape[1]}px" if native_shape is not None else "unknown"
        raise ValueError(
            f"A {shown} VAE tile contains {rows} latent rows, but --use_parallel_vae uses "
            f"{ranks} ranks and requires at least one latent row per rank"
            + (
                f"; the smallest tile shape that satisfies this requirement is "
                f"--vae_tile_size_height {smallest[0]} "
                f"--vae_tile_size_width {smallest[1]}."
                if smallest
                else f". No tile shape up to the VAE's default {native_shown} shape has enough "
                f"latent rows. Decode without tiling, increase --vae_tile_size_height and "
                "--vae_tile_size_width, or use fewer VAE ranks."
            )
        )

    @staticmethod
    def _minimum_vae_tile_shape(
        vae,
        shape: Optional[Tuple[int, int]],
        native_shape: Optional[Tuple[int, int]],
        min_latent_rows: int,
    ) -> Optional[Tuple[int, int]]:
        """Find the first exact plan with enough rows, preserving tile width."""
        if shape is None or native_shape is None:
            return None
        height, width = shape
        native_height = native_shape[0]
        if height > native_height:
            return None
        for candidate in range(height, native_height + 1):
            plan = vae_tiling.tile_shape_plan(vae, candidate, width)
            if plan is None:
                continue
            rows = latent_rows(vae, plan)
            if rows is not None and rows >= min_latent_rows:
                return candidate, width
        return None

    def _install_vae_tiled_decode(self, vae) -> None:
        """Install local or group-dispatched tiled decode when DistVAE provides one."""
        context = vae_tile_parallel.context_of(vae)
        if context is None:
            if self._requested_vae_tile_shape() is None and self._requested_vae_tile_overlap() is None:
                return
            installed = vae_tiling.tiled_decode_for(vae)
        else:
            dispatch, assemble = vae_tile_parallel.sharing(context)
            installed = vae_tiling.tiled_decode_for(vae, dispatch, assemble)
        if installed is None:
            return
        vae.tiled_decode = installed
        if context is None:
            log(f"VAE tiled decode on {type(vae).__name__}: using DistVAE's local overlap loop.")
            return
        log(
            f"VAE tiled decode on {type(vae).__name__}: assigning contiguous groups of "
            f"neighboring tiles across {context.world_size} ranks and blending the decoded "
            "outputs."
        )

    def _install_vae_decode_guard(self, vae, tile_shape: Optional[Tuple[int, int]] = None) -> None:
        """Add actionable OOM and narrow-tile diagnostics to a VAE decode."""
        vae._xfuser_guarded_tile_shape = tile_shape
        if getattr(vae, "_xfuser_decode_guarded", False):
            return
        original_decode = vae.decode

        @functools.wraps(original_decode)
        def decode_guard(*args, **kwargs):
            try:
                return original_decode(*args, **kwargs)
            except VAERowSplitError as e:
                raise ValueError(self._vae_row_split_hint(vae, e)) from e
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(f"{self._vae_decode_oom_hint(vae)}\n{e}") from e
            except RuntimeError as e:
                shape = getattr(vae, "_xfuser_guarded_tile_shape", None)
                if shape is None or not vae_tiling.is_tile_padding_error(e):
                    raise
                shown = f"{shape[0]}x{shape[1]}"
                raise RuntimeError(
                    f"VAE tiled decode failed with a {shown}px tile shape. At this output size, "
                    "an edge tile is too small for decoder padding. Try another exact height and "
                    "width, or remove --vae_tile_size_height and --vae_tile_size_width to use the "
                    f"VAE's default tile shape.\n{e}"
                ) from e

        vae.decode = decode_guard
        vae._xfuser_decode_guarded = True

    def _vae_row_split_hint(self, vae, error: VAERowSplitError) -> str:
        message = (
            f"Cannot row-shard {error.rows} latent rows in the VAE decoder: this VAE "
            f"processes rows in groups of {error.factor}, but {error.rows} is not divisible "
            f"by {error.factor}."
        )
        scale = getattr(getattr(vae, "config", None), "scale_factor_spatial", None)
        actions = [
            (
                f"Use an output height divisible by {scale * error.factor}"
                if isinstance(scale, int) and scale > 0
                else f"Use a latent height divisible by {error.factor}"
            )
        ]
        if self.capabilities.enable_tiling and vae_tiling.supports_tile_parallel(vae):
            actions.append("enable --enable_tiling to distribute complete tiles instead")
        actions.append("disable --use_parallel_vae")
        return f"{message} {', '.join(actions[:-1])}, or {actions[-1]}."

    def _vae_decode_oom_hint(self, vae) -> str:
        if not getattr(vae, "use_tiling", False):
            if self.capabilities.enable_tiling:
                return (
                    "VAE decode ran out of memory with tiling disabled. Re-run with --enable_tiling to decode in tiles."
                )
            return f"VAE decode ran out of memory, and model {self.settings.model_name} does not support VAE tiling."

        shape = vae_tiling.tile_shape(vae)
        if shape is None:
            return (
                f"VAE tiled decode ran out of memory. The {type(vae).__name__} VAE used by "
                f"{self.settings.model_name} does not expose an adjustable pixel-space tile shape."
            )
        height, width = shape
        return (
            f"VAE tiled decode ran out of memory with a {height}x{width}px tile shape. Choose "
            "smaller exact values for --vae_tile_size_height and --vae_tile_size_width, then "
            "rerun and compare peak VRAM."
        )
