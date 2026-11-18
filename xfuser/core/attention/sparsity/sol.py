"""Sol-Attn: block-sparse attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.
"""

import inspect
from types import SimpleNamespace
from typing import NamedTuple, Optional

import torch

# gfx950 carries a mode-2 manifest row for every recipe below. Which recipe a device serves is read
# from aiter's manifest, per recipe.
_SUPPORTED_ARCHS = ("gfx950",)


class SolAttnUnsupported(RuntimeError):
    """Raised when the inputs or the environment cannot be served correctly by the Sol-Attn kernel."""


def _keep_bf16(tensor):
    """Pass a BF16 operand through unquantized, in the (tensor, descale) shape the others return.

    The NONE scale mode makes the kernel ignore the descale; the tensor is the placeholder, as in
    aiter's mha_v4.
    """
    return tensor, tensor


def _probe_aiter():
    """aiter's Sol-Attn entry points, or None when this build does not ship them.

    Bound once at import so no import runs inside a traced graph. native_fp8_format is held as a
    function because it queries the device, which import time is too early for.
    """
    try:
        from aiter.ops.mha_v4 import (
            AttentionFormat,
            AttentionPack,
            AttentionScaleMode,
            MHA_V4_SOL_MODE,
            mha_v4_block_tile,
            mha_v4_block_tiles,
            mha_v4_kv_tile_for_q_tile,
            mha_v4_operands,
            mha_v4_packed,
            mha_v4_q_multiplier,
            mha_v4_sol,
            mxfp4_k_view,
            mxfp4_v_view,
            mxfp6_k_view,
            native_fp8_format,
            quantize_fp8,
            quantize_fp8_rotated,
            quantize_int8,
            quantize_mxfp4_k,
            quantize_mxfp4_q,
            quantize_mxfp6_k,
            quantize_mxfp6_q,
            quantize_mxfp8_k,
            quantize_mxfp8_q,
            quantize_v_mxfp4_fp6_p,
            quantize_v_mxfp6_fp6_p,
        )
        from aiter.ops.triton.attention.utils import sol_prepare
    except ImportError:
        return None
    try:
        from aiter.ops.mha_v4 import quantize_v_fp8
    except ImportError:
        # Predates the MXFP6 Q/K, FP8 V row, which is then reported as not served.
        quantize_v_fp8 = None
    return SimpleNamespace(
        sol=mha_v4_sol,
        packed=mha_v4_packed,
        prepare=sol_prepare,
        native_fp8_format=native_fp8_format,
        block_tile=mha_v4_block_tile,
        kv_tile_for_q_tile=mha_v4_kv_tile_for_q_tile,
        block_tiles=mha_v4_block_tiles,
        operands=mha_v4_operands,
        sol_mode=MHA_V4_SOL_MODE,
        q_multiplier=mha_v4_q_multiplier,
        quantize_fp8=quantize_fp8,
        quantize_fp8_rotated=quantize_fp8_rotated,
        quantize_int8=quantize_int8,
        quantize_mxfp8_q=quantize_mxfp8_q,
        quantize_mxfp8_k=quantize_mxfp8_k,
        quantize_mxfp4_q=quantize_mxfp4_q,
        quantize_mxfp4_k=quantize_mxfp4_k,
        quantize_mxfp6_q=quantize_mxfp6_q,
        quantize_mxfp6_k=quantize_mxfp6_k,
        quantize_v_mxfp4_fp6_p=quantize_v_mxfp4_fp6_p,
        quantize_v_mxfp6_fp6_p=quantize_v_mxfp6_fp6_p,
        quantize_v_fp8=quantize_v_fp8,
        mxfp4_k_view=mxfp4_k_view,
        mxfp4_v_view=mxfp4_v_view,
        mxfp6_k_view=mxfp6_k_view,
        quantize_bf16=_keep_bf16,
        fmt=AttentionFormat,
        pack=AttentionPack,
        no_scale=AttentionScaleMode.NONE,
        per_tensor_scale=AttentionScaleMode.F32_PER_TENSOR,
        per_channel_scale=AttentionScaleMode.F32_PER_CHANNEL,
        block_scale=AttentionScaleMode.E8M0_PER_1X32,
    )


def _probe_scale_storage():
    """aiter's padded E8M0 scale allocator, its row tile and the lookahead its readers need."""
    try:
        from aiter.ops.mha_v4_quant import (
            MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
            MHA_V4_KV_TILE_ROWS,
            block_scale_storage,
        )
    except ImportError:
        return None, 0, 0
    return block_scale_storage, MHA_V4_KV_TILE_ROWS, MHA_V4_KV_SCALE_LOOKAHEAD_ROWS


def _probe_ragged_kv():
    """aiter's per-row ragged-KV query, or None on a build that predates it, where every row
    has its key length padded to the tile."""
    try:
        from aiter.ops.mha_v4 import mha_v4_ragged_kv
    except ImportError:
        return None
    return mha_v4_ragged_kv


def _probe_second_order():
    """aiter's Jensen cap and pooled value tilt builders, or Nones on a build without them."""
    try:
        from aiter.ops.triton.attention.sol_tilt import (
            sol_jensen_cap,
            sol_q_abs_max,
            sol_tilt_factors,
            sol_tilt_planes,
            sol_tilt_scaled_params,
            sol_tilt_valu_params,
        )
        from aiter.ops.triton.attention.utils import _e8m0_dequantize
    except ImportError:
        return (None,) * 7
    return (
        sol_q_abs_max,
        sol_jensen_cap,
        sol_tilt_factors,
        sol_tilt_planes,
        sol_tilt_valu_params,
        sol_tilt_scaled_params,
        _e8m0_dequantize,
    )


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None

# Module globals, not _AITER fields: these are functools.cache wrappers, and Dynamo binds one
# reached through an attribute as a method, which fails with "Too many positional arguments".
_default_block_tile = _AITER.block_tile if _AITER is not None else None
_kv_tile_for_q_tile = _AITER.kv_tile_for_q_tile if _AITER is not None else None
_ragged_kv = _probe_ragged_kv() if _AITER is not None else None
_block_scale_storage, _SCALE_TILE_ROWS, _SCALE_LOOKAHEAD_ROWS = (
    _probe_scale_storage() if _AITER is not None else (None, 0, 0)
)
# Module globals for the same reason.
(
    _q_abs_max,
    _jensen_cap,
    _tilt_factors,
    _tilt_planes,
    _tilt_valu_params,
    _tilt_scaled_params,
    _e8m0_dequantize,
) = _probe_second_order() if _AITER is not None else (None,) * 7
# Whether sol_prepare can add forced blocks to its selection (exact KV tokens).
_SOL_PREPARE_FORCES_BLOCKS = _AITER is not None and "force_block_mask" in inspect.signature(_AITER.prepare).parameters


def sol_attn_block_tile(recipe, block_tile=None):
    """The (q_tile, kv_tile) a call routes, pools and dispatches `recipe` at.

    Routing, pooling and dispatch must agree on it, and a mismatch is silently wrong, so all three
    read it from here. block_tile is --sparseattn_block_size, already checked against the recipe's
    row; None takes the recipe's default, which differs between rows.
    """
    if block_tile is not None:
        return tuple(block_tile)
    return _default_block_tile(_recipe_operands(recipe), _AITER.sol_mode)


class _Recipe(NamedTuple):
    """One mode-2 manifest row: its formats, its quantizers, and how its scales survive pooling.

    A per-tensor or per-channel scale does not vary along the pooled sequence axis, so the pooled
    operand reuses it. An E8M0 1x32 scale does, so that operand pools in dequantized space and hands
    the kernel a pooled scale of its own.

    quantize_q takes a multiplier because the MX quantizers fold softmax_scale * log2(e) into Q;
    the per-tensor ones ignore it and the kernel applies the scale itself.
    """

    id: str
    qk_format: str  # attribute name on AttentionFormat, resolved per device
    qk_scale_mode: str  # "none", "per_tensor" or "block"
    quantize_q: str  # attribute name on the _AITER namespace
    quantize_k: str
    quantize_v: str = "quantize_fp8"
    v_format: str = "native"
    v_scale_mode: str = "per_tensor"  # or "none", "per_channel" or "block"
    # Set, with sol_prepare's name for the packing, when that operand's codes are sub-byte and in
    # tile order. They cannot be pooled or routed on, so both use the BF16 source.
    k_packed: Optional[str] = None
    v_packed: Optional[str] = None
    # The aiter builder path the row's pooled value tilt goes through (see _second_order), or None
    # for a row aiter builds without the tilt. aiter's manifest has no column for it; this follows
    # mha_v4_packed's documentation.
    tilt: Optional[str] = None

    @property
    def v_pack(self) -> str:
        """The AttentionPack V is stored in: the FP6-P order for every MX V, else the default."""
        return "V_FOR_FP6_P" if (self.v_packed or "").endswith("_fp6_p") else "DEFAULT"

    @property
    def routes_through_raw(self) -> bool:
        """Whether mha_v4_sol serves this row. MXFP8's addressable block-scaled K needs its scale
        handed to the pooling, which the raw entry point cannot take, so it goes packed."""
        return not (self.qk_scale_mode == "block" and self.k_packed is None)


_RECIPES = {
    r.id: r
    for r in (
        _Recipe(
            "bf16",
            "BF16",
            "none",
            "quantize_bf16",
            "quantize_bf16",
            quantize_v="quantize_bf16",
            v_format="BF16",
            v_scale_mode="none",
            tilt="bf16_planes",
        ),
        _Recipe("bf16fp8", "BF16", "none", "quantize_bf16", "quantize_bf16", tilt="bf16_planes"),
        _Recipe("fp8", "native", "per_tensor", "quantize_fp8_rotated", "quantize_fp8_rotated", tilt="fp8_factors"),
        _Recipe("i8fp8", "INT8", "per_tensor", "quantize_int8", "quantize_int8", tilt="int8_planes"),
        _Recipe("mxfp8", "native", "block", "quantize_mxfp8_q", "quantize_mxfp8_k", tilt="mxfp8_planes"),
        # The MX-V rows store V in the order the kernel's FP6 P operand reads it.
        _Recipe(
            "f8f6",
            "native",
            "per_tensor",
            "quantize_fp8_rotated",
            "quantize_fp8_rotated",
            quantize_v="quantize_v_mxfp6_fp6_p",
            v_format="MXFP6",
            v_scale_mode="block",
            v_packed="mxfp6_fp6_p",
            tilt="fp8_k_mx_v_planes",
        ),
        _Recipe(
            "mxfp6",
            "MXFP6",
            "block",
            "quantize_mxfp6_q",
            "quantize_mxfp6_k",
            quantize_v="quantize_v_fp8",
            v_scale_mode="per_channel",
            k_packed="mxfp6",
        ),
        _Recipe(
            "f6f6",
            "MXFP6",
            "block",
            "quantize_mxfp6_q",
            "quantize_mxfp6_k",
            quantize_v="quantize_v_mxfp6_fp6_p",
            v_format="MXFP6",
            v_scale_mode="block",
            k_packed="mxfp6",
            v_packed="mxfp6_fp6_p",
            tilt="packed_k_planes",
        ),
        _Recipe(
            "f6f4",
            "MXFP6",
            "block",
            "quantize_mxfp6_q",
            "quantize_mxfp6_k",
            quantize_v="quantize_v_mxfp4_fp6_p",
            v_format="MXFP4",
            v_scale_mode="block",
            k_packed="mxfp6",
            v_packed="mxfp4_fp6_p",
            tilt="packed_k_planes",
        ),
        _Recipe(
            "mxfp4",
            "MXFP4",
            "block",
            "quantize_mxfp4_q",
            "quantize_mxfp4_k",
            quantize_v="quantize_v_mxfp4_fp6_p",
            v_format="MXFP4",
            v_scale_mode="block",
            k_packed="mxfp4",
            v_packed="mxfp4_fp6_p",
            tilt="packed_k_planes",
        ),
    )
}

SOL_ATTN_RECIPES = tuple(_RECIPES)


def check_sol_attn_device(device=None):
    """Raise unless this build and device can run Sol-Attn. Defaults to the current CUDA device.

    Checked once at backend setup, not per call; direct callers of sol_attn_bhsd check it themselves.
    """
    if not SOL_ATTN_AVAILABLE:
        raise SolAttnUnsupported("Sol-Attn requires aiter with mha_v4_sol and sol_prepare; please update AITER")
    if device is None:
        if not torch.cuda.is_available():
            raise SolAttnUnsupported("Sol-Attn is a GPU kernel and no CUDA device is available")
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise SolAttnUnsupported(f"Sol-Attn is a GPU kernel, got device {device}")
    arch = _device_arch(device)
    if arch is None:
        reported = torch.cuda.get_device_properties(device).gcnArchName or ""
        raise SolAttnUnsupported(
            f"Sol-Attn ships kernels for {', '.join(_SUPPORTED_ARCHS)}, this device reports '{reported}'"
        )


def _device_arch(device=None):
    """The supported arch this device is, or None. gcnArchName carries a target-feature suffix."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    name = torch.cuda.get_device_properties(device).gcnArchName or ""
    return next((arch for arch in _SUPPORTED_ARCHS if name.startswith(arch)), None)


def check_sol_attn_recipe(recipe_id, device=None, block_tile=None):
    """Raise unless this device has a Sol-Attn manifest row for recipe_id, at block_tile if given.

    block_tile is --sparseattn_block_size, which is valid only where the recipe's own row serves it.
    """
    check_sol_attn_device(device)
    recipe = _resolve_recipe(recipe_id)
    if not _serves_recipe(recipe):
        raise SolAttnUnsupported(
            f"Sol-Attn '{recipe_id}' has no {_device_arch(device)} manifest row; "
            f"this device serves {', '.join(sol_attn_recipes_served()) or 'none'}"
        )
    if block_tile is None:
        return
    q_tile, kv_tile = block_tile
    operands = _recipe_operands(recipe)
    if _kv_tile_for_q_tile(q_tile, operands, _AITER.sol_mode) == kv_tile:
        return
    served = _AITER.block_tiles(operands, _AITER.sol_mode)
    raise SolAttnUnsupported(
        f"--sparseattn_block_size {q_tile}x{kv_tile} has no Sol-Attn kernel with the '{recipe.id}' "
        f"recipe on this GPU; it serves {', '.join(f'{qo}x{kv}' for qo, kv in served) or 'none'}. "
        f"Omit it for the default geometry ({'x'.join(str(part) for part in sol_attn_block_tile(recipe))})."
    )


def sol_attn_recipes_served():
    """The recipe ids this device has a Sol-Attn row for, in declaration order."""
    return [r for r in SOL_ATTN_RECIPES if _serves_recipe(_RECIPES[r])]


def _serves_recipe(recipe):
    """Whether the active device's manifest has a mode-2 row, at any geometry, for `recipe`."""
    if getattr(_AITER, recipe.quantize_v) is None:
        return False
    return bool(_AITER.block_tiles(_recipe_operands(recipe), _AITER.sol_mode))


def _recipe_operands(recipe):
    """The seven manifest values identifying `recipe`'s row, for aiter's geometry queries.

    Resolved per device, since a recipe's format may be "native".
    """
    qk_format, v_format = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale, v_scale = _scale_mode(recipe.qk_scale_mode), _scale_mode(recipe.v_scale_mode)
    return _AITER.operands(
        qk_format, qk_format, v_format, qk_scale, qk_scale, v_scale, getattr(_AITER.pack, recipe.v_pack)
    )


def _takes_ragged_kv(recipe, block_tile):
    """Whether `recipe`'s row at `block_tile` masks a short last KV block itself. A row that
    does not needs K/V padded to a whole number of blocks."""
    if _ragged_kv is None:
        return False
    return _ragged_kv(_recipe_operands(recipe), _AITER.sol_mode, block_tile)


def check_sol_attn_supported(query, key, value, is_causal, ring_world_size=1):
    """Validate the per-call constraints aiter's Sol-Attn contract does not already cover."""
    if not query.dtype == key.dtype == value.dtype == torch.bfloat16:
        raise SolAttnUnsupported(
            f"Sol-Attn's mha_v4 row takes bf16 Q/K/V and returns bf16, got q={query.dtype} "
            f"k={key.dtype} v={value.dtype}. Select another attention backend for other dtypes."
        )
    if is_causal:
        raise SolAttnUnsupported(
            "Sol-Attn has no causal variant: its pooled correction assumes every skipped block is "
            "fully attendable, which a causal mask breaks. Use AITER_FP8 for causal attention."
        )
    if ring_world_size > 1:
        raise SolAttnUnsupported(
            "Sol-Attn does not support ring parallelism: its routing threshold is taken over the "
            "KV blocks one call is shown, so each ring rank would select against its own shard and "
            "the merged result would depend on how the keys were split. Use ulysses_degree for "
            "sequence parallelism instead."
        )


def _resolve_recipe(recipe):
    """The named recipe, or a clear error listing what this build actually ships."""
    try:
        return _RECIPES[recipe]
    except KeyError:
        raise SolAttnUnsupported(
            f"unknown Sol-Attn recipe {recipe!r}; this build has {', '.join(SOL_ATTN_RECIPES)}"
        ) from None


def _format(name):
    """An AttentionFormat by name, with "native" resolved against the active GPU."""
    return _AITER.native_fp8_format() if name == "native" else getattr(_AITER.fmt, name)


def _scale_mode(name):
    return {
        "none": _AITER.no_scale,
        "per_tensor": _AITER.per_tensor_scale,
        "per_channel": _AITER.per_channel_scale,
        "block": _AITER.block_scale,
    }[name]


def _quantize(recipe, query, key, value, softmax_scale):
    """Quantize BSHD bf16 Q/K/V for one recipe, returning (tensor, descale, source) per operand.

    As mha_v4_sol quantizes the same row, except that K is not smoothed: once K's shared component
    passes aiter's smoothing gate, this path's output differs from the raw one. The MX quantizers
    fold softmax_scale * log2(e) into Q; the per-tensor ones leave the scale to the kernel.

    source is the BF16 input, kept only for a packed operand, whose codes pooling and routing
    cannot read.
    """
    multiplier = _AITER.q_multiplier(softmax_scale)

    if recipe.qk_scale_mode == "block":
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query, multiplier)
    else:
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query)

    k, k_descale = getattr(_AITER, recipe.quantize_k)(key)
    # The packers return a flat buffer; the kernel takes the strided view aiter builds over it.
    if recipe.k_packed == "mxfp4":
        k = _AITER.mxfp4_k_view(k, k_descale)
    elif recipe.k_packed == "mxfp6":
        k, k_descale = _AITER.mxfp6_k_view(k, k_descale, *key.shape[:3])

    v, v_descale = getattr(_AITER, recipe.quantize_v)(value)
    if recipe.v_packed == "mxfp4_fp6_p":
        v = _AITER.mxfp4_v_view(v, v_descale, value.shape[1])

    qk_packed = recipe.k_packed is not None
    return (
        (q, q_descale, query if qk_packed else None),
        (k, k_descale, key if qk_packed else None),
        (v, v_descale, value if recipe.v_packed is not None else None),
    )


def sol_attn_routing_for(
    q,
    k,
    v,
    beta,
    recipe=_RECIPES["fp8"],
    block_attn_mask=None,
    block_tile=None,
    force_block_mask=None,
    k_variance=False,
):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them.

    q/k/v are the (tensor, descale, source) triples _quantize returns. block_attn_mask supplies an
    already chosen selection instead of routing from beta; pass exactly one of the two.
    force_block_mask, ending in num_kv_blocks, is OR-ed into the selection either way.
    k_variance also returns each block's K variance, for the Jensen term.
    """
    qk_packed = recipe.k_packed is not None
    # An addressable E8M0-scaled operand's codes are not proportional to its values, so routing
    # and pooling are handed the scale.
    qk_block_scaled = recipe.qk_scale_mode == "block" and not qk_packed
    v_block_scaled = recipe.v_scale_mode == "block" and recipe.v_packed is None
    # A packed recipe routes on the BF16 Q. Routing is invariant to the scale and to the packers'
    # orthogonal rotation.
    q_routing = q[2] if qk_packed else q[0]
    return _AITER.prepare(
        q_routing,
        k[0],
        v[0],
        beta,
        # Not sol_prepare's default tile, which is only one of the rows' tiles.
        *sol_attn_block_tile(recipe, block_tile),
        num_heads=q_routing.shape[2],
        k_scale=k[1] if qk_block_scaled else None,
        v_scale=v[1] if v_block_scaled else None,
        k_source=k[2],
        v_source=v[2],
        k_packed_format=recipe.k_packed,
        v_packed_format=recipe.v_packed,
        block_attn_mask=block_attn_mask,
        k_variance=k_variance,
        q_scale=q[1] if qk_block_scaled else None,
        # Only when set: older sol_prepare builds take no force_block_mask.
        **({} if force_block_mask is None else {"force_block_mask": force_block_mask}),
    )


def _with_scale_lookahead(scale):
    """A pooled E8M0 scale copied into storage that runs on, zeroed, past its last row.

    aiter's 64x64 MXFP8 Sol row reads the pooled K scale past the rows sol_prepare allocates for
    it, and faults the GPU whenever that allocation ends at an unmapped page. aiter backs its own
    K scales with lookahead rows for its readers; this gives the pooled one the same backing.
    """
    if _block_scale_storage is None:
        return scale
    backed = _block_scale_storage(scale, *scale.shape, _SCALE_TILE_ROWS, lookahead_rows=_SCALE_LOOKAHEAD_ROWS)
    backed.copy_(scale)
    return backed


# The recipes whose Sol rows take the pooled value tilt: every row but f6f8 (MXFP6 Q/K, FP8 V),
# which is the mxfp6 recipe.
_TILT_RECIPES = frozenset(r.id for r in _RECIPES.values() if r.tilt is not None)


def check_sol_tilt(recipe_id, tilt_rank):
    """Raise unless this aiter can serve the pooled value tilt at `tilt_rank` on `recipe_id`'s row."""
    if tilt_rank < 0:
        raise SolAttnUnsupported(f"--solattn_tilt_rank must be >= 0, got {tilt_rank}")
    if tilt_rank == 0:
        return
    if recipe_id not in _TILT_RECIPES:
        raise SolAttnUnsupported(
            f"--solattn_tilt_rank has no Sol-Attn '{recipe_id}' row with the pooled value tilt; "
            f"the rows with it are {', '.join(r for r in SOL_ATTN_RECIPES if r in _TILT_RECIPES)}"
        )
    if _tilt_factors is None or _jensen_cap is None:
        raise SolAttnUnsupported("--solattn_tilt_rank needs aiter's sol_jensen_cap and sol_tilt builders")


def _second_order(recipe, q, k, v, routing, softmax_scale, block_tile, tilt_rank):
    """The capped Jensen term and, with tilt_rank, the pooled value tilt, as mha_v4_packed kwargs.

    The Jensen term adds 0.5 * scale^2 * q^2 . var_k to each pooled block's logit, capped per block
    at ln(kv_tile): uncapped, a block whose scores are far from Gaussian takes most of the softmax.
    The tilt then adds each pooled block's leading Cov(v, k) q terms to its mean value, scaled by
    the block's capped gain. The recipes build the tilt through different aiter kernels.
    """
    out = dict(
        mean_k=routing["mean_k"],
        mean_v=routing["mean_v"],
        mean_k_scale=routing["mean_k_scale"],
        mean_v_scale=routing["mean_v_scale"],
    )
    q_tile, kv_tile = block_tile
    packed = recipe.k_packed is not None
    q_cap, qk_descales, block_scaled = q[0], (None, None), {}
    if recipe.qk_format == "BF16":
        q_abs_max = _q_abs_max(q[0], q_tile)
    elif recipe.qk_scale_mode != "block":
        # Per-tensor FP8 / INT8 Q and K.
        q_abs_max = _q_abs_max(q[0], q_tile)
        qk_descales = (q[1], k[1])
    elif not packed:
        # MXFP8: the quantizer folds softmax_scale * log2(e) into the Q codes.
        q_abs_max = _q_abs_max(q[0], q_tile, q_scale=q[1])
        block_scaled = dict(
            mean_k_var_scale=routing["mean_k_var_scale"], q_multiplier=_AITER.q_multiplier(softmax_scale)
        )
    else:
        # A packed Q cannot be read back, so its BF16 source goes through the packers' rotation.
        q_cap = q[2]
        q_abs_max = _q_abs_max(q_cap, q_tile, rotate=True)
        block_scaled = dict(
            mean_k_var_scale=routing["mean_k_var_scale"], mean_k_var_pooled=routing["mean_k_var_pooled"]
        )
    out["mean_k_var"], out["mean_k_var_scale"], gain = _jensen_cap(
        q_cap,
        *qk_descales,
        routing["mean_k_var"],
        softmax_scale,
        kv_tile,
        block_attn_mask=routing["block_attn_mask"],
        q_tile=q_tile,
        q_abs_max=q_abs_max,
        **block_scaled,
    )
    if not tilt_rank:
        return out

    gain = gain.sqrt()
    mean_k, mean_v = out["mean_k"], out["mean_v"]
    mean_k_scale, mean_v_scale = out["mean_k_scale"], out["mean_v_scale"]
    if recipe.tilt == "fp8_factors":
        out["mean_k"], out["mean_v"], out["sol_tilt"] = _tilt_factors(
            q[0],
            k[0],
            v[0],
            q[1],
            k[1],
            v[1],
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            block_gain=gain,
            q_abs_max=q_abs_max,
        )
        return out
    if recipe.tilt == "mxfp8_planes":
        # The score MFMA decodes Q and the A planes, so the multiplier carries no descale; the
        # K-scale plane stride rides in the parameter block's third dword. The planes are built
        # from K as the kernel sees it, its E8M0 codes dequantized: an addressable MXFP8 K keeps
        # no BF16 source, and the planes have to match the quantized K the exact pass reads.
        mult = _AITER.q_multiplier(softmax_scale)
        planes = _tilt_planes(
            _e8m0_dequantize(k[0], k[1]).to(torch.bfloat16),
            v[0],
            q_abs_max,
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            k_format="mxfp8",
            v_format="e4m3",
            q_descale=1.0 / mult,
            v_descale=v[1],
            q_multiplier=mult,
            mean_k_scale=mean_k_scale,
            block_gain=gain,
        )
        params = _tilt_scaled_params(planes, planes.k_scale_plane_stride)
    elif recipe.tilt == "bf16_planes":
        # BF16 A planes against the BF16 Q. BF16/FP8's B planes are FP8, half the K planes' width,
        # so the V plane stride rides in the parameter block's third dword.
        bf16_v = recipe.v_format == "BF16"
        planes = _tilt_planes(
            k[0],
            v[0],
            q_abs_max,
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            k_format="bf16",
            v_format="bf16" if bf16_v else "e4m3",
            v_descale=1.0 if bf16_v else v[1],
            block_gain=gain,
        )
        params = _tilt_scaled_params(planes, planes.v_plane_stride)
    elif recipe.tilt == "int8_planes":
        planes = _tilt_planes(
            k[0],
            v[0],
            q_abs_max,
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            k_format="int8",
            v_format="e4m3",
            q_descale=q[1],
            k_descale=k[1],
            v_descale=v[1],
            block_gain=gain,
        )
        params = _tilt_valu_params(planes, q[1], k[1])
    elif recipe.tilt == "fp8_k_mx_v_planes":
        # e4m3 A planes against the rotated FP8 Q; MXFP6 B planes packed from the pooled BF16 V by
        # V's own packer, their data and scale strides after the score multiplier.
        planes = _tilt_planes(
            k[0],
            v[2],
            q_abs_max,
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            k_format="e4m3",
            v_format="mxfp6_fp6_p",
            q_descale=q[1],
            k_descale=k[1],
            mean_v_scale=mean_v_scale,
            mean_v_pooled=routing["mean_v_pooled"],
            block_gain=gain,
        )
        params = _tilt_valu_params(planes, q[1], k[1], strides=(planes.v_plane_stride, planes.v_scale_plane_stride))
    elif recipe.tilt == "packed_k_planes":
        # Packed K: the A planes go through K's own packer, which rotates them as it rotates K, so
        # the factors take the unrotated sources and Q maxima. MXFP4's A-plane scales have an
        # image of their own, whose stride rides after V's two.
        planes = _tilt_planes(
            k[2],
            v[2],
            _q_abs_max(q[2], q_tile),
            mean_k,
            mean_v,
            softmax_scale,
            tilt_rank,
            kv_tile,
            k_format=recipe.k_packed,
            v_format=recipe.v_packed,
            q_multiplier=_AITER.q_multiplier(softmax_scale),
            mean_k_scale=mean_k_scale,
            mean_v_scale=mean_v_scale,
            mean_k_pooled=routing["mean_k_pooled"],
            mean_v_pooled=routing["mean_v_pooled"],
            block_gain=gain,
        )
        strides = (planes.v_plane_stride, planes.v_scale_plane_stride)
        if recipe.k_packed == "mxfp4":
            strides += (planes.k_scale_plane_stride,)
        params = _tilt_valu_params(planes, strides=strides)
    else:
        raise SolAttnUnsupported(f"Sol-Attn '{recipe.id}' names an unknown tilt path {recipe.tilt!r}")
    out["mean_k"], out["mean_v"] = planes.mean_k, planes.mean_v
    if planes.mean_k_scale is not None:
        out["mean_k_scale"] = planes.mean_k_scale
    if planes.mean_v_scale is not None:
        out["mean_v_scale"] = planes.mean_v_scale
    out["sol_tilt"] = (planes.rank, planes.k_plane_stride, params)
    return out


def _head_cost_from_routing(routing):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,)."""
    mask = routing["block_attn_mask"]  # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    return mask.sum(dim=(0, 2, 3), dtype=torch.float32)


# Below this many KV blocks the threshold's mean and standard deviation are noise. A heuristic.
_MIN_USEFUL_KV_BLOCKS = 16
_short_kv_warned = set()


def _warn_if_kv_too_short(seqlen_k, kv_tile):
    """Warn, in eager runs only, when a call is too short for Sol-Attn to route usefully."""
    if torch.compiler.is_compiling():
        return
    blocks = -(-seqlen_k // kv_tile)
    if blocks >= _MIN_USEFUL_KV_BLOCKS or blocks in _short_kv_warned:
        return
    _short_kv_warned.add(blocks)
    import warnings

    warnings.warn(
        f"Sol-Attn was handed a {seqlen_k}-token KV, i.e. {blocks} blocks of {kv_tile}. Its "
        f"routing threshold is a mean plus a multiple of a standard deviation over the blocks, "
        f"which says little at that count, and the pooled correction is then standing in for much "
        f"of the sequence. Expect a worse result than a dense backend, not a faster one. If this "
        f"is a short text-only attention inside a model whose main attention is long, that site "
        f"should be on a dense backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def _force_blocks_from_tokens(exact_tokens, seqlen_k, kv_tile):
    """(num_kv_blocks,) bool: the KV blocks holding any token `exact_tokens` flags.

    exact_tokens covers at least the call's seqlen_k keys; anything past them is the caller's pad,
    already dropped from K/V.
    """
    exact_tokens = exact_tokens[:seqlen_k]
    pad = -seqlen_k % kv_tile
    if pad:
        exact_tokens = torch.nn.functional.pad(exact_tokens, (0, pad))
    return exact_tokens.reshape(-1, kv_tile).any(dim=1)


def _force_rows_from_tokens(exact_query_tokens, seqlen_q, q_tile):
    """(num_q_tiles, 1) bool: the query tiles holding any row `exact_query_tokens` flags.

    Routing selects per query tile, so a flagged row brings its whole tile along.
    """
    exact_query_tokens = exact_query_tokens[:seqlen_q]
    pad = -seqlen_q % q_tile
    if pad:
        exact_query_tokens = torch.nn.functional.pad(exact_query_tokens, (0, pad))
    return exact_query_tokens.reshape(-1, q_tile).any(dim=1, keepdim=True)


def _pad_kv_to_tile(key, value, recipe, block_tile):
    """Right-pad BSHD K/V with zero tokens to a whole number of KV blocks, unless the row is ragged.

    A zero key scores 0 rather than -inf, so the pad draws a little softmax weight: at most one
    block's worth, exponentially small in the row max.
    """
    pad = -key.shape[1] % block_tile[1]
    if pad == 0 or _takes_ragged_kv(recipe, block_tile):
        return key, value
    widths = (0, 0, 0, 0, 0, pad)
    return (torch.nn.functional.pad(key, widths), torch.nn.functional.pad(value, widths))


def sol_attn_bhsd(
    query,
    key,
    value,
    is_causal=False,
    beta=1.0,
    softmax_scale=None,
    routing=None,
    ring_world_size=1,
    return_head_cost=False,
    recipe="fp8",
    key_seqlen=None,
    block_tile=None,
    query_seqlen=None,
    exact_tokens=None,
    exact_query_tokens=None,
    tilt_rank=0,
    *,
    _jensen_only=False,
):
    """Sol-Attn over BHSD tensors, returning (BHSD bf16 output, per-head cost or None).

    query/key/value are (batch, nheads, seqlen, head_dim) bf16 tensors. They are permuted into the
    BSHD layout the kernel takes and handed to aiter's mha_v4 Sol-Attn row for `recipe`, which owns
    the Q/K Hadamard rotation, format validation and the ASM launch. See SOL_ATTN_RECIPES for the
    rows this build ships.

    return_head_cost asks for the per-head count of exactly-computed KV blocks, float32 (nheads_q,),
    for the Ulysses head balancer. The raw entry point keeps its routing internal, so asking for it
    moves the call onto aiter's packed API with quantization and routing done here. MXFP8 and a
    caller-supplied `routing` always take that path.

    key_seqlen is the number of real KV tokens when the caller has padded the sequence. Sol-Attn
    takes no mask, so the tail is dropped rather than attended.

    query_seqlen is the same for the query rows. Routing takes one selection per query tile, so
    pad rows left in would vote on the selection of the real rows sharing their tile. They are
    dropped before the call and come back as zero rows.

    exact_tokens is a (seqlen_k,) bool mask, or longer by the caller's pad: every KV block holding
    a flagged token is computed exactly for every query, on top of what routing selects. For a
    sequence whose minority -- a modality, say -- would otherwise be routed against a threshold
    the majority sets. It moves the call onto the packed API.

    exact_query_tokens is the query-side counterpart, a (seqlen_q,) bool mask or longer: every
    query tile holding a flagged row computes all of its KV blocks exactly. For rows another part
    of the model reads back, such as a prompt that conditions the blocks after it.

    block_tile is the (q_tile, kv_tile) to route and dispatch at, from --sparseattn_block_size.
    None takes the recipe's default.

    tilt_rank, from --solattn_tilt_rank, adds the leading Cov(v, k) q terms of each pooled block to
    its mean value, on top of the capped Jensen term on its logit, and takes the packed API.
    _jensen_only applies the capped term alone: a comparison for the tests, which no option reaches.
    """
    if _AITER is None:
        raise SolAttnUnsupported("Sol-Attn requires aiter with mha_v4_sol and sol_prepare; please update AITER")
    check_sol_attn_supported(query, key, value, is_causal, ring_world_size=ring_world_size)
    recipe = _resolve_recipe(recipe)
    block_tile = sol_attn_block_tile(recipe, block_tile)

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    query_pad = 0
    if query_seqlen is not None and query_seqlen < query.shape[1]:
        query_pad = query.shape[1] - query_seqlen
        query = query[:, :query_seqlen]
    # Before the tile pad, so the caller's pad and the tile pad do not stack.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    if (exact_tokens is not None or exact_query_tokens is not None) and not _SOL_PREPARE_FORCES_BLOCKS:
        raise SolAttnUnsupported(
            "exact KV or query tokens need an aiter whose sol_prepare takes force_block_mask; please update AITER"
        )
    force_blocks = (
        None if exact_tokens is None else _force_blocks_from_tokens(exact_tokens, key.shape[1], block_tile[1])
    )
    if exact_query_tokens is not None:
        rows = _force_rows_from_tokens(exact_query_tokens, query.shape[1], block_tile[0])
        num_kv_blocks = -(-key.shape[1] // block_tile[1])
        force_blocks = rows.expand(-1, num_kv_blocks) if force_blocks is None else rows | force_blocks
    key, value = _pad_kv_to_tile(key, value, recipe, block_tile)
    _warn_if_kv_too_short(key.shape[1], block_tile[1])

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    check_sol_tilt(recipe.id, tilt_rank)
    second_order = _jensen_only or tilt_rank > 0
    raw = routing is None and not return_head_cost and force_blocks is None and not second_order
    if raw and recipe.routes_through_raw:
        qk, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _AITER.sol(
            query, key, value, qk, qk, v_fmt, beta=beta, softmax_scale=softmax_scale, block_tile=block_tile
        )
        return _restore_pad_rows(out, query_pad).permute(0, 2, 1, 3), None

    q, k, v = _quantize(recipe, query, key, value, softmax_scale)
    if routing is None:
        routing = sol_attn_routing_for(
            q, k, v, beta, recipe, block_tile=block_tile, force_block_mask=force_blocks, k_variance=second_order
        )
    if second_order:
        pooled = _second_order(recipe, q, k, v, routing, softmax_scale, block_tile, tilt_rank)
    else:
        pooled = dict(
            mean_k=routing["mean_k"],
            mean_v=routing["mean_v"],
            # None where the pooled operand reuses the source descale.
            mean_k_scale=routing["mean_k_scale"],
            mean_v_scale=routing["mean_v_scale"],
        )
    if not recipe.routes_through_raw and "sol_tilt" not in pooled:
        # Only the MXFP8 row reads past the pooled K scale, and it is the one row that is not raw.
        # The tilt's planes come in AITER's own padded storage; sol_prepare's pooled scale does
        # not. Drop the copy once AITER's sol_prepare pads the scale itself.
        pooled["mean_k_scale"] = _with_scale_lookahead(pooled["mean_k_scale"])
    qk_fmt, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale, v_scale = _scale_mode(recipe.qk_scale_mode), _scale_mode(recipe.v_scale_mode)
    out = _AITER.packed(
        q[0],
        k[0],
        v[0],
        q[1],
        k[1],
        v[1],
        qk_fmt,
        qk_fmt,
        v_fmt,
        qk_scale,
        qk_scale,
        v_scale,
        softmax_scale=softmax_scale,
        kv_block_indices=routing["kv_block_indices"],
        lut_start=routing["lut_start"],
        lut_count=routing["lut_count"],
        block_bitmap=routing["block_bitmap"],
        **pooled,
        block_tile=block_tile,
        v_pack=getattr(_AITER.pack, recipe.v_pack),
    )
    return _restore_pad_rows(out, query_pad).permute(0, 2, 1, 3), _head_cost_from_routing(routing)


def _restore_pad_rows(out, pad):
    """Put back, as zeros, the `pad` query rows a BSHD output was computed without."""
    if pad == 0:
        return out
    return torch.nn.functional.pad(out, (0, 0, 0, 0, 0, pad))
