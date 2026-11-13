"""Sol-Attn: block-sparse attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.
"""

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


def _probe_ragged_kv():
    """aiter's per-row ragged-KV query, or None on a build that predates it, where every row
    has its key length padded to the tile."""
    try:
        from aiter.ops.mha_v4 import mha_v4_ragged_kv
    except ImportError:
        return None
    return mha_v4_ragged_kv


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None

# Module globals, not _AITER fields: these are functools.cache wrappers, and Dynamo binds one
# reached through an attribute as a method, which fails with "Too many positional arguments".
_default_block_tile = _AITER.block_tile if _AITER is not None else None
_kv_tile_for_q_tile = _AITER.kv_tile_for_q_tile if _AITER is not None else None
_ragged_kv = _probe_ragged_kv() if _AITER is not None else None


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
        ),
        _Recipe("bf16fp8", "BF16", "none", "quantize_bf16", "quantize_bf16"),
        _Recipe("fp8", "native", "per_tensor", "quantize_fp8_rotated", "quantize_fp8_rotated"),
        _Recipe("i8fp8", "INT8", "per_tensor", "quantize_int8", "quantize_int8"),
        _Recipe("mxfp8", "native", "block", "quantize_mxfp8_q", "quantize_mxfp8_k"),
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


def sol_attn_routing_for(q, k, v, beta, recipe=_RECIPES["fp8"], block_attn_mask=None, block_tile=None):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them.

    q/k/v are the (tensor, descale, source) triples _quantize returns. block_attn_mask supplies an
    already chosen selection instead of routing from beta; pass exactly one of the two.
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
        q_scale=q[1] if qk_block_scaled else None,
    )


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

    block_tile is the (q_tile, kv_tile) to route and dispatch at, from --sparseattn_block_size.
    None takes the recipe's default.
    """
    if _AITER is None:
        raise SolAttnUnsupported("Sol-Attn requires aiter with mha_v4_sol and sol_prepare; please update AITER")
    check_sol_attn_supported(query, key, value, is_causal, ring_world_size=ring_world_size)
    recipe = _resolve_recipe(recipe)
    block_tile = sol_attn_block_tile(recipe, block_tile)

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    # Before the tile pad, so the caller's pad and the tile pad do not stack.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    key, value = _pad_kv_to_tile(key, value, recipe, block_tile)
    _warn_if_kv_too_short(key.shape[1], block_tile[1])

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    if routing is None and not return_head_cost and recipe.routes_through_raw:
        qk, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _AITER.sol(
            query, key, value, qk, qk, v_fmt, beta=beta, softmax_scale=softmax_scale, block_tile=block_tile
        )
        return out.permute(0, 2, 1, 3), None

    q, k, v = _quantize(recipe, query, key, value, softmax_scale)
    if routing is None:
        routing = sol_attn_routing_for(q, k, v, beta, recipe, block_tile=block_tile)
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
        mean_k=routing["mean_k"],
        mean_v=routing["mean_v"],
        block_bitmap=routing["block_bitmap"],
        # None where the pooled operand reuses the source descale.
        mean_k_scale=routing["mean_k_scale"],
        mean_v_scale=routing["mean_v_scale"],
        block_tile=block_tile,
        v_pack=getattr(_AITER.pack, recipe.v_pack),
    )
    return out.permute(0, 2, 1, 3), _head_cost_from_routing(routing)
