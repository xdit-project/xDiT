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

    The BF16 rows carry a NONE scale mode, so the kernel never reads the descale. aiter's own
    mha_v4 hands the tensor itself back as the placeholder rather than a unit scalar, and this
    matches that so the packed path here cannot disagree with the raw one about what it sent.
    """
    return tensor, tensor


def _probe_aiter():
    """aiter's Sol-Attn entry points, or None when this build does not ship them.

    Resolved at import. This module is imported lazily, only once the backend is selected, so builds
    that never touch Sol-Attn pay nothing; binding the entry points once also keeps repeated imports
    out of any traced graph. native_fp8_format is held as a function rather than called here because
    it queries the device, which import time is too early for.
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
    """aiter's per-row ragged-KV query, or None on a build that predates it.

    Optional rather than part of _probe_aiter: without it every row is treated as needing its key
    length padded to the tile, which is what every row needed before the ragged ones shipped.
    """
    try:
        from aiter.ops.mha_v4 import mha_v4_ragged_kv
    except ImportError:
        return None
    return mha_v4_ragged_kv


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None

# Held as module globals rather than _AITER fields, which is not a style choice. aiter decorates
# its geometry queries with functools.cache, so each name is an _lru_cache_wrapper -- a C object
# that implements the descriptor protocol. Reached through an attribute Dynamo binds the owner as
# self and the call dies with "Too many positional arguments", which under fullgraph=True is a hard
# error and otherwise a graph break in the middle of every attention layer. Eager does not bind
# (SimpleNamespace holds it in an instance dict) so this shows up only compiled.
_default_block_tile = _AITER.block_tile if _AITER is not None else None
_kv_tile_for_q_tile = _AITER.kv_tile_for_q_tile if _AITER is not None else None
_ragged_kv = _probe_ragged_kv() if _AITER is not None else None


def sol_attn_block_tile(recipe, block_tile=None):
    """The (q_tile, kv_tile) a call routes, pools and dispatches `recipe` at.

    One source for all three. They are not independently choosable: the LUT and the selection
    bitmap are in units of kv_tile, the pooled K/V are one row per kv_tile tokens, and the kernel
    folds log2(kv_tile) into its softmax bias as a build-time constant. A mismatch between any two
    is silently wrong rather than an error, which is why every site below reads it from here.

    block_tile is --sparseattn_block_size when the run sets one, which check_sol_attn_recipe has
    already checked against the recipe's row. Unset, the recipe picks the default, and it has to:
    the gfx950 rows do not agree on the KV tile at a 256-row query tile, since BF16 and BF16/FP8
    route on 64 tokens and the rest on 128.

    Deliberately not cached: the aiter query is cached and torch_compile_guard'd on its side, so
    this body is a tuple return that Dynamo folds away.
    """
    if block_tile is not None:
        return tuple(block_tile)
    return _default_block_tile(_recipe_operands(recipe), _AITER.sol_mode)


class _Recipe(NamedTuple):
    """One mode-2 manifest row: its formats, its quantizers, and how its scales survive pooling.

    The recipes divide on one question, which is whether an operand's scale varies along the
    sequence axis pooling reduces. A per-tensor descale does not, nor does a per-channel one, so
    the pooled operand reuses it and the kernel's pooled-scale slots stay empty; an unscaled BF16
    operand has nothing to reuse and behaves the same way. An E8M0 1x32 scale does vary, so that
    operand pools in dequantized space and hands the kernel a pooled scale of its own.

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
    # Set when that operand's stored codes are sub-byte and permuted into the ASM's tile order,
    # naming the packing as sol_prepare does. Such an operand cannot be pooled from its codes at
    # all, so pooling works from the BF16 source and its own scale must not be handed over. A
    # packed Q/K is not addressable for routing either, so it routes on the BF16 Q as well.
    k_packed: Optional[str] = None
    v_packed: Optional[str] = None

    @property
    def v_pack(self) -> str:
        """The AttentionPack V is stored in: the FP6-P order for every MX V, else the default."""
        return "V_FOR_FP6_P" if (self.v_packed or "").endswith("_fp6_p") else "DEFAULT"

    @property
    def routes_through_raw(self) -> bool:
        """Whether mha_v4_sol can serve this row, or it has to go packed.

        mha_v4_sol serves every row here but one: an addressable block-scaled K, which is MXFP8,
        pools in dequantized space from a scale it would have to be handed, and the raw entry
        point has no way to take one. MXFP8 reaches the same kernel through mha_v4_packed with
        operands quantized here.
        """
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
        # Q/K stay BF16 and only V drops to FP8, which halves the traffic on the operand the
        # correction pass reads once per KV block while leaving the scores exact.
        _Recipe("bf16fp8", "BF16", "none", "quantize_bf16", "quantize_bf16"),
        _Recipe("fp8", "native", "per_tensor", "quantize_fp8_rotated", "quantize_fp8_rotated"),
        _Recipe("i8fp8", "INT8", "per_tensor", "quantize_int8", "quantize_int8"),
        _Recipe("mxfp8", "native", "block", "quantize_mxfp8_q", "quantize_mxfp8_k"),
        # The MX-V rows, all on the FP6-P layout: V is stored in the order the kernel's FP6 P
        # operand reads it, which is what lets them return an LSE and take a ragged KV.
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
        # The one MX Q/K row whose V is addressable: FP8, scaled per channel, so it pools on its
        # codes like the per-tensor rows and needs no FP6-P layout.
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

    Both facts are fixed for the process, so runtime_state calls this once during backend setup and
    the per-call path does not repeat it. Callers reaching sol_attn_bhsd directly, outside xDiT's
    setup, should call it themselves.
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

    Separate from check_sol_attn_device because the answer is per recipe, read from aiter's
    manifest, so selecting a recipe the device has no row for fails at setup with the reason rather
    than at the first launch with a missing-kernel error. block_tile is --sparseattn_block_size:
    a geometry need not exist in every precision, so it is valid only for the recipes whose own
    row serves it.
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

    Resolved per device, since a recipe's format may be "native". V's pack is one of them: the
    FP6-P rows share their formats and scale modes with rows of the default layout.
    """
    qk_format, v_format = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale, v_scale = _scale_mode(recipe.qk_scale_mode), _scale_mode(recipe.v_scale_mode)
    return _AITER.operands(
        qk_format, qk_format, v_format, qk_scale, qk_scale, v_scale, getattr(_AITER.pack, recipe.v_pack)
    )


def _takes_ragged_kv(recipe, block_tile):
    """Whether `recipe`'s row at `block_tile` masks a short last KV block itself.

    Every gfx950 Sol row does today, so it is handed the true key length; a row that does not needs
    K/V padded to a whole number of blocks.
    """
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

    Exactly as mha_v4_sol quantizes the same row, so that asking for the head cost moves a call
    onto the packed API without changing its output. The MX quantizers fold
    softmax_scale * log2(e) into Q, which is why the scale has to be resolved before quantizing
    rather than left for the kernel. The per-tensor quantizers take no multiplier and the kernel
    applies the scale itself, so it is passed on either way.

    source is the pre-quantization tensor, kept only for a packed operand: its codes are sub-byte
    and permuted into tile order, so neither pooling nor routing can read them back. A packed K
    keeps Q's source too, because a packed Q is no more addressable for routing.
    """
    multiplier = _AITER.q_multiplier(softmax_scale)

    if recipe.qk_scale_mode == "block":
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query, multiplier)
    else:
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query)

    k, k_descale = getattr(_AITER, recipe.quantize_k)(key)
    # These packers emit a flat backing buffer plus its scale; the kernel wants the strided view
    # over that buffer, so build it here exactly as aiter's own recipe does.
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

    q/k/v are the (tensor, descale, source) triples _quantize returns. num_heads is passed
    explicitly because the routing is per query head under GQA, while the pooled K/V it returns
    carry the KV head count instead.

    block_attn_mask supplies an already chosen selection instead of routing from beta, which is
    what lets a caller measure some other rule's selection through this same kernel and pooled
    correction. Pass exactly one of it and beta.
    """
    qk_packed = recipe.k_packed is not None
    # An addressable E8M0-scaled operand: its codes are not proportional to the values, since the
    # scale varies per token and channel group, so routing and pooling are handed the scale.
    qk_block_scaled = recipe.qk_scale_mode == "block" and not qk_packed
    v_block_scaled = recipe.v_scale_mode == "block" and recipe.v_packed is None
    # Routing scores Q, and a packed Q is no more addressable than a packed K, so a packed recipe
    # routes on its source. Routing is scale invariant, which is what makes the two
    # interchangeable, and the packers' Hadamard rotation is orthogonal so it drops out too.
    q_routing = q[2] if qk_packed else q[0]
    return _AITER.prepare(
        q_routing,
        k[0],
        v[0],
        beta,
        # Not aiter's SOL_ATTN_TS_QO/TS_KV defaults, which are the gfx950 256x128 row: the BF16
        # rows pool 64 keys per block, and an opted-in run may be on a 64x64 row. Pooling at the
        # wrong tile is not a rounding difference, it hands the kernel pooled tensors of the wrong
        # height. The dispatch reads the same source, which keeps the two from drifting apart.
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


# Below this many KV blocks, routing has too few samples to say anything. tau is a mean plus a
# multiple of a standard deviation taken across the blocks, so with a handful of them the
# threshold is noise, and whatever it does not select is replaced by block means that are then a
# large fraction of the whole sequence. 16 is a judgement call, not a measured cliff.
_MIN_USEFUL_KV_BLOCKS = 16
_short_kv_warned = set()


def _warn_if_kv_too_short(seqlen_k, kv_tile):
    """Say something when a call is too short for Sol-Attn to be worth routing.

    This is the failure mode that does not announce itself: nothing raises, the answer is just
    quietly worse than the dense backend would have given, which is easy to ship by accident when
    a model turns out to have a second attention site on a short sequence.

    Skipped under torch.compile. The branch folds away at trace time, which keeps the warning from
    being a side effect in a traced region -- so a fully compiled run will not see it, and this is
    a backstop for eager runs rather than a guarantee.
    """
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
    """Right-pad BSHD K/V with zero tokens to a whole number of KV blocks, where the row needs it.

    A ragged row masks the keys past the end of a short last block, so it is handed the true key
    length and nothing here runs. That is every gfx950 Sol row, which matters because Wan lands on
    a short block at every standard size -- 720p is 21 latent frames x 80 x 45 = 75600 tokens,
    which is 590.6 blocks of 128.

    A row that indexes KV in whole tiles makes aiter reject a ragged seqlen_k, so this pads.
    Zero tokens are not free: a zero key scores q . 0 = 0 rather than -inf, so the pad draws
    softmax weight exp(-max) per token instead of none, and its zero value pulls the row toward
    the origin by that weight. The pad is at most one block and the mass it takes is
    exponentially small in the row max, which is why this is a pad and not a mask. Q is
    deliberately left alone: nothing constrains seqlen_q, and padding it would only add rows to
    slice back off the output.
    """
    pad = -key.shape[1] % block_tile[1]
    if pad == 0 or _takes_ragged_kv(recipe, block_tile):
        return key, value
    # BSHD, so the seqlen axis is the second of four and F.pad counts from the last.
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
    which the Ulysses head balancer consumes. On a per-tensor recipe it is opt-in rather than free:
    the routing dict it reduces is internal to the raw entrypoint, so asking for it moves the call
    onto aiter's packed API and makes quantization and routing explicit here. MXFP8 has no raw
    entrypoint and takes that path always. A caller-supplied `routing` also forces it. Every path is
    traceable; none graph-break.

    key_seqlen is the number of real KV tokens when the caller has already padded the sequence for
    its own alignment. Sol-Attn takes no attention mask, so the padded tail is dropped here rather
    than attended: left in, it would draw softmax weight and also enter the routing threshold's
    per-tile mean and standard deviation, which is a quieter error than the weight itself. Defaults
    to the whole tensor. What remains is handed over as it is on a row that takes a ragged KV, so
    the kernel attends exactly key_seqlen keys; only a row that cannot is padded back up to its
    tile.

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

    # Before the tile pad, so the two do not stack: the caller's alignment is dropped, and only a
    # row that cannot take a ragged KV has its own remainder padded back, at most one block of zero
    # tokens.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    key, value = _pad_kv_to_tile(key, value, recipe, block_tile)
    _warn_if_kv_too_short(key.shape[1], block_tile[1])

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    # The raw entrypoint routes internally from beta alone, so it cannot be asked for the head cost.
    if routing is None and not return_head_cost and recipe.routes_through_raw:
        qk, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _AITER.sol(
            query, key, value, qk, qk, v_fmt, beta=beta, softmax_scale=softmax_scale, block_tile=block_tile
        )
        return out.permute(0, 2, 1, 3), None

    # Quantize exactly as the raw entrypoint would. On the fp8 row that means quantize_fp8_rotated
    # for Q/K: rotating both by the same orthonormal matrix leaves Q @ K.T alone while spreading the
    # outliers that dominate fp8 error, and V is not rotated because nothing cancels a rotation of
    # it. Reusing aiter's fused rotation rather than doing a second one here is what keeps this path
    # numerically identical to the raw one, so asking for the head cost cannot change the output.
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
        # None on the per-tensor rows, where the pooled operand reuses the source descale. The MX
        # rows pool in dequantized space and requantize, so the pooled tensor has a scale of its own
        # and the kernel has to be told about it or it reads the approximate branch at the wrong
        # magnitude.
        mean_k_scale=routing["mean_k_scale"],
        mean_v_scale=routing["mean_v_scale"],
        # The row to dispatch, not a preference: routing above already pooled and packed for this
        # geometry, so the kernel has to be the one that reads it that way.
        block_tile=block_tile,
        v_pack=getattr(_AITER.pack, recipe.v_pack),
    )
    return out.permute(0, 2, 1, 3), _head_cost_from_routing(routing)
