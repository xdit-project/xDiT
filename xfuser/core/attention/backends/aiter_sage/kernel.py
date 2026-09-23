"""Sage mask sources. Imported when a Sage backend is selected."""

import math

from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.ssta import (
    expand_block_mask,
    get_sparse_mask,
    setup_ssta,
    untile_ssta_output,
)

from .spec import _TRITON_SSTA_BLOCK, _block_r


def _resolve(target: str):
    module, _, symbol = target.partition(":")
    return getattr(__import__(module, fromlist=[symbol]), symbol)


def _attn(kernel):
    return _resolve(kernel.wrapper)


def _config(kernel) -> dict:
    return _resolve(kernel.configs)()


def _extra(kernel, query) -> dict:
    if not kernel.rotates:
        return {}
    return {"hadamard_rotation": True, "R": hadamard.matrix(_block_r(), str(query.device))}


def _prepare(kernel, *tensors):
    # aiter-shim: added 2026-03-12. Sage v2 needed contiguous inputs in older
    # builds. Pre-floor, so it is a removal candidate -- but unlike the
    # probe-guarded shims this is an unconditional copy, and "AITER no longer
    # needs it" cannot be established by introspection, only by a bitwise run
    # with and without. Kept until that is measured.
    if not kernel.force_contiguous:
        return tensors
    return tuple(t.contiguous() for t in tensors)


def _causal(kernel, call: AttnCall) -> dict:
    return {"causal": call.is_causal} if kernel.passes_causal else {}

def dense(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """No mask. The only cell that can participate in ring attention, and only
    then does the wrapper produce an LSE."""
    q, k, v = _prepare(kernel, query, key, value)
    attn = _attn(kernel)
    extra = _extra(kernel, q)

    if call.ctx.ring_world_size > 1:
        # aiter-shim cut 2026-09: AITER_SAGE_SUPPORTS_RING /
        # AITER_SAGE_V2_SUPPORTS_RING (both added 2026-06-17) probed for these
        # parameters. Every build at or after the July floor has them.
        lse_args = {"return_lse": True}
        if not kernel.rotates:
            lse_args["smooth_k"] = True
        return attn(q, k, v, layout="bhsd", **extra, **lse_args, **_causal(kernel, call))

    return attn(q, k, v, layout="bhsd", **extra, **_causal(kernel, call)), None


def ssta(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """Tile-based static mask, supplied by the model's sparse config."""
    # Not hoisted: only the SSTA and sparge specs require this symbol; the
    # dense specs in this family must load without it.
    from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

    kwargs = call.attention_kwargs
    kwargs["sp_size"] = call.ctx.ulysses_world_size
    block_size = math.prod(kwargs["tile_size"])

    config = _config(kernel)
    config["BLOCK_M"] = _TRITON_SSTA_BLOCK
    config["BLOCK_N"] = _TRITON_SSTA_BLOCK

    q, k, v, mask_config, state = setup_ssta(query, key, value, kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=kwargs["attn_sparse_type"])
    if block_size != _TRITON_SSTA_BLOCK:
        block_mask = expand_block_mask(block_mask, factor=block_size // _TRITON_SSTA_BLOCK)

    output = _attn(kernel)(
        q, k, v,
        layout="bhsd", config=config, **_extra(kernel, q),
        block_lut=block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1]),
        **_causal(kernel, call),
    )
    output = untile_ssta_output(
        output, state, kwargs["encoder_sequence_length"], kwargs["sp_size"]
    )
    return output, None


def sparge(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """Data-dependent mask computed from Q/K."""
    # Not hoisted: only the SSTA and sparge specs require this symbol; the
    # dense specs in this family must load without it.
    from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

    query, key, value = _prepare(kernel, query, key, value)
    config = _config(kernel)

    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=config["BLOCK_M"], block_n=config["BLOCK_N"],
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
    )

    output = _attn(kernel)(
        q, k, v,
        layout="bhsd", config=config, **_extra(kernel, q),
        block_lut=block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1]),
        **_causal(kernel, call),
    )
    return restore_sparge_output(output, state), None


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

