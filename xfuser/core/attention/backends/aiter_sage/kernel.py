"""Sage mask sources, one function per backend.

Six functions rather than three parameterised by a version flag: the two
versions share only their mask construction, and what they do differ in is not
uniform. Sage v2 wants contiguous inputs -- except under SSTA, which does not
apply them -- and is passed `causal`, which v1's wrapper accepts but is never
given. A flag would assert those hold per version; they do not.
"""

import math

from aiter.ops.triton.attention.fav3_sage import (
    fav3_sage_wrapper_func as SAGE_V1,
    get_sage_fwd_configs as CONFIG_V1,
)
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_wrapper as SAGE_V2,
    get_sage_fwd_configs_mxfp4 as CONFIG_V2,
)
from aiter.ops.triton.attention.utils import (
    block_attn_mask_to_ragged_lut as RAGGED_LUT,
)

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.numerics.layout import make_contiguous
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

from .spec import BLOCK_R, TRITON_SSTA_BLOCK


# ---------------------------------------------------------------------------
# shared by both versions
# ---------------------------------------------------------------------------

def _ssta_mask(query, key, value, call: AttnCall):
    """Tile Q/K/V against the model's static sparse config and build the LUT."""
    kwargs = call.attention_kwargs
    kwargs["sp_size"] = call.ctx.ulysses_world_size
    block_size = math.prod(kwargs["tile_size"])

    q, k, v, mask_config, state = setup_ssta(query, key, value, kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=kwargs["attn_sparse_type"])
    if block_size != TRITON_SSTA_BLOCK:
        block_mask = expand_block_mask(block_mask, factor=block_size // TRITON_SSTA_BLOCK)

    return q, k, v, RAGGED_LUT(block_mask, num_heads=q.shape[1]), state


def _ssta_config(config_fn) -> dict:
    """SSTA drives the Triton kernels at a fixed block size, not the tuned one."""
    config = config_fn()
    config["BLOCK_M"] = TRITON_SSTA_BLOCK
    config["BLOCK_N"] = TRITON_SSTA_BLOCK
    return config


def _untile(output, state, call: AttnCall):
    kwargs = call.attention_kwargs
    return untile_ssta_output(
        output, state, kwargs["encoder_sequence_length"], kwargs["sp_size"]
    )


def _sparge_mask(query, key, value, call: AttnCall, config: dict):
    """Reorder Q/K/V by a mask computed from Q/K, at the config's tile size."""
    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=config["BLOCK_M"], block_n=config["BLOCK_N"],
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
    )
    return q, k, v, RAGGED_LUT(block_mask, num_heads=q.shape[1]), state


def _rotation(query) -> dict:
    """Sage v2 rotates Q/K by a Hadamard matrix inside the kernel."""
    return {"hadamard_rotation": True, "R": hadamard.matrix(BLOCK_R, str(query.device))}


# ---------------------------------------------------------------------------
# v1
# ---------------------------------------------------------------------------

def sage(query, key, value, call: AttnCall):
    """No mask. The only cell that can join a ring, and only then is there an
    LSE to merge."""
    if call.ctx.ring_world_size > 1:
        return SAGE_V1(query, key, value, layout="bhsd", return_lse=True, smooth_k=True)
    return SAGE_V1(query, key, value, layout="bhsd"), None


def sparse_sage(query, key, value, call: AttnCall):
    """Tile-based static mask, supplied by the model's sparse config."""
    config = _ssta_config(CONFIG_V1)
    q, k, v, block_lut, state = _ssta_mask(query, key, value, call)
    output = SAGE_V1(q, k, v, layout="bhsd", config=config, block_lut=block_lut)
    return _untile(output, state, call), None


def sparge(query, key, value, call: AttnCall):
    """Data-dependent mask computed from Q/K."""
    config = CONFIG_V1()
    q, k, v, block_lut, state = _sparge_mask(query, key, value, call, config)
    output = SAGE_V1(q, k, v, layout="bhsd", config=config, block_lut=block_lut)
    return restore_sparge_output(output, state), None


# ---------------------------------------------------------------------------
# v2
# ---------------------------------------------------------------------------

def sage_v2(query, key, value, call: AttnCall):
    """No mask. The only cell that can join a ring, and only then is there an
    LSE to merge."""
    q, k, v = make_contiguous(query, key, value)
    rotation = _rotation(q)
    if call.ctx.ring_world_size > 1:
        return SAGE_V2(q, k, v, layout="bhsd", **rotation,
                       return_lse=True, causal=call.is_causal)
    return SAGE_V2(q, k, v, layout="bhsd", **rotation, causal=call.is_causal), None


def sparse_sage_v2(query, key, value, call: AttnCall):
    """Tile-based static mask. The one v2 path that does not make its inputs
    contiguous: setup_ssta reshapes and expands Q/K/V on the way through, so
    what the wrapper receives here is not what the other two hand it. Whether
    that is safe has not been established."""
    config = _ssta_config(CONFIG_V2)
    q, k, v, block_lut, state = _ssta_mask(query, key, value, call)
    output = SAGE_V2(
        q, k, v, layout="bhsd", config=config, **_rotation(q),
        block_lut=block_lut, causal=call.is_causal,
    )
    return _untile(output, state, call), None


def sparge_v2(query, key, value, call: AttnCall):
    """Data-dependent mask computed from Q/K."""
    query, key, value = make_contiguous(query, key, value)
    config = CONFIG_V2()
    q, k, v, block_lut, state = _sparge_mask(query, key, value, call, config)
    output = SAGE_V2(
        q, k, v, layout="bhsd", config=config, **_rotation(q),
        block_lut=block_lut, causal=call.is_causal,
    )
    return restore_sparge_output(output, state), None
