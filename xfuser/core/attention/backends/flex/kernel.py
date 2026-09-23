"""FlexAttention-backed sparse kernels: block masks and VSA-H3 tiling."""

import math

import torch

from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttentionBackendType, AttnCall
from xfuser.core.distributed.ssta import (
    get_sparse_mask,
    setup_ssta,
    untile_ssta_output,
)



def _flex_op():
    """The custom op still lives in the legacy module; importing it here (at
    call time, to avoid a cycle) guarantees registration. Phase 6 moves it."""
    import xfuser.core.distributed.attention_backend  # noqa: F401

    return torch.ops.xfuser.flex_block_attn


def flex_block(query, key, value, call: AttnCall):
    """SSTA tile mask through FlexAttention."""
    kwargs = call.attention_kwargs
    kwargs["sp_size"] = call.ctx.ulysses_world_size
    block_size = math.prod(kwargs["tile_size"])

    q, k, v, mask_config, state = setup_ssta(query, key, value, kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=kwargs["attn_sparse_type"])
    output = _flex_op()(q, k, v, block_size, block_size, block_mask)
    output = untile_ssta_output(
        output, state, kwargs["encoder_sequence_length"], kwargs["sp_size"]
    )
    return output, None


def flex_sparge(query, key, value, call: AttnCall):
    """Sparge block mask through FlexAttention, at a fixed 256 tile."""
    block = 256
    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=block, block_n=block,
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=True,
    )
    output = _flex_op()(
        q.contiguous(), k.contiguous(), v.contiguous(), block, block, block_mask
    )
    return restore_sparge_output(output, state), None


def flex_vsa_h3(query, key, value, call: AttnCall):
    """FastH3's 64-token VSA-H3. USP has already gathered the sequence; the
    compression gate rides the same exchange. Without that metadata -- the
    MiniMax-H3 token refiner -- this runs dense, the way VSA does without thw.
    """
    from xfuser.core.vsa_h3_attention import (
        flex_h3_vsa_attention,
        tile_h3_vsa_tensor,
        untile_h3_vsa_tensor,
    )

    kwargs = call.attention_kwargs
    metadata = kwargs.get("vsa_h3_metadata")
    gate = kwargs.get("vsa_h3_gate")
    if metadata is None or gate is None:
        return _dense_fallback(query, key, value, call)

    sequence_length = metadata.total_seq_length
    gathered_length = query.shape[2]
    query, key, value, gate = (
        t[:, :, :sequence_length] for t in (query, key, value, gate)
    )

    def tile(tensor):
        return tile_h3_vsa_tensor(
            tensor.transpose(1, 2), metadata
        ).transpose(1, 2).contiguous()

    sparse, compressed = flex_h3_vsa_attention(
        tile(query), tile(key), tile(value), metadata
    )
    tiled = sparse + compressed.to(sparse.dtype) * tile(gate)
    packed = untile_h3_vsa_tensor(tiled.transpose(1, 2), metadata).transpose(1, 2)

    if gathered_length > sequence_length:
        padded = packed.new_zeros(
            packed.shape[0], packed.shape[1], gathered_length, packed.shape[3]
        )
        padded[:, :, :sequence_length] = packed
        packed = padded

    return packed, None


def _dense_fallback(query, key, value, call: AttnCall):
    from xfuser.core.attention import registry

    aiter = registry.REGISTRY.get(AttentionBackendType.AITER)
    if aiter is not None and aiter.unavailable() is None:
        return aiter.resolved()(query, key, value, call)

    from xfuser.core.attention.backends.sdpa.kernel import sdpa

    return sdpa(query, key, value, call)


