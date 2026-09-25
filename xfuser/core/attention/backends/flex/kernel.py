"""flex_block_attn: SSTA and Sparge block masks through FlexAttention."""

import math

import torch

from xfuser.core.attention.numerics.layout import make_contiguous
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.ssta import (
    get_sparse_mask,
    setup_ssta,
    untile_ssta_output,
)

from .spec import _FLEX_BLOCK_ATTN

# flex_block_attn is a separate third-party package, so it really can be
# absent
_FLEX_BLOCK_ATTN_FUNC = resolve(_FLEX_BLOCK_ATTN)

if _FLEX_BLOCK_ATTN_FUNC is not None:
    # Wrapped as a custom op so Dynamo treats the call as opaque instead of
    # tracing into the Triton kernel behind it.
    @torch.library.custom_op("xfuser::flex_block_attn", mutates_args=())
    def flex_block_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_m: int,
        block_n: int,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _FLEX_BLOCK_ATTN_FUNC(q, k, v, block_m, block_n, block_mask)

    @flex_block_attn.register_fake
    def _(q, k, v, block_m, block_n, block_mask):
        return torch.empty_like(q)


def flex_block(query, key, value, call: AttnCall):
    """SSTA tile mask through FlexAttention."""
    kwargs = call.attention_kwargs
    kwargs["sp_size"] = call.ctx.ulysses_world_size
    block_size = math.prod(kwargs["tile_size"])

    q, k, v, mask_config, state = setup_ssta(query, key, value, kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=kwargs["attn_sparse_type"])
    output = flex_block_attn(q, k, v, block_size, block_size, block_mask)
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
    output = flex_block_attn(
        *make_contiguous(q, k, v), block, block, block_mask
    )
    return restore_sparge_output(output, state), None
