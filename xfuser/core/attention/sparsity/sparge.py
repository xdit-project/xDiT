"""Sparge: data-dependent block masking.

The Ulysses degree is a parameter rather than a global read, so a mask can be
built without a process group.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from xfuser.core.sparge_attention.head_balance import COST_SINK_KEY
from xfuser.core.sparge_attention.sparge import (
    compute_sparge_block_mask,
    mask_padded_kv_blocks,
    restore_sparge_output,
    setup_sparge,
)


@dataclass(frozen=True)
class SpargeConfig:
    """Sparge knobs. Defaults match xFuserArgs, which is where they come from
    in a real run; they are repeated here only so the type is constructible."""

    thw: Optional[tuple] = None
    encoder_sequence_length: int = 0
    simthreshold: float = 0.3
    cdfthreshold: float = 0.92
    reorder_sequence: bool = True
    use_static_block_mask: bool = True

    @classmethod
    def from_kwargs(cls, attention_kwargs: Optional[dict]) -> "SpargeConfig":
        kwargs = attention_kwargs or {}
        return cls(
            thw=kwargs.get("thw"),
            encoder_sequence_length=kwargs.get("encoder_sequence_length", 0),
            simthreshold=kwargs.get("spargeattn_simthreshold", 0.3),
            cdfthreshold=kwargs.get("spargeattn_cdfthreshold", 0.92),
            reorder_sequence=kwargs.get("spargeattn_reorder_sequence", True),
            use_static_block_mask=kwargs.get("use_spargeattn_static_block_mask", True),
        )


def build_block_mask(
    query,
    key,
    value,
    *,
    is_causal: bool,
    config: SpargeConfig,
    block_m: int,
    block_n: int,
    ulysses_world_size: int,
    cost_sink: Optional[torch.Tensor] = None,
    pad_block_divisible: bool = False,
):
    """Reorder/pad q/k/v and compute the block mask. Returns the transformed
    tensors, the state needed to undo the transform, and the mask."""
    q, k, v, state, static_mask = setup_sparge(
        query, key, value,
        thw=config.thw,
        sp_size=ulysses_world_size,
        encoder_sequence_length=config.encoder_sequence_length,
        reorder_sequence=config.reorder_sequence,
        use_static_block_mask=config.use_static_block_mask,
        block_m=block_m, block_n=block_n,
        pad_block_divisible=pad_block_divisible,
    )
    block_mask = compute_sparge_block_mask(
        q, k,
        simthreshd1=config.simthreshold,
        cdfthreshd=config.cdfthreshold,
        is_causal=is_causal,
        static_block_mask=static_mask,
        text_len=state.text_len + state.tail_pad,
        block_m=block_m, block_n=block_n,
    )
    block_mask = mask_padded_kv_blocks(block_mask, state, block_n)

    # Per-head selected-block cost for the Ulysses head balancer. USP injects
    # the sink only when balancing is active.
    if cost_sink is not None:
        cost_sink.copy_(block_mask.to(torch.float32).sum(dim=(0, 2, 3)))

    return q, k, v, state, block_mask


def cost_sink_from(attention_kwargs: Optional[dict]) -> Optional[torch.Tensor]:
    return (attention_kwargs or {}).get(COST_SINK_KEY)


__all__ = ["SpargeConfig", "build_block_mask", "cost_sink_from", "restore_sparge_output"]
