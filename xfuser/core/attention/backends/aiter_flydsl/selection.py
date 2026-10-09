"""Empirically calibrated kernel selection for AITER FlyDSL attention."""

import torch


_FP8_CROSSOVERS = {
    # (head_dim, num_heads): (self-attention, patch-Q/full-KV)
    (64, 19): (4096, None),
    (64, 38): (3584, None),
    (128, 6): (4352, 2304),
    (128, 8): (4352, 2304),
    (128, 12): (3584, 2304),
    (128, 15): (3584, 2304),
    (128, 16): (3584, 2304),
    (128, 24): (2688, 1920),
    (128, 30): (3072, 2048),
    (128, 32): (3072, 2048),
    (128, 48): (2688, 2048),
}


def fp8_min_seq(head_dim: int, num_heads: int, *, is_cross: bool = False) -> int | None:
    """Return the measured gfx1201 FP8 crossover, if this exact shape was calibrated."""
    crossovers = _FP8_CROSSOVERS.get((head_dim, num_heads))
    if crossovers is None:
        return None
    return crossovers[is_cross]


def fp8_eligible(
    dtype: torch.dtype,
    query_length: int,
    key_length: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    if dtype != torch.bfloat16:
        return False
    # Ordinary cross-attention has short conditioning K/V and does not
    # amortize quantization. PipeFusion patch attention has patch Q against
    # a longer cached image K/V and can cross over earlier than self-attention.
    if key_length < query_length:
        return False
    minimum = fp8_min_seq(
        head_dim,
        num_heads,
        is_cross=key_length > query_length,
    )
    return minimum is not None and query_length >= minimum
