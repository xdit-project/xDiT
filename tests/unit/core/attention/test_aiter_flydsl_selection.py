"""Regression coverage for the measured gfx1201 FlyDSL crossover policy.

The boundaries were measured with ``benchmarks/aiter_flydsl_crossover.py`` on
gfx1201 using BF16 inputs, non-causal attention, and 20 post-warmup samples.
FP8 timings include the Q/K/V quantization pass used by the runtime.
"""

import pytest
import torch

from xfuser.core.attention.backends.aiter_flydsl.selection import fp8_eligible


@pytest.mark.parametrize(
    ("dtype", "query_length", "key_length", "num_heads", "head_dim", "expected"),
    [
        (torch.bfloat16, 2687, 2687, 24, 128, False),
        (torch.bfloat16, 2688, 2688, 24, 128, True),
        (torch.bfloat16, 1919, 6912, 24, 128, False),
        (torch.bfloat16, 1920, 6912, 24, 128, True),
        (torch.bfloat16, 1280, 4352, 24, 128, False),
        (torch.bfloat16, 4096, 256, 24, 128, False),
        (torch.bfloat16, 4351, 4351, 6, 128, False),
        (torch.bfloat16, 4352, 4352, 6, 128, True),
        (torch.bfloat16, 2304, 8448, 6, 128, True),
        (torch.bfloat16, 3583, 3583, 12, 128, False),
        (torch.bfloat16, 3584, 3584, 12, 128, True),
        (torch.bfloat16, 2304, 8448, 16, 128, True),
        (torch.bfloat16, 3071, 3071, 30, 128, False),
        (torch.bfloat16, 3072, 3072, 30, 128, True),
        (torch.bfloat16, 2048, 7424, 32, 128, True),
        (torch.bfloat16, 2687, 2687, 48, 128, False),
        (torch.bfloat16, 2688, 2688, 48, 128, True),
        (torch.bfloat16, 4095, 4095, 19, 64, False),
        (torch.bfloat16, 4096, 4096, 19, 64, True),
        (torch.bfloat16, 3583, 3583, 38, 64, False),
        (torch.bfloat16, 3584, 3584, 38, 64, True),
        (torch.bfloat16, 4352, 16640, 38, 64, False),
        (torch.float16, 4352, 16640, 24, 128, False),
        (torch.bfloat16, 8192, 8192, 20, 128, False),
        (torch.bfloat16, 8192, 8192, 24, 96, False),
    ],
)
def test_flydsl_fp8_shape_eligibility(
    dtype,
    query_length,
    key_length,
    num_heads,
    head_dim,
    expected,
):
    assert (
        fp8_eligible(
            dtype,
            query_length,
            key_length,
            num_heads,
            head_dim,
        )
        is expected
    )
