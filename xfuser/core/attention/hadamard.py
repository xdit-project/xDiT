"""Normalised Hadamard matrices, shared by the fp8 and Sage v2 rotations.

Rotating Q and K by the same orthonormal R leaves Q @ K.T unchanged, so the
kernel sees identical scores while outliers are spread across the head
dimension, cutting quantisation error.
"""

import functools

import torch

# aiter-shim cut 2026-09: a local Sylvester construction (added 2026-06-18)
# stood in when create_hadamard_matrix was unavailable. Builds at or after the
# July floor all ship it, and a matrix that does not match what the kernel
# expects is worse than a clear failure.
SYMBOL_PATHS = (
    "aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4"
    ":create_hadamard_matrix",
    "aiter.ops.triton.quant.sage_attention_quant_wrappers:create_hadamard_matrix",
)


def _create():
    """The symbol moved modules between AITER versions; both spellings are
    alive in the wild."""
    try:
        from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
            create_hadamard_matrix,
        )
    except ImportError:
        from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
            create_hadamard_matrix,
        )
    return create_hadamard_matrix


@functools.lru_cache(maxsize=None)
def matrix(block_r: int, device_key: str) -> torch.Tensor:
    """Orthonormal block_r x block_r matrix on the given device."""
    built = _create()(block_r, dtype=torch.bfloat16) / (block_r ** 0.5)
    return built.to(torch.device(device_key))


def rotate(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rotate along head_dim, in blocks of r.shape[-1]."""
    head_dim = x.shape[-1]
    block_r = r.shape[-1]
    r = r.to(x.dtype)
    if block_r == head_dim:
        return torch.matmul(x, r)
    return torch.matmul(x.unflatten(-1, (head_dim // block_r, block_r)), r).flatten(-2)
