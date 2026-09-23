"""Normalised Hadamard matrices, shared by the fp8 and Sage v2 rotations.

Rotating Q and K by the same orthonormal R leaves Q @ K.T unchanged, so the
kernel sees identical scores while outliers are spread across the head
dimension, cutting quantisation error.
"""

import functools

import torch

from xfuser.core.attention.requirements import FIRST_OF

# aiter-shim cut 2026-09: a local Sylvester construction (added 2026-06-18)
# stood in when create_hadamard_matrix was unavailable. Builds at or after the
# July floor all ship it, and a matrix that does not match what the kernel
# expects is worse than a clear failure.
#
# The symbol moved modules between versions; both spellings are alive in the
# wild. FIRST_OF gates on either resolving and hands back whichever does, so
# backends can require this without repeating the paths.
CREATE_HADAMARD = FIRST_OF(
    "aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4"
    ":create_hadamard_matrix",
    "aiter.ops.triton.quant.sage_attention_quant_wrappers:create_hadamard_matrix",
)


@functools.lru_cache(maxsize=None)
def matrix(block_r: int, device_key: str) -> torch.Tensor:
    """Orthonormal block_r x block_r matrix on the given device."""
    built = CREATE_HADAMARD.resolve()(block_r, dtype=torch.bfloat16) / (block_r ** 0.5)
    return built.to(torch.device(device_key))


def rotate(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rotate along head_dim, in blocks of r.shape[-1]."""
    head_dim = x.shape[-1]
    block_r = r.shape[-1]
    r = r.to(x.dtype)
    if block_r == head_dim:
        return torch.matmul(x, r)
    return torch.matmul(x.unflatten(-1, (head_dim // block_r, block_r)), r).flatten(-2)


def rotate_qk(query, key):
    """Rotate Q and K by a shared matrix ahead of fp8 quantisation.

    128-blocked when head_dim is a multiple of 128 (every current model);
    full-head for smaller power-of-two dims such as LTX-2.5 audio. Q and K must
    be rotated identically or Q @ K.T changes.

    Both the quantisation site in USP and the calibration site that freezes the
    per-layer scale call this: they have to measure and quantise the same
    distribution, or the scale describes a tensor that is never quantised.
    """
    head_dim = query.shape[-1]
    block_r = 128 if head_dim % 128 == 0 else head_dim
    r = matrix(block_r, str(query.device))
    return rotate(query, r).contiguous(), rotate(key, r).contiguous()
