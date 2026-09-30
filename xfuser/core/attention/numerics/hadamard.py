"""Normalised Hadamard matrices, shared by the fp8 and Sage v2 rotations.

Rotating Q and K by the same orthonormal R leaves Q @ K.T unchanged, so the
kernel sees identical scores while outliers are spread across the head
dimension, cutting quantisation error.
"""

import functools

import torch

from xfuser.core.attention.requirements import FIRST_OF

# The symbol moved modules between versions; both spellings are alive in the
# wild. FIRST_OF gates on either resolving and hands back whichever does, so
# backends can require this without repeating the paths.
CREATE_HADAMARD = FIRST_OF(
    "aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4:create_hadamard_matrix",
    "aiter.ops.triton.quant.sage_attention_quant_wrappers:create_hadamard_matrix",
)

# Resolved by prepare(). Not at import: this module is reached from the spec
# modules, which load on any machine, and resolving would import a vendor
# library that may not be there.
_create = None


def prepare(block_sizes=(128,), device=None) -> None:
    """Resolve the vendor symbol and build the matrices a kernel will need.

    Called from kernel modules at import -- which is backend selection, and so
    outside every compiled region. It has to happen there because resolution
    goes through importlib, which Dynamo refuses to trace: reaching it from
    inside a graph is a hard failure under fullgraph=True.

    matrix() being lru_cached is what makes that dangerous rather than
    obvious. A warm cache hides the import entirely, so whether a model
    compiles depends on what ran before it and warmed the entry it needs. Warm
    it deliberately instead.

    ``block_sizes`` are the block widths the caller's rotation can ask for;
    rotate_qk uses 128 for every head dimension that is a multiple of it.
    """
    global _create
    if _create is None:
        _create = CREATE_HADAMARD.resolve()
    if device is None:
        if not torch.cuda.is_available():
            return
        device = f"cuda:{torch.cuda.current_device()}"
    for block_r in block_sizes:
        matrix(block_r, device)


@functools.lru_cache(maxsize=None)
def matrix(block_r: int, device_key: str) -> torch.Tensor:
    """Orthonormal block_r x block_r matrix on the given device."""
    create = _create if _create is not None else CREATE_HADAMARD.resolve()
    built = create(block_r, dtype=torch.bfloat16) / (block_r**0.5)
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
