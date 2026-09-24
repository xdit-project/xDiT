"""AITER Sage: two kernel versions crossed with three mask sources.

        no mask          SSTA                  Sparge
  v1    AITER_SAGE       AITER_SPARSE_SAGE     AITER_SPARGE
  v2    AITER_SAGE_V2    AITER_SPARSE_SAGE_V2  AITER_SPARGE_V2

The versions differ in which wrapper they call and, together, in every
behavioural respect -- v2 rotates Q/K, wants contiguous inputs and is passed
`causal`; v1 does none of those. One flag, `v2`, carries all of it.
"""

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.constraints import (
    HEAD_DIM,
    NO_VARLEN,
    SELF_ATTENTION,
)
from xfuser.core.attention.requirements import PARAM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec
from xfuser.envs import environment_variables

_SAGE_V1 = "aiter.ops.triton.attention.fav3_sage:fav3_sage_wrapper_func"
_SAGE_V2 = (
    "aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper"
    ":fav3_sage_mxfp4_wrapper"
)
_RAGGED_LUT = "aiter.ops.triton.attention.utils:block_attn_mask_to_ragged_lut"

# SSTA drives the Triton kernels at a fixed block size; a coarser model tile is
# expanded up to it.
TRITON_SSTA_BLOCK = 128


def _block_r() -> int:
    """Sage v2's Hadamard block size. Shared with the kernel so the head-dim
    guard below and the matrix it guards cannot disagree."""
    try:
        value = int(environment_variables["AITER_SAGE_V2_BLOCK_R"]())
    except (TypeError, ValueError):
        return 128
    return value if value in (16, 32, 64, 128) else 128


BLOCK_R = _block_r()

V1 = SYMBOL(_SAGE_V1)
V2 = SYMBOL(_SAGE_V2) & hadamard.CREATE_HADAMARD
LUT = SYMBOL(_RAGGED_LUT)

# The rotation matrix is BLOCK_R wide and the kernel reads a full block per
# head, so a smaller head dimension reads past its end: two allocations of the
# same matrix give different results. The legacy path has no guard and returns
# whatever was in adjacent memory.
V2_CALLS = HEAD_DIM(BLOCK_R)

# Both masked sources reorder Q and K/V against one spatial layout, so a
# cross-attention call indexes K/V out of bounds. The legacy path has no such
# guard and cores the process rather than raising.
MASKED_CALLS = SELF_ATTENTION & NO_VARLEN


SPECS = [
    # NOTE: v1's wrapper accepts `causal`, but the legacy backend never passed
    # it -- a causal request silently returns non-causal output. Ported
    # unchanged; declaring NON_CAUSAL here is the one-line fix.
    Spec(
        AttentionBackendType.AITER_SAGE,
        impl=Impl("kernel:sage"),
        returns_lse=True,
        low_precision=True,
        accepts=NO_VARLEN,
        requires=V1,
    ),
    Spec(
        AttentionBackendType.AITER_SPARSE_SAGE,
        impl=Impl("kernel:sparse_sage"),
        sparsity="ssta",
        low_precision=True,
        accepts=MASKED_CALLS,
        requires=V1 & LUT,
    ),
    Spec(
        AttentionBackendType.AITER_SPARGE,
        impl=Impl("kernel:sparge"),
        sparsity="sparge",
        head_balanced=True,
        low_precision=True,
        accepts=MASKED_CALLS,
        requires=V1 & LUT & PARAM(_SAGE_V1, "block_lut"),
    ),
    Spec(
        AttentionBackendType.AITER_SAGE_V2,
        impl=Impl("kernel:sage_v2"),
        returns_lse=True,
        low_precision=True,
        accepts=V2_CALLS & NO_VARLEN,
        requires=V2,
    ),
    Spec(
        AttentionBackendType.AITER_SPARSE_SAGE_V2,
        impl=Impl("kernel:sparse_sage_v2"),
        sparsity="ssta",
        low_precision=True,
        accepts=V2_CALLS & MASKED_CALLS,
        requires=V2 & LUT,
    ),
    Spec(
        AttentionBackendType.AITER_SPARGE_V2,
        impl=Impl("kernel:sparge_v2"),
        sparsity="sparge",
        head_balanced=True,
        low_precision=True,
        accepts=V2_CALLS & MASKED_CALLS,
        requires=V2 & LUT & PARAM(_SAGE_V2, "block_lut"),
    ),
]
