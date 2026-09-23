"""AITER Sage: two kernel versions crossed with three mask sources.

        no mask          SSTA                  Sparge
  v1    AITER_SAGE       AITER_SPARSE_SAGE     AITER_SPARGE
  v2    AITER_SAGE_V2    AITER_SPARSE_SAGE_V2  AITER_SPARGE_V2

The two versions differ only in which wrapper and config they use and whether
they need a Hadamard rotation; the three mask sources differ only in how the
block mask is produced and undone. Six enum members, two axes.
"""

import functools
import math
from dataclasses import dataclass
from typing import Optional

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.constraints import (
    ANY_CALL,
    HEAD_DIM,
    NO_VARLEN,
    SELF_ATTENTION,
    CallConstraint,
)
from xfuser.core.attention.requirements import PARAM, SYMBOL, Requirement
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec
from xfuser.core.distributed.ssta import (
    expand_block_mask,
    get_sparse_mask,
    setup_ssta,
    untile_ssta_output,
)
from xfuser.envs import environment_variables

_SAGE_V1 = "aiter.ops.triton.attention.fav3_sage:fav3_sage_wrapper_func"
_SAGE_V2 = (
    "aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper"
    ":fav3_sage_mxfp4_wrapper"
)
_RAGGED_LUT = "aiter.ops.triton.attention.utils:block_attn_mask_to_ragged_lut"

# SSTA drives the Triton kernels at a fixed block size; a coarser model tile is
# expanded up to it.
_TRITON_SSTA_BLOCK = 128


# ---------------------------------------------------------------------------
# rotation block size (sage v2 only; the matrix itself is shared, see
# xfuser/core/attention/numerics/hadamard.py)
# ---------------------------------------------------------------------------

def _block_r() -> int:
    try:
        value = int(environment_variables["AITER_SAGE_V2_BLOCK_R"]())
    except (TypeError, ValueError):
        return 128
    return value if value in (16, 32, 64, 128) else 128


# ---------------------------------------------------------------------------
# kernel versions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SageKernel:
    name: str
    wrapper: str            # "module:function"
    configs: str            # "module:function"
    rotates: bool          # Hadamard-rotate Q/K (sage v2 only)
    force_contiguous: bool
    passes_causal: bool
    requires: Requirement
    accepts: CallConstraint = ANY_CALL

    def prepare(self, *tensors):
        # aiter-shim: added 2026-03-12. Sage v2 needed contiguous inputs in
        # older builds. Pre-floor, so it is a removal candidate -- but unlike
        # the probe-guarded shims this is an unconditional copy, and "AITER no
        # longer needs it" cannot be established by introspection, only by a
        # bitwise run with and without. Kept until that is measured.
        if not self.force_contiguous:
            return tensors
        return tuple(t.contiguous() for t in tensors)


V1 = SageKernel(
    name="", wrapper=_SAGE_V1,
    configs="aiter.ops.triton.attention.fav3_sage:get_sage_fwd_configs",
    rotates=False, force_contiguous=False,
    # NOTE: the wrapper accepts `causal`, but the legacy backend never passed
    # it -- a causal request silently returns non-causal output. Ported
    # unchanged. Declaring NON_CAUSAL in `accepts` below is the one-line fix.
    passes_causal=False,
    requires=SYMBOL(_SAGE_V1),
)

V2 = SageKernel(
    name="_V2", wrapper=_SAGE_V2,
    configs=(
        "aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper"
        ":get_sage_fwd_configs_mxfp4"
    ),
    rotates=True, force_contiguous=True, passes_causal=True,
    requires=SYMBOL(_SAGE_V2) & hadamard.CREATE_HADAMARD,
    # The rotation matrix is block_r wide and the kernel reads a full block per
    # head, so a smaller head dimension reads past its end: two allocations of
    # the same matrix give different results. The legacy path has no guard and
    # returns whatever was in adjacent memory.
    accepts=HEAD_DIM(_block_r()),
)


SPECS = []
for _kernel in (V1, V2):
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SAGE{_kernel.name}"],
        impl=Impl("kernel:dense", {"kernel": _kernel}),
        returns_lse=True,
        low_precision=True,
        accepts=_kernel.accepts & NO_VARLEN,
        requires=_kernel.requires,
    ))
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SPARSE_SAGE{_kernel.name}"],
        impl=Impl("kernel:ssta", {"kernel": _kernel}),
        sparsity="ssta",
        accepts=_kernel.accepts & SELF_ATTENTION & NO_VARLEN,
        low_precision=True,
        requires=_kernel.requires & SYMBOL(_RAGGED_LUT),
    ))
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SPARGE{_kernel.name}"],
        impl=Impl("kernel:sparge", {"kernel": _kernel}),
        sparsity="sparge",
        head_balanced=True,
        # Both mask sources reorder Q and K/V against one spatial layout, so a
        # cross-attention call indexes K/V out of bounds. The legacy path has
        # no such guard and cores the process rather than raising.
        accepts=_kernel.accepts & SELF_ATTENTION & NO_VARLEN,
        low_precision=True,
        requires=(
            _kernel.requires
            & SYMBOL(_RAGGED_LUT)
            & PARAM(_kernel.wrapper, "block_lut")
        ),
    ))
