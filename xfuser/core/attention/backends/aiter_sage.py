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

import torch

from xfuser.core.attention.constraints import (
    ANY_CALL,
    HEAD_DIM,
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
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec
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
# hadamard (sage v2 only)
# ---------------------------------------------------------------------------

def _block_r() -> int:
    try:
        value = int(environment_variables["AITER_SAGE_V2_BLOCK_R"]())
    except (TypeError, ValueError):
        return 128
    return value if value in (16, 32, 64, 128) else 128


@functools.lru_cache(maxsize=None)
def _hadamard(device_key) -> torch.Tensor:
    """AITER's own matrix. Unlike the fp8 one there is no local fallback: sage
    v2's rotation must match what the kernel expects."""
    try:
        from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
            create_hadamard_matrix,
        )
    except ImportError:
        # The symbol moved modules; both spellings are alive in the wild, which
        # is what SYMBOL(a) | SYMBOL(b) declares below.
        from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
            create_hadamard_matrix,
        )

    block_r = _block_r()
    matrix = create_hadamard_matrix(block_r, dtype=torch.bfloat16) / (block_r ** 0.5)
    return matrix.to(torch.device(device_key))


_HADAMARD_SYMBOL = (
    SYMBOL(
        "aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4"
        ":create_hadamard_matrix"
    )
    | SYMBOL(
        "aiter.ops.triton.quant.sage_attention_quant_wrappers:create_hadamard_matrix"
    )
)


# ---------------------------------------------------------------------------
# kernel versions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SageKernel:
    name: str
    wrapper: str            # "module:function"
    configs: str            # "module:function"
    hadamard: bool
    force_contiguous: bool
    passes_causal: bool
    requires: Requirement
    accepts: CallConstraint = ANY_CALL

    def resolve(self):
        module, _, symbol = self.wrapper.partition(":")
        return getattr(__import__(module, fromlist=[symbol]), symbol)

    def config(self) -> dict:
        module, _, symbol = self.configs.partition(":")
        return getattr(__import__(module, fromlist=[symbol]), symbol)()

    def extra(self, query) -> dict:
        if not self.hadamard:
            return {}
        return {"hadamard_rotation": True, "R": _hadamard(str(query.device))}

    def prepare(self, *tensors):
        # aiter-shim: sage v2 needed contiguous inputs in older builds. Kept
        # because no version is recorded for the fix and dropping it silently
        # changes numerics rather than raising.
        if not self.force_contiguous:
            return tensors
        return tuple(t.contiguous() for t in tensors)


V1 = SageKernel(
    name="", wrapper=_SAGE_V1,
    configs="aiter.ops.triton.attention.fav3_sage:get_sage_fwd_configs",
    hadamard=False, force_contiguous=False,
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
    hadamard=True, force_contiguous=True, passes_causal=True,
    requires=SYMBOL(_SAGE_V2) & _HADAMARD_SYMBOL,
    # The rotation matrix is block_r wide and the kernel reads a full block per
    # head, so a smaller head dimension reads past its end: two allocations of
    # the same matrix give different results. The legacy path has no guard and
    # returns whatever was in adjacent memory.
    accepts=HEAD_DIM(_block_r()),
)


def _causal(kernel: SageKernel, call: AttnCall) -> dict:
    return {"causal": call.is_causal} if kernel.passes_causal else {}


# ---------------------------------------------------------------------------
# the three mask sources
# ---------------------------------------------------------------------------

def dense(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """No mask. The only cell that can participate in ring attention, and only
    then does the wrapper produce an LSE."""
    q, k, v = kernel.prepare(query, key, value)
    attn = kernel.resolve()
    extra = kernel.extra(q)

    if call.ctx.ring_world_size > 1:
        # aiter-shim cut 2026-09: AITER_SAGE_SUPPORTS_RING /
        # AITER_SAGE_V2_SUPPORTS_RING (both added 2026-06-17) probed for these
        # parameters. Every build at or after the July floor has them.
        lse_args = {"return_lse": True}
        if not kernel.hadamard:
            lse_args["smooth_k"] = True
        return attn(q, k, v, layout="bhsd", **extra, **lse_args, **_causal(kernel, call))

    return attn(q, k, v, layout="bhsd", **extra, **_causal(kernel, call)), None


def ssta(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """Tile-based static mask, supplied by the model's sparse config."""
    from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

    kwargs = call.attention_kwargs
    kwargs["sp_size"] = call.ctx.ulysses_world_size
    block_size = math.prod(kwargs["tile_size"])

    config = kernel.config()
    config["BLOCK_M"] = _TRITON_SSTA_BLOCK
    config["BLOCK_N"] = _TRITON_SSTA_BLOCK

    q, k, v, mask_config, state = setup_ssta(query, key, value, kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=kwargs["attn_sparse_type"])
    if block_size != _TRITON_SSTA_BLOCK:
        block_mask = expand_block_mask(block_mask, factor=block_size // _TRITON_SSTA_BLOCK)

    output = kernel.resolve()(
        q, k, v,
        layout="bhsd", config=config, **kernel.extra(q),
        block_lut=block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1]),
        **_causal(kernel, call),
    )
    output = untile_ssta_output(
        output, state, kwargs["encoder_sequence_length"], kwargs["sp_size"]
    )
    return output, None


def sparge(query, key, value, call: AttnCall, *, kernel: SageKernel):
    """Data-dependent mask computed from Q/K."""
    from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

    query, key, value = kernel.prepare(query, key, value)
    config = kernel.config()

    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=config["BLOCK_M"], block_n=config["BLOCK_N"],
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
    )

    output = kernel.resolve()(
        q, k, v,
        layout="bhsd", config=config, **kernel.extra(q),
        block_lut=block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1]),
        **_causal(kernel, call),
    )
    return restore_sparge_output(output, state), None


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

SPECS = []
for _kernel in (V1, V2):
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SAGE{_kernel.name}"],
        impl=functools.partial(dense, kernel=_kernel),
        returns_lse=True,
        low_precision=True,
        accepts=_kernel.accepts,
        requires=_kernel.requires,
    ))
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SPARSE_SAGE{_kernel.name}"],
        impl=functools.partial(ssta, kernel=_kernel),
        is_sparse=True,
        accepts=SELF_ATTENTION,
        low_precision=True,
        requires=_kernel.requires & SYMBOL(_RAGGED_LUT),
    ))
    SPECS.append(Spec(
        AttentionBackendType[f"AITER_SPARGE{_kernel.name}"],
        impl=functools.partial(sparge, kernel=_kernel),
        is_sparse=True,
        head_balanced=True,
        # Both mask sources reorder Q and K/V against one spatial layout, so a
        # cross-attention call indexes K/V out of bounds. The legacy path has
        # no such guard and cores the process rather than raising.
        accepts=_kernel.accepts & SELF_ATTENTION,
        low_precision=True,
        requires=(
            _kernel.requires
            & SYMBOL(_RAGGED_LUT)
            & PARAM(_kernel.wrapper, "block_lut")
        ),
    ))
