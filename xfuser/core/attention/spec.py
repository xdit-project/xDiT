"""Attention backend specifications.

A backend is a callable plus a small set of facts that subsystems *outside* the
attention layer need in order to reason about it. That set is deliberately
closed: a field belongs here only if something else in xDiT consumes it.

    returns_lse    ring attention merges per-rank outputs via the LSE
    requires       availability gating, and skip decisions in the test suite
    is_sparse      model capability checks in base_model
    head_balanced  the Ulysses head balancer in usp
    low_precision  the quality warning in runtime_state
    accepts        pre-call validation, and shape selection in the test suite

Everything else -- quantisation formats, tile sizes, drop rates, routing
conditions -- is private to the backend module that implements it.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch

from xfuser.core.attention.layout import VarlenPacking
from xfuser.core.attention.requirements import ALWAYS, Requirement
from xfuser.core.attention.constraints import ANY_CALL, CallConstraint


class AttentionBackendType(Enum):
    SDPA = "SDPA"
    SDPA_MATH = "SDPA with Math backend"
    SDPA_EFFICIENT = "SDPA with memory-efficient backend"
    SDPA_FLASH = "SDPA with FLASH backend"
    FLASH = "Flash Attention V2"
    CUDNN =  "cuDNN"
    FLASH_3 = "Flash Attention V3"
    FLASH_3_FP8 = "Flash Attention v3 FP8"
    NVTE_FP8 = "NVTE FP8"
    FLASH_4 = "Flash Attention V4"
    FLASH_4_FP4 = "Flash Attention V4 FP4"
    SAGE = "Sage Attention"
    FLEX_BLOCK_ATTN = "Flex Block Attention"
    AITER = "AITER"
    AITER_BF16 = "AITER BF16 MHA v4"
    AITER_BF16FP8 = "AITER BF16/FP8 MHA v4"
    AITER_MLA = "AITER MLA" # deprecated, use AITER_FP8
    AITER_I8FP8 = "AITER I8FP8"
    AITER_FP8 = "AITER FP8"
    AITER_MXFP8 = "AITER MXFP8"
    AITER_F8F6 = "AITER F8F6"
    AITER_MXFP6 = "AITER MXFP6"
    AITER_F6F4 = "AITER F6F4"
    AITER_MXFP4 = "AITER MXFP4"
    AITER_F4F4 = "AITER F4F4"
    AITER_I8FP8_SPARGE = "AITER I8FP8 Sparge"
    AITER_FP8_SPARGE = "AITER FP8 Sparge"
    AITER_MXFP8_SPARGE = "AITER MXFP8 Sparge"
    AITER_F8F6_SPARGE = "AITER F8F6 Sparge"
    AITER_MXFP6_SPARGE = "AITER MXFP6 Sparge"
    AITER_F6F4_SPARGE = "AITER F6F4 Sparge"
    AITER_MXFP4_SPARGE = "AITER MXFP4 Sparge"
    AITER_F4F4_SPARGE = "AITER F4F4 Sparge"
    AITER_SAGE = "AITER Sage"
    AITER_SPARSE_SAGE = "AITER Sparse Sage"
    AITER_SAGE_V2 = "AITER Sage V2"
    AITER_SPARSE_SAGE_V2 = "AITER Sparse Sage V2"
    AITER_SPARGE = "AITER Sparge"
    AITER_SPARGE_V2 = "AITER Sparge V2"
    AITER_VSA = "AITER VSA CK"
    FLEX_VSA_H3 = "Flex VSA-H3"
    FLEX_BLOCK_SPARGE = "Flex Block Sparge"
    AITER_FLYDSL = "AITER FlyDSL"
    AITER_FLYDSL_FP8 = "AITER FlyDSL FP8"
    NPU = "NPU"


@dataclass(frozen=True)
class ParallelContext:
    """Sequence-parallel degrees, passed in rather than read from globals.

    Backends previously called get_ulysses_parallel_world_size() and friends
    directly, which made them untestable without an initialised process group.
    """

    ulysses_world_size: int = 1
    ring_world_size: int = 1


@dataclass
class AttnCall:
    """Everything a backend needs about one call that is not q/k/v."""

    dropout_p: float = 0.0
    is_causal: bool = False
    varlen: Optional[VarlenPacking] = None
    ctx: ParallelContext = field(default_factory=ParallelContext)

    # Transitional: the untyped attention_kwargs dict carried through from the
    # model. Phase 6 replaces this with per-strategy typed config objects owned
    # by the sparsity strategy that reads them.
    attention_kwargs: dict = field(default_factory=dict)


# (output, softmax_lse) -- lse is None when the backend does not produce one.
AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, AttnCall], tuple]


@dataclass(frozen=True)
class Spec:
    """One named backend.

    ``returns_lse`` defaults to False: a backend that produces a softmax
    log-sumexp ring attention can merge on must say so. Conservative, because
    a wrong True yields silently incorrect ring output while a wrong False only
    forgoes ring. The old-vs-new equivalence suite checks this field against
    the legacy ring blocklist, so an omission surfaces as a failing test.

    ``requires`` defaults to ALWAYS -- nothing to check.
    """

    type: AttentionBackendType
    impl: AttnFn
    returns_lse: bool = False
    requires: Requirement = ALWAYS

    is_sparse: bool = False
    head_balanced: bool = False
    low_precision: bool = False
    accepts: CallConstraint = ANY_CALL

    def unavailable(self) -> Optional[str]:
        """Why this backend cannot run here, or None. Never says 'update X':
        the same missing symbol means too-old or too-new, and we cannot tell."""
        return self.requires.unmet()

    def rejects(self, query, key, value, call: AttnCall) -> Optional[str]:
        """Why this backend cannot serve this particular call, or None."""
        return self.accepts.unmet(query, key, value, call)

    def run(self, query, key, value, call: AttnCall):
        """Enforce ``accepts``, then dispatch. Callers use this rather than
        ``impl`` so the constraint is declared once and checked in one place;
        a kernel function stays pure kernel code."""
        reason = self.rejects(query, key, value, call)
        if reason is not None:
            raise NotImplementedError(f"{self.type.name} {reason}")
        return self.impl(query, key, value, call)
