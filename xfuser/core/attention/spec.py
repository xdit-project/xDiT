"""Attention backend specifications.

A backend is a callable plus a small set of facts that subsystems *outside* the
attention layer need in order to reason about it. That set is deliberately
closed: a field belongs here only if something else in xDiT consumes it.
"""

import functools
import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch

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
class VarlenPacking:
    """Per-call key packing supplied by the model.

    ``indices_k`` selects the surviving K/V rows out of a flattened B*S; the
    cumulative lengths and maximum describe the packed result.
    """

    indices_k: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_k: int

    @classmethod
    def from_kwargs(cls, attention_kwargs: Optional[dict]) -> Optional["VarlenPacking"]:
        kwargs = attention_kwargs or {}
        indices_k = kwargs.get("indices_k")
        if indices_k is None:
            return None
        return cls(
            indices_k=indices_k,
            cu_seqlens_k=kwargs["cu_seqlens_k"],
            max_seqlen_k=kwargs["max_seqlen_k"],
        )


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
class Impl:
    """A reference to a kernel function, resolved when the backend is selected.

    A kernel module imports its vendor library at module level, so it can only
    be imported once that library is known present. Resolving here rather than
    on first dispatch keeps the import out of any compiled region: Dynamo
    refuses to trace importlib, which makes a lazy import a hard failure under
    fullgraph=True (tests/attention/test_lazy_op_registration.py pins this).

    ``target`` is "module:function" relative to the backend package. ``bound``
    is applied with functools.partial, which is how the generated families bind
    a table row to a shared launcher.
    """

    target: str
    bound: dict = field(default_factory=dict)

    def resolve(self, package: str) -> AttnFn:
        module_name, _, symbol = self.target.partition(":")
        module = importlib.import_module(f"{package}.{module_name}")
        fn = getattr(module, symbol)
        return functools.partial(fn, **self.bound) if self.bound else fn


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

    # The enum member this spec answers to. The registry keys on it, and it is
    # what --attention_backend names on the command line.
    type: AttentionBackendType

    # The kernel. Written as an Impl("module:function") relative to the backend
    # package; the registry turns it into the callable when the backend is
    # selected.
    impl: AttnFn

    # Whether the second return value is a softmax log-sumexp that ring
    # attention can merge across ranks. runtime_state refuses the backend when
    # ring_degree > 1 and this is False.
    returns_lse: bool = False

    # What the machine must provide. Checked once, at backend selection, and a
    # failure names the missing piece rather than crashing mid-denoising.
    requires: Requirement = ALWAYS

    # Which sparsity strategy, if any: "ssta", "sparge", "vsa", "h3".
    # A kind rather than a flag because the consumers distinguish them --
    # base_model gates SSTA and sparge separately, and they are not
    # interchangeable for a given model.
    sparsity: Optional[str] = None

    # The kernel writes per-head cost into the head-balance cost sink, so usp
    # can even out the Ulysses split. True only where the kernel actually does
    # it; elsewhere head balancing is a no-op.
    head_balanced: bool = False

    # The kernel quantises internally (fp8, mxfp8, int8, mxfp4...), which
    # runtime_state warns about at startup since it costs output quality.
    low_precision: bool = False

    # fp8 comms quantises Q/K/V before the Ulysses all-to-all and hands the
    # kernel fp8 plus descales. Both facts belong to the backend: whether it
    # can consume that, and what rotation the comms layer must apply first so
    # calibration measures the distribution that is actually quantised.
    accepts_prequantized: bool = False
    prequant_rotate: Optional[Callable] = None

    # Which calls the kernel can serve -- head dim, causality, varlen packing,
    # dropout, self- vs cross-attention. Enforced by run() before dispatch, so
    # an unsupported call raises with a reason instead of computing something
    # wrong. Anything not declared here is silently accepted.
    accepts: CallConstraint = ANY_CALL

    # Filled in by the registry from the module the spec came from, so Impl
    # targets can be written relative to the backend package.
    package: str = ""

    # The resolved kernel, cached by resolved() so dispatch is not an import.
    _resolved: Optional[AttnFn] = None

    @property
    def is_sparse(self) -> bool:
        return self.sparsity is not None

    def unavailable(self) -> Optional[str]:
        """Why this backend cannot run here, or None. Never says 'update X':
        the same missing symbol means too-old or too-new, and we cannot tell."""
        return self.requires.unmet()

    def rejects(self, query, key, value, call: AttnCall) -> Optional[str]:
        """Why this backend cannot serve this particular call, or None."""
        return self.accepts.unmet(query, key, value, call)

    def resolved(self) -> AttnFn:
        """The callable, importing the kernel module if it has not been yet.

        Called when a backend is selected, never from the hot path -- see Impl.
        """
        if isinstance(self.impl, Impl):
            fn = self.impl.resolve(self.package)
            object.__setattr__(self, "_resolved", fn)
            return fn
        return self.impl

    def run(self, query, key, value, call: AttnCall):
        """Enforce ``accepts``, then dispatch. Callers use this rather than
        ``impl`` so the constraint is declared once and checked in one place;
        a kernel function stays pure kernel code."""
        reason = self.rejects(query, key, value, call)
        if reason is not None:
            # Raised from inside the traced region this surfaces as
            # torch._dynamo.exc.Unsupported; the message survives in the debug
            # context. Only reachable on a misconfigured run.
            raise NotImplementedError(f"{self.type.name} {reason}")
        return (self._resolved or self.resolved())(query, key, value, call)
