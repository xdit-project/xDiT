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
    AITER_I8FP8 = "AITER I8FP8"
    AITER_FP8 = "AITER FP8"
    AITER_MXFP8 = "AITER MXFP8"
    AITER_F8F6 = "AITER F8F6"
    AITER_MXFP6 = "AITER MXFP6"
    AITER_F6F4 = "AITER F6F4"
    AITER_MXFP4 = "AITER MXFP4"
    AITER_F4F4 = "AITER F4F4"
    AITER_BF16_SPARGE = "AITER BF16 Sparge"
    AITER_BF16FP8_SPARGE = "AITER BF16/FP8 Sparge"
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
    TRITON_VSA_H3 = "FastH3 VSA-H3 (Triton)"
    FLEX_BLOCK_SPARGE = "Flex Block Sparge"
    AITER_FLYDSL = "AITER FlyDSL"
    AITER_FLYDSL_FP8 = "AITER FlyDSL FP8"
    NPU = "NPU"


class Sparsity(Enum):
    """Which sparsity strategy a backend implements, for the consumers that
    gate on it.

    A closed set rather than a free string: consumers select by exact match --
    base_model gates SSTA apart from sparge -- so a mistyped strategy is not an
    error but a backend that quietly belongs to no group and is never gated.
    """

    SSTA = "ssta"
    SPARGE = "sparge"
    VSA = "vsa"
    H3 = "h3"


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


@dataclass
class AttnCall:
    """Everything a backend needs about one call that is not q/k/v."""

    dropout_p: float = 0.0
    is_causal: bool = False
    varlen: Optional[VarlenPacking] = None

    # Sequence-parallel degrees, passed in rather than read from globals: a
    # backend calling get_ulysses_parallel_world_size() directly could not be
    # exercised without an initialised process group.
    #
    # Flat integers rather than a ParallelContext holding them, because Dynamo
    # cannot trace a read of a user-defined object that a default_factory
    # produced. It has no source for one it did not watch being built, and its
    # sourceless builder fails on it -- with "AttributeError: 'NoneType'
    # object has no attribute 'name'", which says nothing about the cause.
    # Two AttnCalls that compare equal would then compile differently,
    # depending only on whether the caller spelled the default out.
    ulysses_world_size: int = 1
    ring_world_size: int = 1

    # The untyped dict the model passes through. Each sparsity strategy digs
    # out its own keys; a typed config per strategy would be better.
    #
    # A default_factory is safe here where it is not above: dict is a builtin
    # Dynamo constructs natively, so there is no user-defined class to wrap.
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
    fullgraph=True. Pinned by the compile cases in tests/test_aiter_mixed_
    attention.py and tests/test_minimax_h3.py, both -k compiles_fullgraph.

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

    ``requires``, ``accepts`` and ``ring`` have no default. Each is a claim
    about what a backend will tolerate, and the values that could serve as
    defaults -- ALWAYS and ANY_CALL -- are the permissive ones, so an omission
    would read as "runs anywhere, serves anything" rather than as an omission.
    Writing ALWAYS is no more work than leaving it out and says it was decided.
    """

    # The enum member this spec answers to. The registry keys on it, and it is
    # what --attention_backend names on the command line.
    type: AttentionBackendType

    # The kernel. Written as an Impl("module:function") relative to the backend
    # package; the registry turns it into the callable when the backend is
    # selected.
    impl: AttnFn

    # When this backend may take part in ring attention, which merges per-rank
    # partials on a softmax log-sumexp. NEVER for a kernel that returns none at
    # all; a predicate where the LSE depends on the build or the device, as on
    # MHA v4. runtime_state reports the unmet reason, so a refusal explains
    # itself rather than saying only that ring is unavailable.
    #
    # Be conservative: a backend whose LSE is wrong yields silently incorrect
    # ring output, and O never reads the LSE, so no output check can see it.
    ring: Requirement = field(kw_only=True)

    # What the machine must provide. Checked once, at backend selection, and a
    # failure names the missing piece rather than crashing mid-denoising.
    # Keyword-only and without a default, so a backend that runs anywhere says
    # ALWAYS rather than saying nothing; see the class docstring.
    requires: Requirement = field(kw_only=True)

    # Which sparsity strategy, if any. A kind rather than a flag because the
    # consumers distinguish them -- base_model gates SSTA and sparge
    # separately, and they are not interchangeable for a given model.
    sparsity: Optional[Sparsity] = None

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
    # wrong. Anything not declared here is silently accepted, which is why it
    # is keyword-only and without a default: a backend that serves every call
    # says ANY_CALL.
    accepts: CallConstraint = field(kw_only=True)

    # Where a call this backend cannot serve goes instead of raising. For a
    # kernel that covers part of a model's shapes and wants the rest served
    # rather than refused: MHA v4 runs head dim 128, and LTX-2 pairs 128-wide
    # video blocks with 64-wide audio ones under one backend choice.
    #
    # The fallback is a whole backend, so it answers with its own accepts and
    # its own kernel -- but the *selected* spec is still what base_model, usp
    # and fp8_comms read for sparsity, head balancing and pre-quantization.
    # install() therefore refuses a fallback on a spec declaring any of those,
    # rather than let a call route around a fact something else acted on.
    fallback: Optional[AttentionBackendType] = None

    # Filled in by the registry from the module the spec came from, so Impl
    # targets can be written relative to the backend package.
    package: str = ""

    # The resolved kernel, cached by resolved() so dispatch is not an import.
    # Out of compare because the spec is a frozen dataclass and so hashable by
    # its fields: caching would otherwise change an instance's hash after
    # construction, and a resolved spec would not equal the same spec unresolved.
    # Out of repr because a bound impl is a functools.partial, which prints the
    # whole table row it was bound with.
    _resolved: Optional[AttnFn] = field(default=None, compare=False, repr=False)

    # The spec ``fallback`` names, stamped by the registry once every spec is
    # placed. Out of compare and repr for the reason above and one more: a Spec
    # inside a Spec would otherwise recurse through __eq__ and __hash__.
    #
    # Holding the object rather than looking it up at dispatch is deliberate.
    # run() is traced, and this way the lookup never happens there; resolved()
    # walks this reference, so the instance we import for is the instance we
    # dispatch to.
    _fallback: Optional["Spec"] = field(default=None, compare=False, repr=False)

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

        Cached, and the cache is the point rather than an optimisation: an Impl
        with ``bound`` resolves to a fresh functools.partial each time, so
        resolving twice would hand out two callables that compare unequal and
        make Dynamo treat the second as a different function. Plain callables
        are cached too, so run() takes the same fast path for them.

        Resolves the fallback behind it too. run() reaches a fallback from
        inside the traced region, too late to import anything, so resolving a
        spec has to mean resolving everything the selection can dispatch to.

        The chain is walked after the cache check, not before, so a warm spec
        costs one attribute read. That is safe because ``_resolved`` is only
        ever set below, after the fallback has been resolved: a spec being
        warm therefore implies the whole chain behind it is.
        """
        fn = self._resolved
        if fn is not None:
            return fn
        if self._fallback is not None:
            self._fallback.resolved()
        fn = self.impl.resolve(self.package) if isinstance(self.impl, Impl) else self.impl
        object.__setattr__(self, "_resolved", fn)
        return fn

    def run(self, query, key, value, call: AttnCall):
        """Enforce ``accepts``, then dispatch -- to the fallback when there is
        one and the call is outside what this kernel serves.

        Callers use this rather than ``impl`` so the constraint is declared
        once and checked in one place; a kernel function stays pure kernel
        code."""
        reason = self.rejects(query, key, value, call)
        if reason is not None:
            fallback = self._fallback
            if fallback is not None:
                return fallback.run(query, key, value, call)
            # Raised from inside the traced region this surfaces as
            # torch._dynamo.exc.Unsupported; the message survives in the debug
            # context. Only reachable on a misconfigured run.
            raise NotImplementedError(f"{self.type.name} {reason}")
        # `is None` rather than `or`: a bound impl is a functools.partial, whose
        # truthiness Dynamo cannot evaluate, and the branch is on the hot path.
        fn = self._resolved
        if fn is None:
            fn = self.resolved()
        return fn(query, key, value, call)
