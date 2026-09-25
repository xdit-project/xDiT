"""What a backend can be asked to do, per call.

Distinct from requirements: a requirement is about the machine and resolves
once at startup; these are about the call and can only be checked when it
arrives. They cover tensor geometry (HEAD_DIM, MHA_ONLY, SELF_ATTENTION) and
call parameters (NON_CAUSAL, NO_DROPOUT, NO_VARLEN) alike.

One declaration serves two readers: the pre-call check, and the conformance
suite choosing which shapes to exercise.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


class CallConstraint:
    """``unmet()`` returns a reason, or None when the call is acceptable."""

    def unmet(self, query, key, value, call) -> Optional[str]:
        raise NotImplementedError

    def __and__(self, other: "CallConstraint") -> "CallConstraint":
        return AllConstraints((self, other))

    def head_dims(self) -> Optional[Tuple[int, ...]]:
        """Head dims this constraint admits, when it restricts them."""
        return None


@dataclass(frozen=True)
class AllConstraints(CallConstraint):
    parts: Tuple[CallConstraint, ...]

    def unmet(self, query, key, value, call) -> Optional[str]:
        for part in self.parts:
            reason = part.unmet(query, key, value, call)
            if reason is not None:
                return reason
        return None

    def head_dims(self) -> Optional[Tuple[int, ...]]:
        for part in self.parts:
            dims = part.head_dims()
            if dims is not None:
                return dims
        return None


@dataclass(frozen=True)
class _AnyCall(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        return None


ANY_CALL = _AnyCall()


@dataclass(frozen=True)
class HEAD_DIM(CallConstraint):
    allowed: Tuple[int, ...]

    def __init__(self, *allowed: int):
        object.__setattr__(self, "allowed", tuple(allowed))

    def unmet(self, query, key, value, call) -> Optional[str]:
        found = query.shape[-1]
        if found in self.allowed:
            return None
        allowed = ", ".join(str(d) for d in self.allowed)
        return f"supports head dimension {allowed} only, got {found}"

    def head_dims(self) -> Tuple[int, ...]:
        return self.allowed


@dataclass(frozen=True)
class _NonCausal(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        return "does not support causal masking" if call.is_causal else None


@dataclass(frozen=True)
class _MhaOnly(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
            return (
                "supports MHA only (equal query and key/value head counts), got "
                f"{query.shape[1]} and {key.shape[1]}"
            )
        return None


@dataclass(frozen=True)
class _SelfAttention(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        if query.shape[2] != key.shape[2]:
            return "supports self attention only (query and key lengths differ)"
        return None


@dataclass(frozen=True)
class _NoVarlen(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        return "does not support varlen packed keys" if call.varlen else None


class _TrailingPadOnly(CallConstraint):
    """Packed keys are acceptable only as a declared uniform trailing pad.

    A kernel with no key-padding mask can still serve a padded request when the
    pad is one trailing block: keeping every query row and shortening K/V is
    the same computation. Only the producer knows the pad is trailing --
    cu_seqlens_k having one segment does not imply it, and interior gaps would
    be silently mis-served -- so the declaration is what makes it safe, and a
    packed call without one is refused.
    """

    def unmet(self, query, key, value, call) -> Optional[str]:
        if call.varlen is None:
            return None

        valid_kv_len = call.attention_kwargs.get("valid_kv_len")
        if valid_kv_len is None:
            return "does not support varlen packed keys"
        if not 0 < valid_kv_len <= key.shape[2]:
            return (
                f"needs valid_kv_len in [1, {key.shape[2]}], got {valid_kv_len}"
            )
        if call.varlen.max_seqlen_k != valid_kv_len:
            return (
                "needs a trailing pad, whose longest segment is its valid key "
                f"count: valid_kv_len={valid_kv_len} but "
                f"max_seqlen_k={call.varlen.max_seqlen_k}"
            )
        return None


@dataclass(frozen=True)
class _NoDropout(CallConstraint):
    def unmet(self, query, key, value, call) -> Optional[str]:
        if call.dropout_p not in (None, 0.0):
            return "does not support attention dropout"
        return None


NON_CAUSAL = _NonCausal()
MHA_ONLY = _MhaOnly()
SELF_ATTENTION = _SelfAttention()
NO_VARLEN = _NoVarlen()
TRAILING_PAD_ONLY = _TrailingPadOnly()
NO_DROPOUT = _NoDropout()
