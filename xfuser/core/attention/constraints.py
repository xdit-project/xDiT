"""What a backend can be asked to do, per call.

Distinct from requirements: a requirement is about the machine and resolves
once at startup; these are about the call and can only be checked when it
arrives. Some constrain tensor geometry (HEAD_DIM, MHA_ONLY, SELF_ATTENTION),
others constrain call parameters (NON_CAUSAL, NO_DROPOUT, NO_VARLEN) -- both
answer the same question, "can this backend serve this call", and are checked
at the same moment, so they are one family.

Two things read them from one declaration: the generic pre-call check, and the
conformance suite choosing which shapes to exercise.
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
NO_DROPOUT = _NoDropout()
