"""Environment requirements, as composable predicates.

A requirement answers one question: can this backend run on this machine?
Predicates are lazy -- nothing is probed until something asks -- and memoised,
so many backends naming the same symbol cause one import attempt.

Failure messages say what is missing and why, but never prescribe a remedy: a
vendor library moves in both directions, so a missing symbol means "too old"
as often as "too new".
"""

import functools
import importlib
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Informational only: what CI has actually verified. Not a floor, not enforced.
TESTED_AGAINST = "AITER @ 49c6fdd45 (2026-09-22)"


# ---------------------------------------------------------------------------
# probes (memoised)
# ---------------------------------------------------------------------------

def resolve(target: str):
    """Import "module:name" and return the object, or None when absent.

    Anything raised while importing means absent: vendor modules fail at
    import for reasons beyond ImportError, device probes among them.

    Not cached itself: it delegates to the memoised _resolve_with_reason, so
    one cache backs both this and the reasons the requirements report.
    """
    return _resolve_with_reason(target)[0]


@functools.lru_cache(maxsize=None)
def _resolve_with_reason(target: str):
    """(object, why-it-is-missing). The reason matters: "not installed" and
    "installed but its extension will not load" are different problems, and a
    bare "not importable" sends people looking for the wrong one."""
    module_name, _, symbol = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] != module_name.split(".")[0]:
            # The module is present; something it imports is not.
            return None, f"{module_name} needs {exc.name}, which is not installed"
        return None, f"{module_name} is not installed"
    except Exception as exc:                # noqa: BLE001 - absence is the answer
        return None, f"{module_name} failed to import: {type(exc).__name__}: {exc}"

    if not symbol:
        return module, None
    obj = getattr(module, symbol, None)
    if obj is None:
        return None, f"{module_name} has no {symbol}"
    return obj, None


@functools.lru_cache(maxsize=None)
def _signature_params(target: str) -> Optional[frozenset]:
    """Parameter names of "module:name", or None when the signature cannot be
    read at all -- C extensions without argument clinic behave this way, and
    that is a different answer from "the parameter is absent"."""
    obj = resolve(target)
    if obj is None:
        return None
    try:
        return frozenset(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return None


@functools.lru_cache(maxsize=None)
def device_arch() -> str:
    """gcnArchName of device 0, or "" when there is no GPU."""
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:                       # noqa: BLE001 - reporting only
        return ""


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------

class Requirement:
    """Base predicate. ``unmet()`` returns a reason, or None when satisfied.

    Backend modules subclass this for checks the generic predicates cannot
    express. Give the subclass a name that reads at the call site, and memoise
    inside it if the check is expensive.
    """

    def unmet(self) -> Optional[str]:
        raise NotImplementedError

    def __and__(self, other: "Requirement") -> "Requirement":
        return All((self, other))

    def __or__(self, other: "Requirement") -> "Requirement":
        return Any((self, other))

    def satisfied(self) -> bool:
        """Yes/no, for branching on a capability. Use unmet() when you want
        the reason -- that is what gating and error messages need."""
        return self.unmet() is None

    def __bool__(self):
        raise TypeError("Requirement is not a bool; call unmet()")


@dataclass(frozen=True)
class All(Requirement):
    parts: Tuple[Requirement, ...]

    def unmet(self) -> Optional[str]:
        for part in self._flat():
            reason = part.unmet()
            if reason is not None:
                return reason
        return None

    def _flat(self):
        for part in self.parts:
            if isinstance(part, All):
                yield from part._flat()
            else:
                yield part


@dataclass(frozen=True)
class Any(Requirement):
    """Satisfied when any part is. For alternatives that are genuinely
    different checks -- a symbol that moved between modules, say. For several
    architectures prefer ARCH("a", "b"), which reports one combined reason
    instead of one per branch."""

    parts: Tuple[Requirement, ...]

    def unmet(self) -> Optional[str]:
        reasons = []
        for part in self._flat():
            reason = part.unmet()
            if reason is None:
                return None
            reasons.append(reason)
        return "none of: " + "; ".join(reasons)

    def _flat(self):
        for part in self.parts:
            if isinstance(part, Any):
                yield from part._flat()
            else:
                yield part


@dataclass(frozen=True)
class _Always(Requirement):
    def unmet(self) -> Optional[str]:
        return None


ALWAYS = _Always()


@dataclass(frozen=True)
class SYMBOL(Requirement):
    """"module:name" must be importable."""

    target: str

    def unmet(self) -> Optional[str]:
        obj, reason = _resolve_with_reason(self.target)
        if obj is None:
            return f"{self.target.replace(':', '.')} is unavailable -- {reason}"
        return None


@dataclass(frozen=True)
class PARAM(Requirement):
    """"module:name" must accept a given keyword parameter."""

    target: str
    parameter: str

    def unmet(self) -> Optional[str]:
        name = self.target.replace(":", ".")
        obj, reason = _resolve_with_reason(self.target)
        if obj is None:
            return f"{name} is unavailable -- {reason}"
        params = _signature_params(self.target)
        if params is None:
            # Refusing is the safe answer under a hard-fail policy, but say why:
            # silently treating this as "absent" would disable a working backend.
            return (f"{name} is unavailable -- its signature cannot be read, so "
                    f"the {self.parameter!r} parameter cannot be confirmed")
        if self.parameter not in params:
            # The function is here; this build's version of it differs. Say so,
            # because "not importable" would send people to the install.
            return f"{name} takes no {self.parameter!r} parameter in this build"
        return None


@dataclass(frozen=True)
class FIRST_OF(Requirement):
    """A symbol that lives at more than one path depending on the version.

    Gates on at least one path resolving, and ``resolve()`` returns the first
    that does. One declaration serves both, so the requirement and the import
    cannot drift -- adding a path is a single line.
    """

    targets: Tuple[str, ...]

    def __init__(self, *targets: str):
        object.__setattr__(self, "targets", tuple(targets))

    def _names(self) -> str:
        return ", ".join(t.replace(":", ".") for t in self.targets)

    def unmet(self) -> Optional[str]:
        if any(resolve(t) is not None for t in self.targets):
            return None
        why = "; ".join(
            f"{t.replace(':', '.')}: {_resolve_with_reason(t)[1]}" for t in self.targets
        )
        return f"none of these is available -- {why}"

    def resolve(self):
        for target in self.targets:
            obj = resolve(target)
            if obj is not None:
                return obj
        raise ImportError(self.unmet())


@dataclass(frozen=True)
class ARCH(Requirement):
    """Device arch must contain one of these fragments (e.g. "gfx950")."""

    names: Tuple[str, ...]

    def __init__(self, *names: str):
        object.__setattr__(self, "names", tuple(names))

    def unmet(self) -> Optional[str]:
        arch = device_arch()
        if not arch:
            return f"requires {' or '.join(self.names)}, no GPU detected"
        if any(name in arch for name in self.names):
            return None
        return f"requires {' or '.join(self.names)}, found {arch}"


@dataclass(frozen=True)
class PLATFORM(Requirement):
    """"cuda" for an NVIDIA build, "rocm" for a HIP build, "npu" for Ascend."""

    name: str

    def unmet(self) -> Optional[str]:
        found = _platform()
        if found != self.name:
            return f"requires {self.name}, found {found}"
        return None


@functools.lru_cache(maxsize=1)
def _platform() -> str:
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except ModuleNotFoundError:
        pass
    return "rocm" if torch.version.hip is not None else "cuda"


@dataclass(frozen=True)
class CUDA_CAPABILITY(Requirement):
    """Minimum NVIDIA compute capability, as (major, minor)."""

    minimum: Tuple[int, int]

    def unmet(self) -> Optional[str]:
        if not torch.cuda.is_available():
            return f"requires compute capability >= {self.minimum}, no GPU detected"
        found = torch.cuda.get_device_capability(0)
        if found < self.minimum:
            return f"requires compute capability >= {self.minimum}, found {found}"
        return None


