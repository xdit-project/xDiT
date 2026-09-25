"""The backend registry.

Backend modules each expose a module-level ``SPECS`` list; the package __init__
imports them explicitly and registers the result. Explicit rather than
auto-discovered, so that grep finds every backend and a forgotten import is a
loud KeyError rather than a silently absent backend.

Consumers ask the registry rather than keeping their own lists: runtime_state
asks which backends quantise, usp which balance heads, base_model which carry
each sparsity strategy. A new backend joins those sets by declaring the field.
"""

from typing import Dict, Iterable, List, Optional

from xfuser.core.attention.spec import AttentionBackendType, Spec

REGISTRY: Dict[AttentionBackendType, Spec] = {}


def register(specs: Iterable[Spec], package: str = "") -> None:
    """Register a module's SPECS. ``package`` is that module's import path, so
    an Impl target can be written relative to it."""
    from dataclasses import replace

    for spec in specs:
        if package and not spec.package:
            spec = replace(spec, package=package)
        if not isinstance(spec.type, AttentionBackendType):
            raise TypeError(f"{spec.type!r} is not an AttentionBackendType")
        if spec.type in REGISTRY:
            raise ValueError(f"{spec.type.name} is already registered")
        REGISTRY[spec.type] = spec


def clear() -> None:
    """Drop every registration. For tests."""
    REGISTRY.clear()


def get(backend: AttentionBackendType) -> Spec:
    try:
        return REGISTRY[backend]
    except KeyError:
        raise KeyError(f"{backend.name} has no registered spec") from None


def where(**flags) -> List[Spec]:
    """Specs whose fields all match, e.g. where(sparsity="sparge").
    Derived properties work too, so where(is_sparse=True) is valid."""
    return [
        spec
        for spec in REGISTRY.values()
        if all(getattr(spec, key) == value for key, value in flags.items())
    ]


def types_where(**flags) -> frozenset:
    """The same query, as a set of enum members."""
    return frozenset(spec.type for spec in where(**flags))


def missing_specs() -> List[AttentionBackendType]:
    """Enum members with no spec. Should be empty: a member without one cannot
    be selected."""
    return [b for b in AttentionBackendType if b not in REGISTRY]


def available(backend: AttentionBackendType) -> Optional[str]:
    """None when the backend can run here, else why not."""
    return get(backend).unavailable()


def prepare(backend: AttentionBackendType) -> None:
    """Import the backend's kernel module. Called once when a backend is
    selected, so the vendor import and any custom op registration happen
    outside every compiled region."""
    get(backend).resolved()


def manifest() -> str:
    """The registry as a table. Rendered from the specs themselves, so it
    cannot drift from them; suitable as a golden-file test in review."""
    rows = sorted(REGISTRY.values(), key=lambda s: s.type.name)
    if not rows:
        return "registry is empty"

    width = max(len(s.type.name) for s in rows)
    header = (
        "BACKEND".ljust(width)
        + "  LSE  SPARSE  HEADBAL  LOWPREC  REQUIRES"
    )
    lines = [header, "-" * len(header)]
    for spec in rows:
        lines.append(
            spec.type.name.ljust(width)
            + "  " + ("y" if spec.returns_lse else "-").center(3)
            + "  " + ("y" if spec.is_sparse else "-").center(6)
            + "  " + ("y" if spec.head_balanced else "-").center(7)
            + "  " + ("y" if spec.low_precision else "-").center(7)
            + "  " + _describe_requirement(spec)
        )
    return "\n".join(lines)


def _describe_requirement(spec: Spec) -> str:
    reason = spec.unavailable()
    return "ok" if reason is None else reason
