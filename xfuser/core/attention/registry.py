"""The backend registry.

Backend modules each expose a module-level ``SPECS`` list; the package __init__
imports them explicitly and installs the result. Explicit rather than
auto-discovered, so that grep finds every backend and a forgotten import is a
loud failure rather than a silently absent backend.

Built once, at import, by a pure function, and read-only afterwards. REGISTRY
is a view onto a dict nothing outside this module holds, so a consumer that
means to query the registry cannot accidentally edit it: the lists other
subsystems used to keep by hand are now derived from it, and a stray write
would quietly change what every one of them believes.

Consumers ask the registry rather than keeping their own lists: runtime_state
asks which backends quantise, usp which balance heads, base_model which carry
each sparsity strategy. A new backend joins those sets by declaring the field.
"""

import contextlib
from dataclasses import replace
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional

from xfuser.core.attention.spec import AttentionBackendType, Spec

# The proxy is a live view, so this is rebound by content, never by name --
# a consumer that imported REGISTRY keeps seeing the installed registry.
#
# Every lookup in this module reads _REGISTRY, not the proxy. Partly because
# the proxy is the outward face and has no business in the implementation, but
# concretely because find() runs inside a traced region and Dynamo cannot
# subscript a mappingproxy by a non-constant key.
_REGISTRY: Dict[AttentionBackendType, Spec] = {}
REGISTRY: Mapping[AttentionBackendType, Spec] = MappingProxyType(_REGISTRY)


def build_registry(modules: Iterable) -> Mapping[AttentionBackendType, Spec]:
    """Every module's SPECS as one read-only mapping, or an exception.

    Pure -- nothing global is touched, so a caller can build a registry to
    inspect it without installing it. Each module's __name__ becomes the
    package its specs' Impl targets resolve against.
    """
    built: Dict[AttentionBackendType, Spec] = {}
    for module in modules:
        for spec in module.SPECS:
            _place(built, spec, module.__name__)
    _link_fallbacks(built)
    return MappingProxyType(built)


def install(modules: Iterable) -> None:
    """Build the registry and publish it. Called once, from the package
    __init__.

    Completeness is checked here rather than in build_registry because it is
    an invariant of *the* registry, not of any registry: an enum member with
    no spec is selectable on the command line and resolves to nothing.
    """
    built = build_registry(modules)
    missing = [b.name for b in AttentionBackendType if b not in built]
    if missing:
        raise ValueError(f"every AttentionBackendType needs a spec; no module declares one for: {', '.join(missing)}")
    _REGISTRY.clear()
    _REGISTRY.update(built)


def _link_fallbacks(built: Dict[AttentionBackendType, Spec]) -> None:
    """Resolve every declared fallback to the spec it names, and refuse the
    arrangements that would be wrong.

    Stamped rather than looked up at dispatch, and stamped by mutation rather
    than by ``replace``: every reference must be the same object the registry
    holds, so that resolving one resolves the one that will actually run. A
    copy would keep its own empty ``_resolved`` and import from inside the
    traced region on first use.
    """
    # Facts a fallback would route around. base_model gates on sparsity, usp
    # on head_balanced, fp8_comms quantises Q/K/V before the call on
    # accepts_prequantized -- all read from the *selected* spec, none of which
    # the fallback's kernel would honour.
    exclusive = ("sparsity", "head_balanced", "accepts_prequantized")

    for spec in built.values():
        if spec.fallback is None:
            continue
        declared = [f for f in exclusive if getattr(spec, f)]
        if declared:
            raise ValueError(
                f"{spec.type.name} declares both a fallback and "
                f"{', '.join(declared)}. Those are read from the selected "
                "backend by other subsystems, which the fallback's kernel "
                "would not honour."
            )
        target = built.get(spec.fallback)
        if target is None:
            raise ValueError(f"{spec.type.name} falls back to {spec.fallback.name}, which no module registers")
        object.__setattr__(spec, "_fallback", target)

    for spec in built.values():
        _refuse_fallback_cycle(spec)


def _refuse_fallback_cycle(start: Spec) -> None:
    """A cycle would make resolved() recurse forever and run() loop."""
    seen = [start.type.name]
    spec = start._fallback
    while spec is not None:
        if spec.type.name in seen:
            raise ValueError("fallback cycle: " + " -> ".join(seen + [spec.type.name]))
        seen.append(spec.type.name)
        spec = spec._fallback


def _place(built: Dict[AttentionBackendType, Spec], spec: Spec, package: str) -> None:
    if package and not spec.package:
        spec = replace(spec, package=package)
    if not isinstance(spec.type, AttentionBackendType):
        raise TypeError(f"{spec.type!r} is not an AttentionBackendType")
    if spec.type in built:
        raise ValueError(f"{spec.type.name} is already registered, by {built[spec.type].package}")
    built[spec.type] = spec


@contextlib.contextmanager
def using(specs: Iterable[Spec], package: str = ""):
    """Swap in a registry built from ``specs`` for the duration of the block.

    The one supported way to change the registry after import, and it exists
    for tests: they need a registry holding two or three known specs, which is
    not a state install() would ever produce. Restores on the way out, so a
    failing test cannot leave the real registry short of its backends.
    """
    saved = dict(_REGISTRY)
    built: Dict[AttentionBackendType, Spec] = {}
    for spec in specs:
        _place(built, spec, package)
    _link_fallbacks(built)
    _REGISTRY.clear()
    _REGISTRY.update(built)
    try:
        yield REGISTRY
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def get(backend: AttentionBackendType) -> Spec:
    """The spec, or KeyError naming the backend. Use find() where absence is
    an ordinary answer."""
    try:
        return _REGISTRY[backend]
    except KeyError:
        raise KeyError(f"{backend.name} has no registered spec") from None


def find(backend: AttentionBackendType) -> Optional[Spec]:
    """The spec, or None. The lookup usp and runtime_state make when they do
    not yet know a backend is registered.

    Reads the dict rather than the REGISTRY proxy, and must keep doing so:
    this runs inside the traced region, and Dynamo refuses a mappingproxy
    subscripted by anything it cannot constant-fold -- "non-const keys in
    mappingproxy" -- which a backend enum read from runtime state is not.
    A plain dict with the same keys traces.
    """
    return _REGISTRY.get(backend)


def where(**flags) -> List[Spec]:
    """Specs whose fields all match, e.g. where(sparsity=Sparsity.SPARGE).
    Derived properties work too, so where(is_sparse=True) is valid."""
    return [spec for spec in _REGISTRY.values() if all(getattr(spec, key) == value for key, value in flags.items())]


def types_where(**flags) -> frozenset:
    """The same query, as a set of enum members."""
    return frozenset(spec.type for spec in where(**flags))


def missing_specs() -> List[AttentionBackendType]:
    """Enum members with no spec. Empty on the installed registry -- install()
    refuses to publish one that is short -- so this is for inspecting a
    registry built but not installed, or one swapped in by using()."""
    return [b for b in AttentionBackendType if b not in _REGISTRY]


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
    rows = sorted(_REGISTRY.values(), key=lambda s: s.type.name)
    if not rows:
        return "registry is empty"

    width = max(len(s.type.name) for s in rows)
    fb_width = max([len(s.fallback.name) for s in rows if s.fallback] + [8])
    header = "BACKEND".ljust(width) + "  RING  SPARSE  HEADBAL  LOWPREC  " + "FALLBACK".ljust(fb_width) + "  REQUIRES"
    lines = [header, "-" * len(header)]
    for spec in rows:
        lines.append(
            spec.type.name.ljust(width)
            # Resolved here rather than printed as a predicate: the manifest
            # describes this machine, and "can it ring here" is the useful fact.
            + "  "
            + ("y" if spec.ring.satisfied() else "-").center(4)
            + "  "
            + ("y" if spec.is_sparse else "-").center(6)
            + "  "
            + ("y" if spec.head_balanced else "-").center(7)
            + "  "
            + ("y" if spec.low_precision else "-").center(7)
            + "  "
            + (spec.fallback.name if spec.fallback else "-").ljust(fb_width)
            + "  "
            + _describe_requirement(spec)
        )
    return "\n".join(lines)


def _describe_requirement(spec: Spec) -> str:
    reason = spec.unavailable()
    return "ok" if reason is None else reason
