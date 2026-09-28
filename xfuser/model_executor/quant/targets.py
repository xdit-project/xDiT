"""What a model declares for GEMM quantization, and how a run resolves it.

A model states which modules may be quantized and which of them should stay at
the better precision. It never names a format: the run does that, through
``--gemm_quantization``. One format quantizes every target; two split them.

That separation is the point. A model knows that its early and late blocks
carry more of the output quality than the middle ones; it does not know
whether this machine is running fp8, mxfp6 or mxfp4.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

from xfuser.config.gemm import GemmQuantizationSpec


def _is_under(path: str, root: str) -> bool:
    """Whether `path` is `root` or sits beneath it, on a segment boundary.

    Segment-aware so "transformer.blocks" cannot match "transformer.blocks_2",
    which a bare startswith would.
    """
    return path == root or path.startswith(f"{root}.")


def _ends_with(path: str, suffix: str) -> bool:
    return path == suffix or path.endswith(f".{suffix}")


@dataclass(frozen=True)
class Select:
    """A set of module paths, named three ways.

    All three match the module's full path on segment boundaries. ``prefixes``
    is absolute like the others -- writing a bare block index would require
    knowing what it is relative to, and a trailing dot to stop "3." matching
    "30.", which is a trap rather than a feature.
    """

    modules: Tuple[str, ...] = ()
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.modules or self.prefixes or self.suffixes)

    def matches(self, path: str) -> bool:
        return (
            any(_is_under(path, m) for m in self.modules)
            or any(_is_under(path, p) for p in self.prefixes)
            or any(_ends_with(path, s) for s in self.suffixes)
        )

    def roots(self) -> Tuple[str, ...]:
        """The paths that can be named up front, for consumers that walk a
        subtree rather than test one module. Suffixes have no root."""
        return tuple(dict.fromkeys(self.modules + self.prefixes))



#: The components a model can target. Adding one -- "vae", say -- means adding
#: it here and as a field below; nothing after that is component-aware.
COMPONENTS = ("transformer", "text_encoder")


@dataclass(frozen=True)
class GemmTargets:
    """One model's quantization targets, named per component.

    Every component is declared the same way and enabled the same way; none is
    privileged. ``keep_high`` cuts across all of them, naming the subset to
    hold at the better precision when the run gives two formats, and is inert
    when it gives one.
    """

    transformer: Select = Select()
    text_encoder: Select = Select()
    keep_high: Select = Select()

    @property
    def components(self) -> Mapping[str, Select]:
        """Declared selections by component, in COMPONENTS order, empties
        dropped -- so a model that targets nothing but the transformer looks
        exactly like one that has no other component to target."""
        found = ((name, getattr(self, name)) for name in COMPONENTS)
        return {name: select for name, select in found if select}

    def __post_init__(self) -> None:
        targeted = [
            root for select in self.components.values() for root in select.roots()
        ]
        stray = [
            root for root in self.keep_high.roots()
            if not any(_is_under(root, t) for t in targeted)
        ]
        if stray:
            raise ValueError(
                f"keep_high selects modules outside the target set: {stray}. "
                "It carves a subset out of what is already targeted, so "
                "anything it names must be quantizable in the first place."
            )


@dataclass(frozen=True)
class GemmPlan:
    """What this run quantizes, and to what.

    Resolved once, from the model's targets and the run's spec. Every field is
    a decision: which components are in play, and which formats they take. No
    accessor re-reads a run option to find out.
    """

    targeted: Mapping[str, Select]
    keep_high: Select
    low: Optional[str]
    high: Optional[str]

    @property
    def quantizes(self) -> bool:
        return self.low is not None and bool(self.targeted)

    def format_for(self, path: str) -> Optional[str]:
        """The format this module is quantized to, or None to leave it alone.

        One rule, whichever component the module belongs to: the low format,
        unless the model carved it into keep_high and the run named a high one.
        """
        if self.low is None:
            return None
        if not any(select.matches(path) for select in self.targeted.values()):
            return None
        if self.high is not None and self.keep_high.matches(path):
            return self.high
        return self.low

    def roots(
        self, format_name: str, *, component: Optional[str] = None
    ) -> Tuple[str, ...]:
        """Declared paths quantized to `format_name`, for consumers that walk a
        subtree rather than test one module.

        Narrow to one `component` for the loaders that handle them separately.
        A root whose descendants are split across formats appears under both
        formats; the per-module decision stays format_for's.
        """
        if self.low is None or format_name not in (self.low, self.high):
            return ()

        if component is None:
            selects = list(self.targeted.values())
        else:
            selects = [self.targeted[component]] if component in self.targeted else []
        declared = [root for select in selects for root in select.roots()]

        found = []
        for root in declared:
            whole = self.high is not None and self.keep_high.matches(root)
            if (format_name == self.high) == whole:
                found.append(root)
        if format_name == self.high:
            # carve-outs named below a target root rather than at it
            found += [
                root
                for root in self.keep_high.roots()
                if root not in found and any(_is_under(root, d) for d in declared)
            ]
        return tuple(dict.fromkeys(found))


def resolve(
    targets: GemmTargets,
    spec: GemmQuantizationSpec,
    *,
    enable: Iterable[str] = ("transformer",),
) -> GemmPlan:
    """Bind a model's targets to the formats and components this run asked for.

    `enable` names the components in play. It defaults to the transformer
    because that is what `--gemm_quantization` alone means; a component with a
    switch of its own is added by the caller that reads the switch.

    A pure spec quantizes every enabled target to one format, so `keep_high`
    has nothing to carve and is inert. A tiered spec splits them.
    """
    enable = tuple(enable)
    unknown = [name for name in enable if name not in COMPONENTS]
    if unknown:
        raise ValueError(
            f"unknown GEMM component(s) {unknown}; expected one of "
            + ", ".join(COMPONENTS)
        )

    low = None if spec.low == "none" else spec.low
    return GemmPlan(
        targeted={
            name: select
            for name, select in targets.components.items()
            if name in enable
        },
        keep_high=targets.keep_high,
        low=low,
        high=spec.high if low is not None else None,
    )
