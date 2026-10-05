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

from xfuser.config.gemm import MIN_M_FORMATS, GemmQuantizationSpec


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

    ``modules``, ``prefixes`` and ``suffixes`` union: a path in any of them is
    selected. They match the module's full path on segment boundaries, and
    ``prefixes`` is absolute like the others -- writing a bare block index
    would require knowing what it is relative to, and a trailing dot to stop
    "3." matching "30.", which is a trap rather than a feature.

    ``only`` narrows instead of widening. Models that quantize part of a block
    -- the projections but not the output layer, say -- name the leaves they
    want, and everything else in the selected subtree is left alone.
    """

    modules: Tuple[str, ...] = ()
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    only: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.modules or self.prefixes or self.suffixes)

    def _selected(self, path: str) -> bool:
        return (
            any(_is_under(path, m) for m in self.modules)
            or any(_is_under(path, p) for p in self.prefixes)
            or any(_ends_with(path, s) for s in self.suffixes)
        )

    def matches(self, path: str) -> bool:
        if not self._selected(path):
            return False
        if not self.only:
            return True
        return any(_ends_with(path, leaf) for leaf in self.only)

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

    ``short_sequence`` also cuts across them, and says something about the
    module rather than about precision: its GEMMs take their M from a sequence
    that sequence parallelism chunks, so M can fall to a handful of tokens. A
    format whose kernel refuses a small M leaves those modules alone -- see
    ``MIN_M_FORMATS``. It names targets exactly, because the resolver drops the
    whole entry rather than carving inside it.
    """

    transformer: Select = Select()
    text_encoder: Select = Select()
    keep_high: Select = Select()
    short_sequence: Select = Select()

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
            root
            for root in self.keep_high.roots()
            if not any(_is_under(root, t) for t in targeted)
        ]
        if stray:
            raise ValueError(
                f"keep_high selects modules outside the target set: {stray}. "
                "It carves a subset out of what is already targeted, so "
                "anything it names must be quantizable in the first place."
            )

        if self.short_sequence.suffixes or self.short_sequence.only:
            raise ValueError(
                "short_sequence names whole targets, so it takes modules or "
                "prefixes; a suffix or a leaf list would have to be carved "
                "inside a target, which is not what it means."
            )
        outside = [root for root in self.short_sequence.roots() if root not in targeted]
        if outside:
            raise ValueError(
                f"short_sequence selects modules that are not declared "
                f"targets: {outside}. The resolver drops the whole entry from "
                "the target set, so it has to name one exactly."
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

    @property
    def only_suffixes(self) -> Tuple[str, ...]:
        """Leaf suffixes the declaration narrows to, for walks that convert a
        whole subtree and need a filter to apply inside it."""
        found = [leaf for select in self.targeted.values() for leaf in select.only]
        return tuple(dict.fromkeys(found))

    def declared_roots(self, *, component: Optional[str] = None) -> Tuple[str, ...]:
        """Every subtree this run quantizes, whatever format each leaf takes.

        The starting points for a walk. `roots` answers a different question --
        which subtrees one format owns whole -- and a walk started from those
        would never reach a carve-out named by suffix, because a suffix belongs
        to no subtree of its own. Walk from here and ask `format_for` per leaf.
        """
        if not self.quantizes:
            return ()
        if component is None:
            selects = list(self.targeted.values())
        else:
            selects = [self.targeted[component]] if component in self.targeted else []
        return tuple(dict.fromkeys(r for select in selects for r in select.roots()))

    def relative_to(self, component_name: str, roots: Iterable[str]) -> Tuple[str, ...]:
        """`roots` rewritten as paths under `component_name`, others dropped.

        A converter is handed one component and walks paths relative to it,
        while the declaration names them from the pipeline root. The component
        itself becomes "", which every consumer reads as "all of it".
        """
        prefix = f"{component_name}."
        return tuple(
            "" if root == component_name else root[len(prefix) :]
            for root in roots
            if root == component_name or root.startswith(prefix)
        )

    @property
    def formats_in_play(self) -> Tuple[str, ...]:
        """The formats this run places, low first.

        The one answer to "which converters does this run need". Each consumer
        rebuilt it from `low` and `high`, which was the last place a tier was
        visible below the resolver -- and three identical expressions to edit
        if a run ever names more than two.
        """
        return tuple(dict.fromkeys(n for n in (self.low, self.high) if n))

    def walk_roots(self, format_name: str, *, paired: bool = False) -> Tuple[str, ...]:
        """Where a converter for `format_name` should start walking.

        `roots` names the subtrees this format owns whole, which is enough
        except for a carve-out with no subtree of its own: a leaf suffix occurs
        inside targets the other format owns, so reaching it means walking
        those too and letting `format_for` reject the rest.

        `paired` says this format is the hybrid schedule's companion, built
        at every leaf the low walk reaches rather than only where the plan
        assigns it. The caller knows which format that is; the plan only knows
        what it means for where a walk starts.
        """
        found = list(self.roots(format_name))
        widen = paired or (format_name == self.high and self.keep_high.suffixes)
        if widen:
            found += [r for r in self.declared_roots() if r not in found]
        return tuple(found)

    @property
    def splits_a_target(self) -> bool:
        """Whether the high format lands inside a target rather than at it.

        A consumer reasoning in whole subtrees -- "will this FSDP block hold the
        high format?" -- cannot answer from `roots` alone when the carve-out is
        a block prefix or a leaf suffix, because those sit below a target root
        rather than at one. This says when that is so.
        """
        if self.high is None or not self.keep_high:
            return False
        if self.keep_high.suffixes:
            return True
        declared = self.declared_roots()
        return any(
            root not in declared and any(_is_under(root, d) for d in declared)
            for root in self.keep_high.roots()
        )

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


def _format_at(
    path: str, keep_high: Select, low: Optional[str], high: Optional[str]
) -> Optional[str]:
    """The format a declared root takes, before any narrowing."""
    if low is None:
        return None
    if high is not None and keep_high.matches(path):
        return high
    return low


def _without(select: Select, dropped: Tuple[str, ...]) -> Select:
    return Select(
        modules=tuple(m for m in select.modules if m not in dropped),
        prefixes=tuple(p for p in select.prefixes if p not in dropped),
        suffixes=select.suffixes,
        only=select.only,
    )


def resolve(
    targets: GemmTargets,
    spec: GemmQuantizationSpec,
    *,
    enable: Iterable[str] = ("transformer",),
    sp_world_size: int = 1,
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
    high = spec.high if low is not None else None

    targeted = {
        name: select for name, select in targets.components.items() if name in enable
    }
    if sp_world_size > 1 and targets.short_sequence:
        # Sequence parallelism can chunk these modules below the M their
        # format's kernel accepts, so that format leaves them in bf16. The
        # model said which of its modules are short; the format said which
        # kernels mind. Neither had to know about the other.
        dropped = tuple(
            root
            for root in targets.short_sequence.roots()
            if _format_at(root, targets.keep_high, low, high) in MIN_M_FORMATS
        )
        if dropped:
            targeted = {
                name: narrowed
                for name, select in targeted.items()
                if (narrowed := _without(select, dropped))
            }

    return GemmPlan(
        targeted=targeted,
        keep_high=targets.keep_high,
        low=low,
        high=high,
    )
