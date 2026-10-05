"""Which implementation stores each format this run names, and whether FSDP permits it.

Selection is a question about the run, not about being a diffusion model, so it lives here rather
than on the runner base. ``ModelLoader`` owns the one cached selection for the run.

A format is a lookup: the run names one, ``_impl_for`` says which implementation has it on this
machine, and ``adapter_for`` returns that one converter. Nothing above asks which format it got
back, and a format arriving as the run's low tier resolves exactly as it would as the high one.

Whether that implementation is *allowed* is a second question, and not independent: some storage
forms cannot be sharded. A TorchAO Float8Tensor inside an FSDP2 block needs patches the environment
may not have, so ``places_under_fsdp2`` runs before allocation and fails there rather than at the
first all-gather.

Adapters are cached: selecting one probes the environment, and every consumer must get the same
answer. The cache lives on the instance, which is itself cached on the model, so it lasts as long as
the run and no longer.
"""

import functools

from .contracts import IMPL_PREFERENCE as _IMPL_PREFERENCE
from .quant_adapter import module_paths_overlap


class QuantizationBackends:
    """The adapters this run quantizes through, and the placement rules that gate them.

    Holds the ``xFuserModel`` to read its load contract, settings, and config. Keeps no state beyond
    the adapter caches, so it stays correct across the settings edits some runners make while
    loading.
    """

    def __init__(self, loader) -> None:
        self.loader = loader
        self.model = loader.model
        self._adapters = {}

    def preflight(self) -> None:
        """Resolve and validate every adapter this run will need.

        Before allocation, so "this cannot be sharded here" is a startup error
        rather than a failure at the first all-gather.
        """
        for format_name in self._formats_in_play():
            _ = self.adapter_for(format_name)
        self.assert_hybrid_schedule_is_implementable()

    def assert_hybrid_schedule_is_implementable(self) -> None:
        """Refuse a per-step pair neither half of which can be built.

        The schedule holds both formats at every leaf and picks one per step,
        so the low side has to compose the pair and the high side has to be
        installable one layer at a time. Checked here rather than at the first
        converted leaf, where it would be an AttributeError mid-load.
        """

        from .contracts import UnsupportedLoadContract

        if not getattr(self.model.config, "use_hybrid_gemm_schedule", False):
            return
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None or plan.high is None:
            return
        companion = self.adapter_for(plan.high)
        if companion is not None and not companion.builds_one_layer():
            raise UnsupportedLoadContract(
                f"{companion.impl} {companion.format_name} cannot install one "
                "layer at a time, so it cannot be the high half of a per-step "
                "GEMM schedule; drop --use_hybrid_gemm_schedule or name a "
                "high format that can"
            )

    def hybrid_companion(self, format_name: str, *, device):
        """The per-step alternate the schedule pairs with `format_name`, if any.

        The run names both formats and the plan says which is which, so the
        pairing is composed here rather than inside either converter. `device`
        is the one the low half is being built on: the companion is the same
        leaf in another precision and belongs beside it, not wherever the
        source weight happened to be resting.
        """

        if not getattr(self.model.config, "use_hybrid_gemm_schedule", False):
            return None
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None or plan.high is None or format_name != plan.low:
            return None
        companion = self.adapter_for(plan.high)
        if companion is None:
            return None
        return companion.layer_factory(device=device)

    @functools.cached_property
    def capabilities(self):
        """Every (format, implementation) this machine offers, probed once.

        Narrowed to the formats the run named: some probes import kernels,
        which is not free on a machine that will never place them. Read off the
        spec rather than the resolved plan, so a run whose model declares no
        targets still gets a truthful answer about what it asked for.
        """

        from .backends import probe_backend_capabilities

        return probe_backend_capabilities(wanted=self._formats_requested())

    def impl_for(self, format_name: str) -> str | None:
        """Which implementation stores this format on this machine.

        First available wins. Each probe is already gated to the hardware its
        kernels run on -- AITER FP8 block-scale to RDNA4, NVFP4 to Blackwell,
        MXFP4 to ROCm -- so the order below is a preference between two that
        could both work, not a hardware test repeated here.
        """
        for impl in _IMPL_PREFERENCE.get(format_name, ()):
            if self.capabilities.of(format_name, impl).available:
                return impl
        return None

    def _select(self, format_name: str):
        """Resolve one format to its adapter, refusing what cannot be placed.

        The run names a format; this says which implementation stores it here
        and whether what that implementation stores can be sharded where this
        run will put it. Both answers are the same for a format whether it
        arrived as the run's low tier or its high one.
        """

        # Imported for its registrations: each adapter class registers the
        # one pair it stores as it is defined.
        from . import backends  # noqa: F401
        from .contracts import UnsupportedLoadContract
        from .quant_adapter import build_adapter, validate_fsdp_placement

        impl = self.impl_for(format_name)
        if impl is None:
            raise UnsupportedLoadContract(
                f"this run asked for {format_name}, which nothing here can "
                "store: " + self.why_not(format_name)
            )
        capability = self.capabilities.of(format_name, impl)
        plan = self.loader.quantization_plan.gemm_plan
        adapter = build_adapter(
            format_name,
            impl,
            capability=capability,
            # Only the format a step actually runs in bf16-vs-quantized pairs
            # can refuse the schedule; the companion never drives it.
            hybrid=(
                plan is not None
                and format_name == plan.low
                and bool(self.model.config.use_hybrid_gemm_schedule)
            ),
        )
        validate_fsdp_placement(
            adapter,
            capability=capability,
            required=self.places_under_fsdp2(format_name),
        )
        return adapter

    def why_not(self, format_name: str) -> str:
        """What each implementation of a format said when it was probed."""
        tried = _IMPL_PREFERENCE.get(format_name)
        if not tried:
            return f"{format_name} is not a GEMM format this build knows"
        return "; ".join(
            f"{impl}: {self.capabilities.of(format_name, impl).reason or 'unavailable'}"
            for impl in tried
        )

    def assert_offload_is_compatible(self) -> None:
        """Refuse an offload mode an implementation in play cannot survive.

        Each converter says whether its stored weights survive the host round
        trip the hook performs, and carries the measured reason. An
        implementation that says nothing is left alone: refusing it would
        assert a claim nobody has tested.
        """

        from .contracts import UnsupportedLoadContract

        config = self.model.config
        if not getattr(config, "enable_group_cpu_offload", False):
            return
        leg = (
            "pinned"
            if getattr(config, "group_offload_low_cpu_mem", False)
            else "unpinned"
        )
        for format_name in self._formats_in_play():
            adapter = self.adapter_for(format_name)
            refusal = getattr(adapter, "group_offload_refusal", None)
            if not refusal:
                continue
            raise UnsupportedLoadContract(
                f"--enable_group_cpu_offload cannot be combined with "
                f"{format_name} on the {adapter.impl} backend: {refusal[leg]}. "
                "Offload at a format that survives the host round trip, or run "
                "without offload."
            )

    def _formats_requested(self) -> frozenset:
        """Every format this run named, before the plan resolves any targets."""
        spec = getattr(self.model.config, "gemm_quantization_spec", None)
        if spec is None:
            return frozenset()
        return frozenset(spec.formats) - {"none"}

    def _formats_in_play(self) -> tuple:
        """The formats this run will place, low and high."""
        plan = self.loader.quantization_plan.gemm_plan
        return () if plan is None else plan.formats_in_play

    def adapter_for(self, format_name: str):
        """The implementation that stores one format on this machine.

        A lookup, not a slot. The run names formats and the plan says which
        module takes which; this turns a format into the one thing that can
        store it here. Nothing above it asks which format it got back.
        """
        if format_name in (None, "none"):
            return None
        if format_name not in self._adapters:
            self._adapters[format_name] = self._select(format_name)
        return self._adapters[format_name]

    def _fsdp_target_paths(self) -> set:
        """Every pipe-level module path FSDP will wrap, empty when nothing is sharded."""
        if self.model.config.fully_shard_degree <= 1:
            return set()
        return {
            f"{component_name}.{wrap_attr}"
            for component_name, strategy in (
                self.model.settings.fsdp_strategy or {}
            ).items()
            for wrap_attr in strategy.get("wrap_attrs", ())
        }

    def places_under_fsdp2(self, format_name: str) -> bool:
        """Whether an FSDP2-wrapped block will hold this format's parameters.

        One question for every format. `walk_roots` already covers a carve-out
        named below a target rather than at it, and `paired` covers the hybrid
        schedule, which puts both formats at every leaf. Whether landing there
        is a *problem* is the adapter's answer, not this one's -- a plain packed
        parameter shards, a tensor subclass may not.
        """
        fsdp_target_paths = self._fsdp_target_paths()
        if not fsdp_target_paths:
            return False
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None:
            return False
        roots = plan.walk_roots(
            format_name,
            # Only the high format is installed everywhere the low walk goes.
            # Asking for both widened the low format onto subtrees it never
            # touches, which could refuse a placement that was always fine.
            paired=(
                bool(self.model.config.use_hybrid_gemm_schedule)
                and format_name == plan.high
            ),
        )
        return any(
            module_paths_overlap(root, fsdp_path)
            for root in roots
            for fsdp_path in fsdp_target_paths
        )

    def transformer_adapter(self, component_name: str):
        """The adapter and component-relative targets owning one transformer.

        The low format wins where it claims the component, and the high one
        answers for a component the low format never reaches -- which is the
        only sense in which this still knows there are two.
        """
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None:
            return None, ()
        for format_name in self._formats_in_play():
            targets = plan.relative_to(component_name, plan.roots(format_name))
            if targets:
                return self.adapter_for(format_name), targets
        return None, ()
