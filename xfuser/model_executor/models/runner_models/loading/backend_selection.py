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
from types import SimpleNamespace

from xfuser.envs import _is_cuda
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

    @functools.cached_property
    def format_capabilities(self):
        """Probe format capabilities once, and only include MXFP6 when asked."""

        from .format_backends import probe_format_backend_capabilities

        return probe_format_backend_capabilities(
            require_mxfp6="fp6" in self._formats_in_play()
        )

    @functools.cached_property
    def fp8_capabilities(self):
        from .fp8_backends import probe_fp8_backend_capabilities

        return probe_fp8_backend_capabilities()

    def _impl_for(self, format_name: str) -> str | None:
        """Which implementation stores this format on this machine.

        The one place format and implementation meet. A new format adds a row;
        nothing else in the loading layer changes.
        """
        from .fp8_backends import fp8_backend_name

        if format_name == "fp8":
            return fp8_backend_name(self.fp8_capabilities)
        return {
            "fp4": "torchao" if _is_cuda() else "aiter",
            "fp6": "aiter",
            "int8": "torchao",
        }.get(format_name)

    def _select(self, format_name: str):
        """Resolve one format to its adapter, refusing an FSDP placement it cannot take.

        The run names a format; this says which implementation stores it here
        and whether the shape that implementation stores can be sharded where
        this run will put it. Both answers are the same for a format whether it
        arrived as the run's low tier or its high one.
        """

        from .contracts import QuantizationBackend, QuantizationFormat
        from .fp8_backends import select_fp8_backend, validate_torchao_fsdp2_patches
        from .format_backends import (
            select_format_backend,
            validate_format_fsdp_placement,
        )

        impl = self._impl_for(format_name)
        if impl is None:
            return None
        contract = SimpleNamespace(
            requested_format=QuantizationFormat(format_name),
            selected_backend=QuantizationBackend(impl),
        )
        places = self.places_under_fsdp2(format_name)
        if format_name == "fp8":
            adapter = select_fp8_backend(contract, capabilities=self.fp8_capabilities)
            validate_torchao_fsdp2_patches(
                contract, capabilities=self.fp8_capabilities, required=places
            )
            return adapter

        plan = self.loader.quantization_plan.gemm_plan
        adapter = select_format_backend(
            contract,
            capabilities=self.format_capabilities,
            # Only the format a step actually runs in bf16-vs-quantized pairs
            # can refuse the hybrid schedule; the companion never drives it.
            hybrid=(
                plan is not None
                and format_name == plan.low
                and bool(self.model.config.use_hybrid_gemm_schedule)
            ),
        )
        validate_format_fsdp_placement(
            contract,
            adapter,
            capabilities=self.format_capabilities,
            required=places,
        )
        return adapter

    def _formats_in_play(self) -> tuple:
        """The formats this run will place, low and high."""
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None:
            return ()
        return tuple(dict.fromkeys(n for n in (plan.low, plan.high) if n))

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
            paired=bool(self.model.config.use_hybrid_gemm_schedule),
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
