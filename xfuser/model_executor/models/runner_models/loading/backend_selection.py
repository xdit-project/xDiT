"""Which quantization implementation a run uses, and whether FSDP placement permits it.

Selection is a question about the run, not about being a diffusion model, so it lives here rather
than on the runner base. ``ModelLoader`` owns the one cached selection for the run.

Two questions, and they are not independent. Which adapter owns a format is decided by the load
contract; whether that adapter is *allowed* is decided by where FSDP will put its tensors, because
some storage forms cannot be sharded. A TorchAO Float8Tensor inside an FSDP2 block needs patches the
environment may not have, so the placement predicates below run before allocation and fail there
rather than at the first all-gather.

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

    def _format_value(self) -> str | None:
        contract = getattr(getattr(self, "loader", None), "load_contract", None)
        return None if contract is None else contract.requested_format.value

    def _uses_mxfp6_contract(self) -> bool:
        return self._format_value() in {"fp6", "fp4_fp6"}

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
        if format_name == "fp8":
            adapter = select_fp8_backend(contract, capabilities=self.fp8_capabilities)
            validate_torchao_fsdp2_patches(
                contract,
                capabilities=self.fp8_capabilities,
                required=self.places_torchao_tensor_subclass_under_fsdp2(adapter),
            )
            return adapter

        plan = self.loader.quantization_plan.gemm_plan
        is_low = plan is not None and format_name == plan.low
        adapter = select_format_backend(
            contract,
            capabilities=self.format_capabilities,
            # Only the format a step actually runs in bf16-vs-quantized pairs
            # can refuse the hybrid schedule; the companion never drives it.
            hybrid=is_low and self.model.config.use_hybrid_gemm_schedule,
        )
        validate_format_fsdp_placement(
            contract,
            adapter,
            capabilities=self.format_capabilities,
            required=(
                self.places_format_backend_under_fsdp2()
                if is_low
                else self.places_mxfp6_backend_under_fsdp2()
            ),
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

    def places_torchao_tensor_subclass_under_fsdp2(
        self,
        fp8_adapter,
        *,
        assume_torchao_fp8: bool = False,
    ) -> bool:
        """Whether configured FSDP2 blocks will contain TorchAO Float8Tensor.

        assume_torchao_fp8 answers the question before an adapter has been selected, which is how
        the selection itself avoids depending on its own result.
        """
        fsdp_target_paths = self._fsdp_target_paths()
        if not fsdp_target_paths:
            return False
        if self._uses_mxfp6_contract():
            return False

        config = self.model.config
        # Everything the run gives to FP8, which is the whole target set in a
        # pure FP8 run and only the held-back modules in a tiered one. Asking
        # `high_tier_targets` instead would answer "none" for a pure run, which
        # has no tier but does put Float8Tensor inside every wrapped block.
        plan = self.loader.quantization_plan.gemm_plan
        fp8_targets = set(plan.roots("fp8"))
        is_torchao = assume_torchao_fp8 or (
            fp8_adapter is not None and fp8_adapter.backend.value == "torchao"
        )
        if is_torchao and any(
            module_paths_overlap(target, fsdp_path)
            for target in fp8_targets
            for fsdp_path in fsdp_target_paths
        ):
            return True

        # An fp4 run still emits fp8 tensors wherever a carve-out or the hybrid schedule
        # holds a layer back from fp4, so fp4 targets count too once either is in play.
        fp4_can_emit_fp8 = bool(
            self._high_tier_scattered() or config.use_hybrid_gemm_schedule
        )
        return bool(
            plan.low == "fp4"
            and fp4_can_emit_fp8
            and any(
                module_paths_overlap(target, fsdp_path)
                for target in self.primary_targets()
                for fsdp_path in fsdp_target_paths
            )
        )

    def places_format_backend_under_fsdp2(self) -> bool:
        fsdp_target_paths = self._fsdp_target_paths()
        if not fsdp_target_paths:
            return False
        targets = set(self._format_entries())
        return any(
            module_paths_overlap(target, fsdp_path)
            for target in targets
            for fsdp_path in fsdp_target_paths
        )

    def places_mxfp6_backend_under_fsdp2(self) -> bool:
        """Whether pure/mixed MXFP6 creates packed parameters inside FSDP2."""

        fsdp_target_paths = self._fsdp_target_paths()
        if not fsdp_target_paths or not self._uses_mxfp6_contract():
            return False
        if self._format_value() == "fp6":
            fp6_targets = set(self._format_entries())
        else:
            fp6_targets = self.high_tier_targets()
            if self._high_tier_scattered():
                # A carve-out scatters the better format inside the low-tier
                # blocks too, so those components count as well.
                fp6_targets.update(self.primary_targets())
        return any(
            module_paths_overlap(target, fsdp_path)
            for target in fp6_targets
            for fsdp_path in fsdp_target_paths
        )

    def _high_tier_scattered(self) -> bool:
        """Whether the better format lands inside the primary format's targets.

        A whole-module high-tier target is visible in `high_tier_targets`; one
        carved out below a target root -- a block prefix, a leaf suffix -- is
        not, and a predicate reading only that set would answer "no high tier
        here" about a block that holds one.
        """
        return self.loader.quantization_plan.gemm_plan.splits_a_target

    def primary_targets(self) -> set:
        """The targets the run's primary format owns."""
        plan = self.loader.quantization_plan.gemm_plan
        return set(plan.roots(plan.low)) if plan.low else set()

    def high_tier_targets(self) -> set:
        """The targets held at the better format, which the primary one skips.

        Three FSDP predicates need this, and each used to rebuild it by
        subtracting the fp4 list from the fp8 one. The plan resolves it once,
        so they can just ask.
        """
        plan = self.loader.quantization_plan.gemm_plan
        return set(plan.roots(plan.high)) if plan.high else set()

    def requires_blockwise_fp8(self) -> bool:
        """Whether FP4 mode declares whole components owned only by FP8."""
        if self._uses_mxfp6_contract():
            return False
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None:
            return False
        return bool(self.high_tier_targets())

    def uses_blockwise_fp8(self) -> bool:
        contract = self.loader.load_contract
        if contract is None or self._uses_mxfp6_contract():
            return False
        if self.requires_blockwise_fp8():
            return True
        if self.places_torchao_tensor_subclass_under_fsdp2(None):
            return True
        if contract.requested_format.value == "fp8" and (
            self.places_torchao_tensor_subclass_under_fsdp2(
                None, assume_torchao_fp8=True
            )
        ):
            return True
        return (
            contract.materialization_mode.value != "eager"
            and contract.requested_format.value == "fp8"
        )

    def _format_entries(self):
        """The transformer targets this run's primary format owns.

        The text encoder is excluded: it is loaded and quantized by its own
        route, which the format backends never touch.
        """
        format_value = self.loader.load_contract.requested_format.value
        if format_value not in {"fp4", "fp4_fp6", "fp6", "int8"}:
            return ()
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None or plan.low is None:
            return ()
        return plan.roots(plan.low, component="transformer")

    def format_entries(self):
        """Public stable view used by eager placement."""

        return tuple(self._format_entries())

    def format_targets_for(self, component_name: str) -> tuple:
        """Primary-format targets under one component, with its prefix stripped."""
        plan = self.loader.quantization_plan.gemm_plan
        if plan is None:
            return ()
        return plan.relative_to(component_name, self._format_entries())

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
