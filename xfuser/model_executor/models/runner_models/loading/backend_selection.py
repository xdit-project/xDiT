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

from .format_backends import module_paths_overlap


class QuantizationBackends:
    """The adapters this run quantizes through, and the placement rules that gate them.

    Holds the ``xFuserModel`` to read its load contract, settings, and config. Keeps no state beyond
    the adapter caches, so it stays correct across the settings edits some runners make while
    loading.
    """

    def __init__(self, loader) -> None:
        self.loader = loader
        self.model = loader.model

    def preflight(self) -> None:
        """Resolve and validate only the adapters required by this run."""
        _ = self.fp8
        _ = self.format
        if self._uses_mxfp6_contract():
            _ = self.fp6
        if self.uses_blockwise_fp8():
            _ = self.blockwise_fp8

    def _format_value(self) -> str | None:
        contract = getattr(getattr(self, "loader", None), "load_contract", None)
        return None if contract is None else contract.requested_format.value

    def _uses_mxfp6_contract(self) -> bool:
        return self._format_value() in {"fp6", "fp4_fp6"}

    @functools.cached_property
    def fp8(self):
        """The selected FP8 implementation, validated before allocation."""
        contract = self.loader.load_contract
        if contract is None or contract.requested_format.value != "fp8":
            return None
        from .fp8_backends import (
            probe_fp8_backend_capabilities,
            select_fp8_backend,
        )

        return select_fp8_backend(
            contract,
            capabilities=probe_fp8_backend_capabilities(),
        )

    @functools.cached_property
    def format_capabilities(self):
        """Probe format capabilities once, and only include MXFP6 when requested."""

        from .format_backends import probe_format_backend_capabilities

        return probe_format_backend_capabilities(
            require_mxfp6=self._uses_mxfp6_contract()
        )

    @functools.cached_property
    def format(self):
        """Primary FP4/FP6/INT8 implementation, validated before allocation."""
        contract = self.loader.load_contract
        if contract is None or contract.requested_format.value not in {
            "fp4",
            "fp6",
            "fp4_fp6",
            "int8",
        }:
            return None
        from .format_backends import (
            select_format_backend,
            validate_format_fsdp_placement,
        )

        capabilities = self.format_capabilities
        adapter = select_format_backend(
            contract,
            capabilities=capabilities,
            hybrid=self.model.config.use_hybrid_gemm_schedule,
        )
        validate_format_fsdp_placement(
            contract,
            adapter,
            capabilities=capabilities,
            required=self.places_format_backend_under_fsdp2(),
        )
        return adapter

    @functools.cached_property
    def fp6(self):
        """MXFP6 owner for pure mode or the mixed mode's FP8 remainder."""

        contract = self.loader.load_contract
        if contract is None:
            return None
        format_value = contract.requested_format.value
        if format_value == "fp6":
            return self.format
        if format_value != "fp4_fp6":
            return None
        from .format_backends import (
            select_mxfp6_backend,
            validate_format_fsdp_placement,
        )

        adapter = select_mxfp6_backend(
            contract,
            capabilities=self.format_capabilities,
        )
        validate_format_fsdp_placement(
            contract,
            adapter,
            capabilities=self.format_capabilities,
            required=self.places_mxfp6_backend_under_fsdp2(),
        )
        return adapter

    def adapter_for(self, format_name: str):
        """The converter that owns one format.

        A run can name any format at either tier, so callers ask for the one
        they resolved rather than picking between named attributes. `fp8` is
        the pure-FP8 implementation when the contract requested exactly FP8,
        and the blockwise converter when FP8 is one tier of a hybrid load.
        """
        if format_name == "fp6":
            return self.fp6
        if format_name == "fp8":
            return self.fp8 or self.blockwise_fp8
        return self.format

    @functools.cached_property
    def blockwise_fp8(self):
        """FP8 converter for pure FP8 and FP8-only portions of hybrid loads."""
        contract = self.loader.load_contract
        if contract is None:
            return None
        from .fp8_backends import (
            probe_fp8_backend_capabilities,
            select_blockwise_fp8_backend,
            validate_torchao_fsdp2_patches,
        )

        capabilities = probe_fp8_backend_capabilities()
        adapter = select_blockwise_fp8_backend(contract, capabilities=capabilities)
        validate_torchao_fsdp2_patches(
            contract,
            capabilities=capabilities,
            required=self.places_torchao_tensor_subclass_under_fsdp2(adapter),
        )
        return adapter

    def fp8_adapter_for_contract(self):
        """The backend that owns FP8 storage for the active contract.

        FP8 storage has two owners depending on the format requested: a pure fp8 run quantizes
        through the fp8 adapter, while an fp4 run quantizes its fp8-only remainder through the
        blockwise one.
        """
        contract = self.loader.load_contract
        if contract is None:
            return None
        format_value = contract.requested_format.value
        if format_value == "fp8":
            return self.fp8
        if format_value == "fp4":
            return self.blockwise_fp8
        return None

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

        Format targets win over fp8 ones: a component listed for fp4 or int8 is quantized by the
        format backend, and its fp8 entries describe the remainder that backend leaves behind.
        """
        format_targets = self.format_targets_for(component_name)
        if format_targets:
            return self.format, format_targets
        plan = self.loader.quantization_plan.gemm_plan
        fp8_targets = (
            plan.relative_to(component_name, plan.roots("fp8"))
            if plan is not None
            else ()
        )
        if not fp8_targets:
            return None, ()
        if self._format_value() == "fp4_fp6":
            return self.fp6, fp8_targets
        return self.fp8_adapter_for_contract(), fp8_targets
