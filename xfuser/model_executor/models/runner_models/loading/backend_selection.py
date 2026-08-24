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

from .format_backends import module_path_is_covered, module_paths_overlap


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
        if self.uses_blockwise_fp8():
            _ = self.blockwise_fp8

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
    def format(self):
        """Primary FP4/INT8 implementation, validated before allocation."""
        contract = self.loader.load_contract
        if contract is None or contract.requested_format.value not in {
            "fp4",
            "fp8_fp4",
            "int8",
        }:
            return None
        from .format_backends import (
            probe_format_backend_capabilities,
            select_format_backend,
            validate_format_fsdp_placement,
        )

        capabilities = probe_format_backend_capabilities()
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
        if format_value in {"fp4", "fp8_fp4"}:
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

        settings, config = self.model.settings, self.model.config
        fp4_targets = set(settings.fp4_gemm_module_list or ())
        fp8_only_targets = {
            target
            for target in self.loader.quantization_plan.module_list()
            if not any(
                module_path_is_covered(target, fp4_target)
                for fp4_target in fp4_targets
            )
        }
        is_torchao = assume_torchao_fp8 or (
            fp8_adapter is not None and fp8_adapter.backend.value == "torchao"
        )
        if is_torchao and any(
            module_paths_overlap(target, fsdp_path)
            for target in fp8_only_targets
            for fsdp_path in fsdp_target_paths
        ):
            return True

        # An fp4 run still emits fp8 tensors wherever a precision override or the hybrid schedule
        # holds a layer back from fp4, so fp4 targets count too once any of those is in play.
        fp4_can_emit_fp8 = bool(
            settings.fp8_precision_overrides
            or settings.fp8_precision_override_suffixes
            or config.use_hybrid_gemm_schedule
        )
        return bool(
            config.use_fp4_gemms
            and fp4_can_emit_fp8
            and any(
                module_paths_overlap(target, fsdp_path)
                for target in fp4_targets
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

    def requires_blockwise_fp8(self) -> bool:
        """Whether FP4 mode declares whole components owned only by FP8."""
        if not self.model.config.use_fp4_gemms:
            return False
        fp4_targets = set(self.model.settings.fp4_gemm_module_list or ())
        return any(
            not any(
                module_path_is_covered(target, fp4_target)
                for fp4_target in fp4_targets
            )
            for target in self.loader.quantization_plan.module_list()
        )

    def uses_blockwise_fp8(self) -> bool:
        contract = self.loader.load_contract
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
        """This run's FP4/INT8 target list, whichever format the contract asked for."""
        format_value = self.loader.load_contract.requested_format.value
        if format_value in {"fp4", "fp8_fp4"}:
            return self.loader.quantization_plan.module_list("fp4")
        if format_value == "int8":
            return self.loader.quantization_plan.module_list("int8")
        return ()

    def format_targets_for(self, component_name: str) -> tuple:
        """This run's FP4/INT8 targets under one component, with the component prefix stripped."""
        prefix = f"{component_name}."
        return tuple(
            "" if entry == component_name else entry[len(prefix) :]
            for entry in self._format_entries()
            if entry == component_name or entry.startswith(prefix)
        )

    def transformer_adapter(self, component_name: str):
        """The adapter and component-relative targets owning one transformer.

        Format targets win over fp8 ones: a component listed for fp4 or int8 is quantized by the
        format backend, and its fp8 entries describe the remainder that backend leaves behind.
        """
        format_targets = self.format_targets_for(component_name)
        if format_targets:
            return self.format, format_targets
        fp8_targets = tuple(
            self.loader.quantization_plan.targets_for(component_name)
        )
        if not fp8_targets:
            return None, ()
        return self.fp8_adapter_for_contract(), fp8_targets
