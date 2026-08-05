"""Dependency-light contracts for checkpoint materialization and quantization."""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Mapping, Protocol, runtime_checkable


class QuantizationFormat(str, Enum):
    NONE = "none"
    FP8 = "fp8"
    FP4 = "fp4"
    FP8_FP4 = "fp8_fp4"
    INT8 = "int8"


class QuantizationBackend(str, Enum):
    NONE = "none"
    AITER = "aiter"
    TORCHAO = "torchao"


class MaterializationMode(str, Enum):
    EAGER = "eager"
    FSDP_META = "fsdp_meta"
    REPLICATED_META = "replicated_meta"


class ConstructionSeam(str, Enum):
    BUILD_TRANSFORMER = "_build_transformer"


class UnsupportedLoadContract(ValueError):
    """The runner cannot honor a requested load contract safely."""


@runtime_checkable
class CheckpointTensorReader(Protocol):
    """Small tensor-reading seam; discovery is intentionally independent of it."""

    def get_tensor(self, key: str):
        ...


@dataclass(frozen=True)
class LoadCapability:
    """What one runner can construct before weights are allocated."""

    fsdp_meta_transformers: tuple[str, ...] = ()
    replicated_meta_transformers: tuple[str, ...] = ()
    materialization_modes: FrozenSet[MaterializationMode] = frozenset(
        {MaterializationMode.EAGER}
    )
    construction_seam: ConstructionSeam | None = None
    quantization_formats: FrozenSet[QuantizationFormat] = frozenset(
        {QuantizationFormat.NONE}
    )
    quantization_backends: FrozenSet[QuantizationBackend] = frozenset(
        {QuantizationBackend.NONE}
    )
    quantization_contracts: FrozenSet[
        tuple[QuantizationFormat, QuantizationBackend]
    ] = frozenset({(QuantizationFormat.NONE, QuantizationBackend.NONE)})
    unsupported_reason: str | None = None

    @property
    def meta_transformers(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.fsdp_meta_transformers
                + self.replicated_meta_transformers
            )
        )

    @classmethod
    def meta(
        cls,
        *transformers: str,
        replicated: bool = False,
        quantization_formats=(),
        quantization_backends=(),
    ) -> "LoadCapability":
        modes = {MaterializationMode.EAGER, MaterializationMode.FSDP_META}
        if replicated:
            modes.add(MaterializationMode.REPLICATED_META)
        formats = frozenset(
            quantization_formats or {QuantizationFormat.NONE}
        )
        if (
            QuantizationFormat.FP8 in formats
            and QuantizationFormat.FP4 in formats
        ):
            formats = formats | {QuantizationFormat.FP8_FP4}
        backends = frozenset(
            quantization_backends or {QuantizationBackend.NONE}
        )
        contracts = {
            (format_, backend)
            for format_ in formats
            for backend in backends
            if (format_ is QuantizationFormat.NONE)
            == (backend is QuantizationBackend.NONE)
        }
        return cls(
            fsdp_meta_transformers=tuple(transformers),
            replicated_meta_transformers=(
                tuple(transformers) if replicated else ()
            ),
            materialization_modes=frozenset(modes),
            construction_seam=ConstructionSeam.BUILD_TRANSFORMER,
            quantization_formats=formats,
            quantization_backends=backends,
            quantization_contracts=frozenset(contracts),
        )

    @classmethod
    def unsupported(cls, reason: str) -> "LoadCapability":
        return cls(unsupported_reason=reason)

    @classmethod
    def for_runner(
        cls,
        model_capabilities,
        *,
        meta_transformers: tuple[str, ...] = (),
        replicated: bool = False,
        fsdp_strategy: Mapping[str, Mapping] | None = None,
        unsupported_reason: str | None = None,
    ) -> "LoadCapability":
        """Derive quantization support while keeping meta loading opt-in."""

        contracts = {
            (QuantizationFormat.NONE, QuantizationBackend.NONE)
        }
        if getattr(model_capabilities, "use_fp8_gemms", False):
            contracts.update(
                {
                    (QuantizationFormat.FP8, QuantizationBackend.AITER),
                    (QuantizationFormat.FP8, QuantizationBackend.TORCHAO),
                }
            )
        if getattr(model_capabilities, "use_fp4_gemms", False):
            contracts.update(
                {
                    (QuantizationFormat.FP4, QuantizationBackend.AITER),
                    (QuantizationFormat.FP4, QuantizationBackend.TORCHAO),
                }
            )
        if (
            getattr(model_capabilities, "use_fp8_gemms", False)
            and getattr(model_capabilities, "use_fp4_gemms", False)
        ):
            contracts.update(
                {
                    (
                        QuantizationFormat.FP8_FP4,
                        QuantizationBackend.AITER,
                    ),
                    (
                        QuantizationFormat.FP8_FP4,
                        QuantizationBackend.TORCHAO,
                    ),
                }
            )
        if getattr(model_capabilities, "use_int8_gemms", False):
            contracts.add(
                (QuantizationFormat.INT8, QuantizationBackend.TORCHAO)
            )

        modes = {MaterializationMode.EAGER}
        seam = None
        strategy = fsdp_strategy or {}
        fsdp_transformers = (
            tuple(
                name
                for name in meta_transformers
                if strategy.get(name, {}).get("wrap_attrs")
            )
            if getattr(
                model_capabilities, "fully_shard_degree", False
            )
            else ()
        )
        replicated_transformers = (
            tuple(meta_transformers) if replicated else ()
        )
        if fsdp_transformers:
            modes.add(MaterializationMode.FSDP_META)
        if fsdp_transformers or replicated_transformers:
            seam = ConstructionSeam.BUILD_TRANSFORMER
        if replicated_transformers:
            modes.add(MaterializationMode.REPLICATED_META)
        formats = frozenset(format_ for format_, _ in contracts)
        backends = frozenset(backend for _, backend in contracts)
        return cls(
            fsdp_meta_transformers=fsdp_transformers,
            replicated_meta_transformers=replicated_transformers,
            materialization_modes=frozenset(modes),
            construction_seam=seam,
            quantization_formats=formats,
            quantization_backends=backends,
            quantization_contracts=frozenset(contracts),
            unsupported_reason=unsupported_reason,
        )

    @classmethod
    def declare(
        cls,
        *meta_transformers: str,
        replicated: bool = False,
        unsupported_reason: str | None = None,
    ):
        """Class decorator deriving quantization support after class creation."""

        def decorate(runner_cls):
            runner_cls.load_capability = cls.for_runner(
                runner_cls.capabilities,
                meta_transformers=tuple(meta_transformers),
                replicated=replicated,
                fsdp_strategy=runner_cls.settings.fsdp_strategy,
                unsupported_reason=unsupported_reason,
            )
            return runner_cls

        return decorate


@dataclass(frozen=True)
class LoadContract:
    requested_format: QuantizationFormat
    selected_backend: QuantizationBackend
    materialization_mode: MaterializationMode


def select_effective_materialization_mode(
    config,
    *,
    world_size: int,
) -> MaterializationMode:
    """Apply the same runtime exclusions used by memory-efficient loading."""

    if (
        config.memory_efficient_sharding
        and config.fully_shard_degree > 1
    ):
        return MaterializationMode.FSDP_META
    splits_weights = (
        config.fully_shard_degree > 1
        or config.pipefusion_parallel_degree > 1
        or config.tensor_parallel_degree > 1
    )
    if (
        config.memory_efficient_replicated_load
        and world_size > 1
        and not splits_weights
    ):
        return MaterializationMode.REPLICATED_META
    return MaterializationMode.EAGER


def validate_materialization_contract(
    capability: LoadCapability,
    mode: MaterializationMode,
    fsdp_strategy: Mapping[str, Mapping],
    *,
    runner_name: str,
) -> None:
    if mode not in capability.materialization_modes:
        reason = (
            f": {capability.unsupported_reason}"
            if capability.unsupported_reason
            else ""
        )
        raise UnsupportedLoadContract(
            f"{runner_name} does not support {mode.value} materialization{reason}"
        )
    if mode is MaterializationMode.EAGER:
        return
    if capability.construction_seam is None:
        raise UnsupportedLoadContract(
            f"{runner_name} declares {mode.value} but no meta construction seam"
        )
    components = (
        capability.fsdp_meta_transformers
        if mode is MaterializationMode.FSDP_META
        else capability.replicated_meta_transformers
    )
    if not components:
        raise UnsupportedLoadContract(
            f"{runner_name} declares {mode.value} but no meta transformers"
        )
    for component in components:
        strategy = fsdp_strategy.get(component)
        if strategy is None:
            raise UnsupportedLoadContract(
                f"{runner_name} declares meta transformer '{component}', but it is "
                "missing from fsdp_strategy"
            )
        if not strategy.get("wrap_attrs"):
            raise UnsupportedLoadContract(
                f"{runner_name} meta transformer '{component}' needs non-empty "
                "fsdp_strategy wrap_attrs"
            )


def select_load_contract(
    *,
    requested_format: QuantizationFormat,
    selected_backend: QuantizationBackend,
    materialization_mode: MaterializationMode,
    capability: LoadCapability,
    fsdp_strategy: Mapping[str, Mapping],
    runner_name: str,
) -> LoadContract:
    """Validate a complete contract before model allocation or collectives."""

    if (
        requested_format,
        selected_backend,
    ) not in capability.quantization_contracts:
        raise UnsupportedLoadContract(
            f"{selected_backend.name} backend for {requested_format.name} is not "
            f"declared by {runner_name}"
        )
    if (requested_format is QuantizationFormat.NONE) != (
        selected_backend is QuantizationBackend.NONE
    ):
        raise UnsupportedLoadContract(
            f"{runner_name} cannot pair {requested_format.name} with "
            f"{selected_backend.name}"
        )
    validate_materialization_contract(
        capability,
        materialization_mode,
        fsdp_strategy,
        runner_name=runner_name,
    )
    return LoadContract(
        requested_format=requested_format,
        selected_backend=selected_backend,
        materialization_mode=materialization_mode,
    )


def select_runtime_quantization(
    config,
    *,
    aiter_fp8_active: bool,
    cuda_active: bool,
) -> tuple[QuantizationFormat, QuantizationBackend]:
    """Translate current flags/platform selection into the explicit contract."""

    if config.use_int8_gemms and (
        config.use_fp8_gemms or config.use_fp4_gemms
    ):
        others = "FP8 + FP4" if (
            config.use_fp8_gemms and config.use_fp4_gemms
        ) else ("FP8" if config.use_fp8_gemms else "FP4")
        raise UnsupportedLoadContract(
            f"INT8 cannot be combined with {others}"
        )
    if config.use_fp8_gemms and config.use_fp4_gemms:
        format_ = QuantizationFormat.FP8_FP4
    elif config.use_fp8_gemms:
        format_ = QuantizationFormat.FP8
    elif config.use_fp4_gemms:
        format_ = QuantizationFormat.FP4
    elif config.use_int8_gemms:
        format_ = QuantizationFormat.INT8
    else:
        return QuantizationFormat.NONE, QuantizationBackend.NONE

    if format_ is QuantizationFormat.FP8:
        backend = (
            QuantizationBackend.AITER
            if aiter_fp8_active
            else QuantizationBackend.TORCHAO
        )
    elif format_ in (
        QuantizationFormat.FP4,
        QuantizationFormat.FP8_FP4,
    ):
        backend = (
            QuantizationBackend.TORCHAO
            if cuda_active
            else QuantizationBackend.AITER
        )
    else:
        backend = QuantizationBackend.TORCHAO
    return format_, backend
