"""Dependency-light contracts for checkpoint materialization and quantization."""

from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import FrozenSet, Mapping, Protocol, runtime_checkable


class QuantizationFormat(str, Enum):
    NONE = "none"
    FP8 = "fp8"
    FP4 = "fp4"
    FP6 = "fp6"
    INT8 = "int8"


#: Which implementations can store each format, best first. The one table that
#: names a format: a new one is a row here plus a registered adapter class, and
#: nothing between them changes. Kept in this dependency-light module so both
#: the registry and the per-model contract list read the same rows rather than
#: each keeping their own copy.
IMPL_PREFERENCE = {
    "fp8": ("aiter", "torchao"),
    "fp4": ("torchao", "aiter"),
    "fp6": ("aiter",),
    "int8": ("torchao",),
}


class QuantizationBackend(str, Enum):
    NONE = "none"
    AITER = "aiter"
    TORCHAO = "torchao"


class MaterializationMode(str, Enum):
    EAGER = "eager"
    FSDP_META = "fsdp_meta"
    REPLICATED_META = "replicated_meta"


class ConstructionSeam(str, Enum):
    LOAD_TRANSFORMER = "loader.load_transformer"


class LoadRoute(Flag):
    """Independent loading routes a runner has explicitly verified."""

    NONE = 0
    STANDARD_COLLECTIVES = auto()
    LOCAL_BLOCKWISE = auto()


STANDARD_LOAD_ROUTES = (
    LoadRoute.STANDARD_COLLECTIVES | LoadRoute.LOCAL_BLOCKWISE
)


@dataclass(frozen=True)
class LoadSupport:
    """Static, model-local loading support resolved after customization."""

    meta_transformers: tuple[str, ...] = ()
    meta_text_encoders: tuple[str, ...] = ()
    replicated_meta: bool = False
    routes: LoadRoute = LoadRoute.NONE


class UnsupportedLoadContract(ValueError):
    """The runner cannot honor a requested load contract safely."""


@runtime_checkable
class CheckpointTensorReader(Protocol):
    """Small tensor-reading seam; discovery is intentionally independent of it."""

    def get_tensor(self, key: str): ...


@dataclass(frozen=True)
class LoadDeclaration:
    """What one runner has declared it can construct before weights are allocated.

    This is a derived instance view over the three objects a runner carries:
    ``quantization_contracts`` comes from ``ModelCapabilities`` (its fp8/fp4/int8 flags)
    and ``materialization_modes`` from the final ``ModelSettings.fsdp_strategy``. Static
    component intent, replicated-meta support, and load routes come from the runner's
    frozen ``LoadSupport``.

    The default is unsupported, because memory-efficient loading is opt-in twice over: the
    user passes a flag defaulting to false, and the runner must have declared support. An
    inherited permissive default would opt a model into the path behind a flag set for a
    different model.
    """

    fsdp_meta_transformers: tuple[str, ...] = ()
    replicated_meta_transformers: tuple[str, ...] = ()
    local_meta_transformers: tuple[str, ...] = ()
    meta_text_encoders: tuple[str, ...] = ()
    materialization_modes: FrozenSet[MaterializationMode] = frozenset(
        {MaterializationMode.EAGER}
    )
    construction_seam: ConstructionSeam | None = None
    routes: LoadRoute = STANDARD_LOAD_ROUTES
    quantization_formats: FrozenSet[QuantizationFormat] = frozenset(
        {QuantizationFormat.NONE}
    )
    quantization_backends: FrozenSet[QuantizationBackend] = frozenset(
        {QuantizationBackend.NONE}
    )
    quantization_contracts: FrozenSet[
        tuple[QuantizationFormat, QuantizationBackend]
    ] = frozenset({(QuantizationFormat.NONE, QuantizationBackend.NONE)})

    @property
    def meta_transformers(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.fsdp_meta_transformers + self.replicated_meta_transformers
            )
        )

    @property
    def all_meta_transformers(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(self.meta_transformers + self.local_meta_transformers)
        )

    @classmethod
    def meta(
        cls,
        *transformers: str,
        replicated: bool = False,
        quantization_formats=(),
        quantization_backends=(),
    ) -> "LoadDeclaration":
        modes = {MaterializationMode.EAGER, MaterializationMode.FSDP_META}
        if replicated:
            modes.add(MaterializationMode.REPLICATED_META)
        formats = frozenset(quantization_formats or {QuantizationFormat.NONE})
        backends = frozenset(quantization_backends or {QuantizationBackend.NONE})
        # From the one table, like `for_runner`: this was the same rows written
        # out again, with a hand-written carve-out saying FP6 is AITER-only --
        # which IMPL_PREFERENCE already says, and says for every other format
        # too, so the restated version was both a duplicate and incomplete.
        contracts = {
            (format_, backend)
            for format_ in formats
            for backend in backends
            if backend.value in IMPL_PREFERENCE.get(format_.value, ())
        }
        if QuantizationFormat.NONE in formats and QuantizationBackend.NONE in backends:
            contracts.add((QuantizationFormat.NONE, QuantizationBackend.NONE))
        return cls(
            fsdp_meta_transformers=tuple(transformers),
            replicated_meta_transformers=(tuple(transformers) if replicated else ()),
            local_meta_transformers=tuple(transformers),
            materialization_modes=frozenset(modes),
            construction_seam=ConstructionSeam.LOAD_TRANSFORMER,
            quantization_formats=formats,
            quantization_backends=backends,
            quantization_contracts=frozenset(contracts),
        )

    @classmethod
    def for_runner(
        cls,
        model_capabilities,
        *,
        load_support: LoadSupport = LoadSupport(),
        fsdp_strategy: Mapping[str, Mapping] | None = None,
    ) -> "LoadDeclaration":
        """Resolve one model's static spec against final instance settings."""

        # Read off the field rather than through ModelCapabilities' accessor:
        # this takes any object that describes a model's capabilities, and a
        # caller that declares no GEMM formats supports none.
        supported = frozenset(getattr(model_capabilities, "gemm_formats", ()) or ())

        # Generated from the one table rather than restated per format: the
        # hand-written version was the same rows a second time, and a format
        # added to one and not the other would have been accepted by the
        # registry and then refused by the contract, or the reverse.
        contracts = {(QuantizationFormat.NONE, QuantizationBackend.NONE)}
        contracts.update(
            (QuantizationFormat(format_name), QuantizationBackend(impl))
            for format_name in supported
            for impl in IMPL_PREFERENCE.get(format_name, ())
        )

        modes = {MaterializationMode.EAGER}
        seam = None
        strategy = fsdp_strategy or {}
        standard_collectives = bool(
            load_support.routes & LoadRoute.STANDARD_COLLECTIVES
        )
        local_transformers = (
            tuple(
                name
                for name in load_support.meta_transformers
                if strategy.get(name, {}).get("wrap_attrs")
            )
            if load_support.routes & LoadRoute.LOCAL_BLOCKWISE
            else ()
        )
        fsdp_transformers = (
            tuple(
                name
                for name in load_support.meta_transformers
                if strategy.get(name, {}).get("wrap_attrs")
            )
            if standard_collectives
            and getattr(model_capabilities, "fully_shard_degree", False)
            else ()
        )
        replicated_transformers = (
            tuple(load_support.meta_transformers)
            if standard_collectives and load_support.replicated_meta
            else ()
        )
        if fsdp_transformers:
            modes.add(MaterializationMode.FSDP_META)
        if fsdp_transformers or replicated_transformers:
            seam = ConstructionSeam.LOAD_TRANSFORMER
        if replicated_transformers:
            modes.add(MaterializationMode.REPLICATED_META)
        formats = frozenset(format_ for format_, _ in contracts)
        backends = frozenset(backend for _, backend in contracts)
        return cls(
            fsdp_meta_transformers=fsdp_transformers,
            replicated_meta_transformers=replicated_transformers,
            local_meta_transformers=local_transformers,
            meta_text_encoders=tuple(load_support.meta_text_encoders),
            materialization_modes=frozenset(modes),
            construction_seam=seam,
            routes=load_support.routes,
            quantization_formats=formats,
            quantization_backends=backends,
            quantization_contracts=frozenset(contracts),
        )


@dataclass(frozen=True)
class LoadContract:
    requested_format: QuantizationFormat
    selected_backend: QuantizationBackend
    materialization_mode: MaterializationMode


def _splits_weights(config) -> bool:
    return (
        config.fully_shard_degree > 1
        or config.pipefusion_parallel_degree > 1
        or config.tensor_parallel_degree > 1
    )


def select_effective_materialization_mode(
    config,
    *,
    world_size: int,
) -> MaterializationMode:
    """Apply the same runtime exclusions used by memory-efficient loading."""

    if config.memory_efficient_sharding and config.fully_shard_degree > 1:
        return MaterializationMode.FSDP_META
    if (
        config.memory_efficient_replicated_load
        and world_size > 1
        and not _splits_weights(config)
    ):
        return MaterializationMode.REPLICATED_META
    return MaterializationMode.EAGER


def assert_requested_materialization_is_honoured(config, *, world_size: int) -> None:
    """Refuse a memory-efficient request that the mode selection would quietly drop.

    Replicated meta loading holds one full copy per rank, so it is defined only when nothing else
    splits the weights. Asking for it alongside a degree that does split them selects an eager
    load, which without this refusal reads as the feature being enabled and doing nothing.

    A single-rank run is not such a case and is left alone: there is no peer to fill, so an eager
    load is correct, and refusing would stop the same command line working on one GPU.
    """
    if not config.memory_efficient_replicated_load:
        return
    if config.fully_shard_degree > 1:
        raise UnsupportedLoadContract(
            "--memory_efficient_replicated_load conflicts with --fully_shard_degree "
            f"{config.fully_shard_degree}: replicated loading keeps a whole copy per rank, "
            "while sharding splits it. Use --memory_efficient_sharding to shard."
        )
    if _splits_weights(config):
        splitters = {
            "--pipefusion_parallel_degree": config.pipefusion_parallel_degree,
            "--tensor_parallel_degree": config.tensor_parallel_degree,
        }
        named = ", ".join(f"{flag} {value}" for flag, value in splitters.items() if value > 1)
        raise UnsupportedLoadContract(
            f"--memory_efficient_replicated_load conflicts with {named}: that degree already "
            "splits the weights, so there is no replicated copy to fill."
        )


def assert_offload_is_compatible_with_sharding(config) -> None:
    """Refuse the offload modes that cannot move an FSDP2-sharded parameter.

    Sharding replaces each parameter with a DTensor, and two of the three offload
    modes reach through that abstraction. Group offloading asks every parameter
    whether it is pinned, and torch registers no sharding strategy for
    aten.is_pinned. Sequential offloading rebuilds each parameter as it moves it,
    which a DTensor cannot survive: it needs a spec that a plain tensor does not
    carry. Both surface deep inside the hook, mid-denoise, long after the load
    the operator was watching.

    Whole-model offload moves entire components rather than reaching inside them,
    and was measured working on top of the blockwise fill, so it is allowed.
    """

    if config.fully_shard_degree <= 1:
        return
    if getattr(config, "enable_group_cpu_offload", False):
        raise UnsupportedLoadContract(
            "--enable_group_cpu_offload cannot be combined with --fully_shard_degree "
            f"{config.fully_shard_degree}: the group hook asks each parameter whether "
            "it is pinned, and torch has no sharding strategy for aten.is_pinned. Use "
            "--enable_model_cpu_offload, which moves whole components, or drop the "
            "sharding degree."
        )
    if getattr(config, "enable_sequential_cpu_offload", False):
        raise UnsupportedLoadContract(
            "--enable_sequential_cpu_offload cannot be combined with "
            f"--fully_shard_degree {config.fully_shard_degree}: it rebuilds each "
            "parameter as it moves it, and a sharded parameter cannot be rebuilt "
            "without its DTensor spec. Use --enable_model_cpu_offload, which moves "
            "whole components, or drop the sharding degree."
        )


def validate_materialization_contract(
    declaration: LoadDeclaration,
    mode: MaterializationMode,
    fsdp_strategy: Mapping[str, Mapping],
    *,
    runner_name: str,
) -> None:
    if mode not in declaration.materialization_modes:
        raise UnsupportedLoadContract(
            f"{runner_name} does not support {mode.value} materialization"
        )
    if mode is MaterializationMode.EAGER:
        return
    if not (declaration.routes & LoadRoute.STANDARD_COLLECTIVES):
        raise UnsupportedLoadContract(
            f"{runner_name} does not support {mode.value} materialization: "
            "the runner does not declare the standard collective load route"
        )
    if declaration.construction_seam is None:
        raise UnsupportedLoadContract(
            f"{runner_name} declares {mode.value} but no meta construction seam"
        )
    components = (
        declaration.fsdp_meta_transformers
        if mode is MaterializationMode.FSDP_META
        else declaration.replicated_meta_transformers
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
    declaration: LoadDeclaration,
    fsdp_strategy: Mapping[str, Mapping],
    runner_name: str,
) -> LoadContract:
    """Validate a complete contract before model allocation or collectives."""

    if (
        requested_format,
        selected_backend,
    ) not in declaration.quantization_contracts:
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
        declaration,
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
    spec,
    *,
    impl_for,
    why_not=None,
) -> tuple[QuantizationFormat, QuantizationBackend]:
    """Name the contract: the run's low format, and what stores it here.

    The high tier gets no contract of its own. It is resolved by name through
    the same registry wherever the plan puts it, so a pair is never a contract
    in itself and no list of acceptable pairs has to be kept anywhere.
    """

    if spec.low in (None, "none"):
        return QuantizationFormat.NONE, QuantizationBackend.NONE
    impl = impl_for(spec.low)
    if impl is None:
        # The same measured reasons the adapter refusal carries: "unavailable"
        # on its own sends the reader to the source to find out which probe
        # failed and why.
        detail = f": {why_not(spec.low)}" if why_not is not None else ""
        raise UnsupportedLoadContract(
            f"--gemm_quantization {spec} asks for {spec.low}, which nothing "
            f"here can store{detail}"
        )
    return QuantizationFormat(spec.low), QuantizationBackend(impl)
