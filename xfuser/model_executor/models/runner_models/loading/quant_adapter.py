"""One adapter type for every quantization format, and how a load describes itself.

An adapter names one concrete thing: a *format* stored by an *implementation*.
"fp8 via aiter" is block-128 scales through ``gemm_a8w8_blockscale``; "fp8 via
torchao" is per-tensor dynamic scaling. Same format, different implementations,
so the pair is what identifies an adapter -- never the format alone.

Everything an implementation differs by is declared here as an attribute, so a
caller never asks which format it is holding. Which side of the device move it
converts on, whether it can quantize on the way in from disk, what kind of
parameter it leaves behind: all data. The only format-specific code is the
layer each adapter finally installs.

``materialization_mode`` is *derived*, not declared. It follows from what the
implementation can do, what the run needs, and whether the declaration narrows
to particular leaves -- see ``prepare_native_load``.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Tuple, Type


def module_paths_overlap(left: str, right: str) -> bool:
    """Whether two module paths are equal or one is a dotted ancestor."""

    if not left or not right:
        return True
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def module_path_is_covered(path: str, owner: str) -> bool:
    """Whether ``owner`` is the same dotted path as, or an ancestor of, ``path``."""

    return not owner or path == owner or path.startswith(f"{owner}.")


@dataclass(frozen=True)
class FormatCapability:
    """What one implementation of one format can do on this machine.

    One record per ``(format, implementation)``. Every question the loading
    layer asks about a format -- can this machine store it, can it be quantized
    on the way in from disk, can FSDP2 shard what it leaves behind -- is a field
    here, so adding an implementation is adding a record rather than four more
    flat fields named after it.

    Each ``reason`` is what the probe measured, kept so a refusal can say what
    was actually missing rather than "unavailable".
    """

    available: bool = False
    reason: str | None = None
    #: Per-weight quantization through the framework's own checkpoint loader.
    streams: bool = False
    streams_reason: str | None = None
    #: The same for a text encoder, which comes in through a different loader.
    te_streams: bool = False
    te_streams_reason: str | None = None
    #: Whether what this leaves in a module survives being sharded by FSDP2.
    #: A plain packed parameter does; a tensor subclass needs patches.
    fsdp_safe: bool = False
    fsdp_reason: str | None = None


@dataclass(frozen=True)
class Capabilities:
    """Every ``(format, implementation)`` pair this machine was probed for."""

    records: Mapping[Tuple[str, str], FormatCapability] = field(default_factory=dict)

    def of(self, format_name: str, impl: str) -> FormatCapability:
        """This machine's record for one pair, absent meaning "never probed"."""

        return self.records.get(
            (format_name, impl),
            FormatCapability(
                reason=f"no {impl} implementation of {format_name} was probed"
            ),
        )

    def merged(self, other: "Capabilities") -> "Capabilities":
        return Capabilities({**self.records, **other.records})


#: Every adapter class, keyed by the pair it stores. Populated by ``stores``
#: at import, so adding an implementation is adding a decorated class.
REGISTRY: dict[Tuple[str, str], Type] = {}


def stores(format_name: str, impl: str):
    """Register the one class that stores `format_name` through `impl`."""

    def register(cls):
        cls.format_name = format_name
        cls.impl = impl
        REGISTRY[(format_name, impl)] = cls
        return cls

    return register


def build_adapter(format_name: str, impl: str, *, capability, hybrid: bool = False):
    """The adapter for one pair, built from what this machine measured.

    Uniform: no class gets its flags from anywhere but its own record, and the
    only thing that can refuse a pair is the record being unavailable or the
    implementation declaring it cannot drive a per-step schedule.
    """

    from .contracts import (
        QuantizationBackend,
        QuantizationFormat,
        UnsupportedLoadContract,
    )

    cls = REGISTRY.get((format_name, impl))
    if cls is None:
        raise UnsupportedLoadContract(
            f"no {impl} implementation of {format_name} exists"
        )
    if not capability.available:
        reason = capability.reason or "backend unavailable"
        raise UnsupportedLoadContract(
            f"{impl} {format_name} is unavailable on this machine: {reason}"
        )
    if hybrid and not cls.builds_one_layer():
        raise UnsupportedLoadContract(
            f"{impl} {format_name} cannot install one layer at a time, so it "
            "cannot drive a low/high GEMM schedule; drop "
            "--use_hybrid_gemm_schedule or name a format that can"
        )
    return cls(
        backend=QuantizationBackend(impl),
        format_=QuantizationFormat(format_name),
        native_transformer_streaming=capability.streams,
        native_unavailable_reason=capability.streams_reason,
        native_text_encoder_streaming=capability.te_streams,
        text_encoder_unavailable_reason=capability.te_streams_reason,
    )


def validate_fsdp_placement(adapter, *, capability, required: bool) -> None:
    """Refuse a placement whose stored parameter kind FSDP2 cannot shard.

    One rule for every implementation: the adapter says what it leaves in a
    wrapped block, the record says whether that survives sharding here. No
    ladder, because there is nothing to dispatch on.
    """

    from .contracts import UnsupportedLoadContract

    if not required or adapter is None or capability.fsdp_safe:
        return
    suffix = f": {capability.fsdp_reason}" if capability.fsdp_reason else ""
    raise UnsupportedLoadContract(
        f"{adapter.parameter_semantics} stored by {adapter.impl} "
        f"{adapter.format_name} cannot be placed under FSDP2{suffix}"
    )


class TargetMappingUnavailable(RuntimeError):
    """The config-built model cannot safely express xDiT target prefixes."""


@dataclass(frozen=True)
class LinearOwnership:
    exclusions: tuple[str, ...]
    streamed: tuple[str, ...]
    residual: tuple[str, ...] = ()


def derive_linear_ownership(
    model,
    targets,
    *,
    min_layer_size: int = 0,
    residual_match: Callable[[str], bool] | None = None,
    is_linear=None,
) -> LinearOwnership:
    """Classify linear leaves for native streaming and post-load ownership.

    An empty target means the component itself, so it covers every leaf -- the
    form ``relative_to`` produces for a whole-component target.
    """

    if is_linear is None:
        from torch import nn

        def is_linear(module):
            return isinstance(module, nn.Linear)

    targets = tuple(dict.fromkeys(targets))
    for target in targets:
        try:
            model.get_submodule(target)
        except (AttributeError, KeyError) as exc:
            raise TargetMappingUnavailable(
                f"target mapping unavailable: model structure is missing '{target}'"
            ) from exc

    def targeted(name):
        return any(
            not target or name == target or name.startswith(f"{target}.")
            for target in targets
        )

    exclusions = []
    streamed = []
    residual = []
    for name, module in model.named_modules():
        if not name or not is_linear(module):
            continue
        too_small = (
            min_layer_size > 0
            and min(module.in_features, module.out_features) < min_layer_size
        )
        if not targeted(name) or too_small:
            exclusions.append(name)
        elif residual_match is not None and residual_match(name):
            exclusions.append(name)
            residual.append(name)
        else:
            streamed.append(name)
    return LinearOwnership(
        exclusions=tuple(exclusions),
        streamed=tuple(streamed),
        residual=tuple(residual),
    )


@dataclass(frozen=True)
class QuantLoadDescriptor:
    """What one component's quantized load will do, for logging and ownership."""

    requested_format: str
    selected_backend: str
    storage_semantics: str
    materialization_mode: str
    parameter_semantics: str
    auxiliary_state_semantics: str
    trainability: str
    serialization: str
    fallback_reason: str | None = None
    component_name: str = "transformer"

    def log_message(self) -> str:
        message = (
            f"{self.component_name} quantization: "
            f"requested={self.requested_format}, "
            f"backend={self.selected_backend}, "
            f"storage={self.storage_semantics}, "
            f"materialization={self.materialization_mode}, "
            f"parameters={self.parameter_semantics}, "
            f"auxiliary={self.auxiliary_state_semantics}, "
            f"trainability={self.trainability}, "
            f"serialization={self.serialization}"
        )
        if self.fallback_reason:
            message += f"; fallback={self.fallback_reason}"
        return message


@dataclass(frozen=True)
class PreparedQuantLoad:
    descriptor: QuantLoadDescriptor
    quantization_config: object | None = None
    streamed_targets: tuple[str, ...] = ()
    residual_targets: tuple[str, ...] = ()


class QuantAdapter:
    """One format stored by one implementation.

    Subclasses declare what they are and install their own layer; nothing above
    them branches on which format that is.
    """

    #: How the weights are stored numerically, for the log.
    storage_semantics = ""
    #: What FSDP2 would find in a wrapped block. A plain parameter shards; a
    #: tensor subclass needs patches the environment may not have.
    parameter_semantics = "tensor_subclass_parameter"
    auxiliary_state_semantics = "backend_managed"
    trainability = "inference_only"
    serialization = "torchao_version_dependent"
    #: Leaves smaller than this in either dimension are left alone.
    min_layer_size = 0
    #: Whether this converter can hold selected leaves at another format while
    #: it converts the rest of a block.
    supports_precision_overrides = False
    #: Whether a native streaming config can still express the run when the
    #: hybrid schedule is on -- it cannot if streaming would take ownership of
    #: leaves the hybrid wrapper needs to keep in two precisions.
    streams_under_hybrid = True
    #: AITER rewrites a module on the host; torchao swaps in subclasses that
    #: want their final device. Decides which side of ``pipe.to`` a walk runs.
    converts_before_device_move = False
    #: Whether the checkpoint can be loaded straight to host memory because this
    #: converter packs there anyway. Only meaningful under CPU offload, where it
    #: saves a device round trip.
    loads_to_host_under_offload = False
    #: Whether a meta-built component already has the layout this stores, so a
    #: broadcast or per-block fill can quantize on the way in. A packed weight
    #: plus a scale buffer does; a tensor subclass has to be converted after,
    #: which the memory-efficient FSDP path refuses.
    meta_layout_matches_storage = False
    #: Whether a native streaming config is expressed by *excluding* the leaves
    #: it must not touch, which needs the model's structure to enumerate. A
    #: converter that takes the targets positively needs no structure at all.
    streams_by_exclusion = True
    supports_text_encoder_post_load = True
    #: Why this implementation's weights do not survive the host round trip a
    #: group-offload hook performs, keyed by whether the hook pins each tensor
    #: first. None means they do, or that nothing has measured otherwise --
    #: refusing on a guess would assert a claim no one has tested.
    group_offload_refusal = None
    #: Filled in by ``stores`` at registration.
    format_name = ""
    impl = ""
    #: What to say when this implementation cannot stream per weight and the
    #: probe recorded no reason of its own. Each implementation names the API
    #: it would have streamed through, so the log says which one was missing.
    default_unavailable_reason = "native per-weight streaming is unavailable"

    def __init__(
        self,
        *,
        backend,
        format_,
        native_transformer_streaming: bool = False,
        native_unavailable_reason: str | None = None,
        native_text_encoder_streaming: bool = False,
        text_encoder_unavailable_reason: str | None = None,
    ) -> None:
        self.backend = backend
        self.format = format_
        self.uses_native_transformer_streaming = native_transformer_streaming
        self.native_unavailable_reason = native_unavailable_reason
        self.uses_native_text_encoder_streaming = native_text_encoder_streaming
        self.text_encoder_unavailable_reason = text_encoder_unavailable_reason

    def _stream_config_factory(self, exclusions):
        raise TargetMappingUnavailable(
            f"{type(self).__name__} cannot build a native streaming config"
        )

    def transformer_stream_plan(
        self,
        targets,
        *,
        model_factory=None,
        residual_match: Callable[[str], bool] | None = None,
    ):
        """The native config for this component, and which leaves it owns."""

        if not self.uses_native_transformer_streaming:
            raise TargetMappingUnavailable(
                self.native_unavailable_reason or self.default_unavailable_reason
            )
        if model_factory is None:
            raise TargetMappingUnavailable(
                "target mapping unavailable: no model structure factory"
            )
        try:
            model = model_factory()
        except Exception as exc:
            raise TargetMappingUnavailable(
                f"target mapping unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        ownership = derive_linear_ownership(
            model,
            targets,
            min_layer_size=self.min_layer_size,
            residual_match=residual_match,
        )
        try:
            config = self._stream_config_factory(ownership.exclusions)
        except TargetMappingUnavailable:
            raise
        except Exception as exc:
            raise TargetMappingUnavailable(
                f"{self.default_unavailable_reason}: {type(exc).__name__}: {exc}"
            ) from exc
        return config, ownership

    def transformer_stream_config(self, targets, *, model_factory=None):
        config, _ = self.transformer_stream_plan(targets, model_factory=model_factory)
        return config

    def _single_layer_factory(self, *, device):
        """Build one layer at a time, which a per-step pair is made of.

        Overridden by every implementation with a single-leaf seam. One that
        has none is neither half of a pair, and says so.
        """

        from .contracts import UnsupportedLoadContract

        raise UnsupportedLoadContract(
            f"{self.impl} {self.format_name} cannot install one layer at a "
            "time, so it cannot be either half of a low/high GEMM schedule"
        )

    def layer_factory(self, *, device, companion=None):
        """One layer, or a per-step pair when the run supplies a companion.

        The pairing is done here rather than in any format's own factory, so
        no implementation has to know it can be half of a pair, and none is
        excluded by having been written before the schedule existed.
        """

        from xfuser.core.utils.runner_utils import paired_layer_factory

        return paired_layer_factory(
            self._single_layer_factory(device=device), companion
        )

    @classmethod
    def builds_one_layer(cls) -> bool:
        """Whether this implementation has a single-leaf seam at all."""

        return cls._single_layer_factory is not QuantAdapter._single_layer_factory

    def convert_module(self, module, *, device, filter_fn=None, offload_to_cpu=False):
        raise NotImplementedError

    def convert_block(self, block, *, device, **kwargs):
        return self.convert_module(block, device=device, **kwargs)


def descriptor_for(adapter, component_name, mode, fallback=None) -> QuantLoadDescriptor:
    return QuantLoadDescriptor(
        requested_format=adapter.format.value,
        selected_backend=adapter.backend.value,
        storage_semantics=adapter.storage_semantics,
        materialization_mode=mode,
        parameter_semantics=adapter.parameter_semantics,
        auxiliary_state_semantics=adapter.auxiliary_state_semantics,
        trainability=adapter.trainability,
        serialization=adapter.serialization,
        fallback_reason=fallback,
        component_name=component_name,
    )


def prepare_native_load(
    adapter,
    *,
    component_name,
    targets,
    stream_quant,
    model_factory=None,
    residual_match: Callable[[str], bool] | None = None,
    hybrid: bool = False,
) -> PreparedQuantLoad:
    """Quantize on the way in from disk, or say why this load cannot.

    This is where ``materialization_mode`` is decided: streaming when the run
    asked for it, the implementation can express these targets, and the targets
    are non-empty; post-load otherwise, with the reason recorded.
    """

    targets = tuple(targets)
    if not stream_quant:
        fallback = "streaming disabled by the runner"
    elif not targets:
        fallback = f"{component_name} has no {adapter.format.value.upper()} targets"
    elif hybrid and not adapter.streams_under_hybrid:
        fallback = (
            f"native {adapter.format.value.upper()} streaming cannot preserve "
            "hybrid ownership"
        )
    else:
        try:
            config, ownership = adapter.transformer_stream_plan(
                targets,
                model_factory=model_factory,
                # Only a converter that can hold leaves back at another format
                # has a residual to keep out of the stream.
                residual_match=(
                    residual_match if adapter.supports_precision_overrides else None
                ),
            )
        except Exception as exc:
            fallback = str(exc)
        else:
            return PreparedQuantLoad(
                descriptor=descriptor_for(adapter, component_name, "streaming"),
                quantization_config=config,
                streamed_targets=(
                    ownership.streamed if ownership.residual else targets
                ),
                residual_targets=ownership.residual,
            )
    return PreparedQuantLoad(
        descriptor=descriptor_for(adapter, component_name, "post_load", fallback)
    )


def describe_blockwise_load(
    adapter,
    *,
    component_name,
    targets,
    wrap_attrs,
) -> QuantLoadDescriptor:
    """How a per-block fill will quantize one component, or why it cannot."""

    targets = tuple(targets)
    wrap_attrs = tuple(wrap_attrs)
    if not targets:
        return descriptor_for(
            adapter,
            component_name,
            "post_load",
            f"{component_name} has no {adapter.format.value.upper()} targets",
        )
    if not wrap_attrs or not any(
        module_paths_overlap(target, attr) for target in targets for attr in wrap_attrs
    ):
        return descriptor_for(
            adapter,
            component_name,
            "post_load",
            "quantization targets do not align with streamed transformer blocks",
        )
    return descriptor_for(adapter, component_name, "blockwise")
