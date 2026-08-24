"""Dependency-light FP8 backend identities, capability checks, and load planning."""

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from packaging.version import InvalidVersion, Version
from types import SimpleNamespace
from typing import Callable

_MIN_TORCHAO_VERSION = Version("0.15.0")
_ProbeResult = bool | tuple[bool, str | None]


@dataclass(frozen=True)
class Fp8BackendCapabilities:
    """Runtime support discovered before transformer allocation."""

    aiter_block_scale: bool = False
    aiter_transformers_streaming: bool = False
    torchao_fp8: bool = False
    torchao_diffusers_streaming: bool = False
    torchao_text_encoder_streaming: bool = False
    torchao_fsdp_patches: bool = False
    aiter_transformers_reason: str | None = None
    torchao_fp8_reason: str | None = None
    torchao_diffusers_reason: str | None = None
    torchao_text_encoder_reason: str | None = None
    torchao_fsdp_reason: str | None = None


def _probe_torchao_fp8_conversion_api() -> tuple[bool, str | None]:
    """Import and instantiate the exact TorchAO APIs used by conversion."""

    try:
        installed = Version(version("torchao"))
    except PackageNotFoundError:
        return False, "torchao is not installed"
    except InvalidVersion as exc:
        return False, f"cannot parse installed torchao version: {exc}"
    if installed < _MIN_TORCHAO_VERSION:
        return (
            False,
            f"torchao {installed} is older than required " f"{_MIN_TORCHAO_VERSION}",
        )

    try:
        torchao_quant = import_module("torchao.quantization.quant_api")
        granularity = import_module("torchao.quantization.granularity")
        common = import_module("torchao.quantization.quantize_.common")
        ao_config_cls = getattr(
            torchao_quant, "Float8DynamicActivationFloat8WeightConfig"
        )
        quantize = getattr(torchao_quant, "quantize_")
        is_linear = getattr(torchao_quant, "_is_linear")
        per_tensor_cls = getattr(granularity, "PerTensor")
        kernel_preference = getattr(common, "KernelPreference")
        config = ao_config_cls(
            granularity=per_tensor_cls(),
            set_inductor_config=False,
            kernel_preference=kernel_preference.AUTO,
        )
        if not callable(quantize) or not callable(is_linear) or config is None:
            return False, "TorchAO FP8 conversion APIs are not callable"
    except Exception as exc:
        return (
            False,
            f"TorchAO FP8 conversion API probe failed: " f"{type(exc).__name__}: {exc}",
        )
    return True, None


def _probe_torchao_diffusers_streaming() -> tuple[bool, str | None]:
    """Probe the native unquantized-checkpoint streaming API in isolation."""

    conversion_available, conversion_reason = _probe_torchao_fp8_conversion_api()
    if not conversion_available:
        return False, conversion_reason

    try:
        diffusers = import_module("diffusers")
        quantizer_module = import_module(
            "diffusers.quantizers.torchao.torchao_quantizer"
        )
        torchao_quant = import_module("torchao.quantization.quant_api")
        granularity = import_module("torchao.quantization.granularity")
        config_cls = getattr(diffusers, "TorchAoConfig")
        ao_config_cls = getattr(
            torchao_quant, "Float8DynamicActivationFloat8WeightConfig"
        )
        per_tensor_cls = getattr(granularity, "PerTensor")
        quantizer_cls = getattr(quantizer_module, "TorchAoHfQuantizer")
        config = config_cls(
            ao_config_cls(
                granularity=per_tensor_cls(),
                set_inductor_config=False,
            ),
            modules_to_not_convert=[],
        )
        available = bool(
            config is not None
            and callable(getattr(quantizer_cls, "create_quantized_param", None))
            and callable(getattr(quantizer_cls, "check_if_quantized_param", None))
        )
        return (
            available,
            (
                None
                if available
                else "Diffusers TorchAO quantizer methods are unavailable"
            ),
        )
    except Exception as exc:
        return (
            False,
            f"Diffusers TorchAoConfig API probe failed: "
            f"{type(exc).__name__}: {exc}",
        )


def _quantizes_parameter_by_parameter(quantizer_cls) -> bool:
    """Whether this quantizer can convert one parameter at a time during a load.

    Two surfaces express that, and which one a library exposes is not something to
    infer from its version. Transformers 5 replaced the pair Transformers 4 had
    (create_quantized_param with check_if_quantized_param) with an op-based pair
    (get_quantize_ops with param_needs_quantization), while Diffusers still carries
    the older one. Requiring only the older pair meant no installed Transformers
    matched, so every text encoder quietly took the post-load fallback and the
    streaming path this asks about was never entered.
    """

    def has(*names: str) -> bool:
        return all(callable(getattr(quantizer_cls, name, None)) for name in names)

    return has("get_quantize_ops", "param_needs_quantization") or has(
        "create_quantized_param", "check_if_quantized_param"
    )


def _probe_torchao_text_encoder_streaming() -> tuple[bool, str | None]:
    """Probe granular Diffusers routing to Transformers TorchAO loading."""

    conversion_available, conversion_reason = _probe_torchao_fp8_conversion_api()
    if not conversion_available:
        return False, conversion_reason
    try:
        diffusers_quantizers = import_module("diffusers.quantizers")
        transformers = import_module("transformers")
        quantizer_module = import_module("transformers.quantizers.quantizer_torchao")
        from xfuser.model_executor.models.runner_models.loading.text_encoder_adapter import (
            TextEncoderFrameworkAdapter,
        )

        quantizer_cls = getattr(quantizer_module, "TorchAoHfQuantizer")
        pipeline_cls = getattr(diffusers_quantizers, "PipelineQuantizationConfig")
        config = TextEncoderFrameworkAdapter().component_quantization_config(
            backend="torchao",
            targets=("probe",),
            exclusions=("probe",),
        )
        pipeline_config = pipeline_cls(quant_mapping={"text_encoder": config})
        available = bool(
            getattr(transformers, "TorchAoConfig", None)
            and config is not None
            and pipeline_config is not None
            and _quantizes_parameter_by_parameter(quantizer_cls)
        )
        return (
            available,
            (
                None
                if available
                else "Transformers TorchAO quantize-on-load methods are unavailable"
            ),
        )
    except Exception as exc:
        return (
            False,
            "TorchAO text-encoder framework API probe failed: "
            f"{type(exc).__name__}: {exc}",
        )


def _probe_aiter_transformers_streaming() -> tuple[bool, str | None]:
    from .text_encoder_adapter import probe_transformers_streaming_loader

    support = probe_transformers_streaming_loader()
    return support.available, support.reason


def _probe_torchao_fsdp_patches() -> tuple[bool, str | None]:
    """Validate xDiT's required TorchAO Float8Tensor FSDP2 patches."""

    try:
        from xfuser.core.utils.runner_utils import (
            torchao_float8_fsdp2_patches_available,
        )

        return torchao_float8_fsdp2_patches_available()
    except Exception as exc:
        return (
            False,
            f"TorchAO FSDP patch validation failed: " f"{type(exc).__name__}: {exc}",
        )


def _probe_result(result) -> tuple[bool, str | None]:
    if isinstance(result, tuple):
        available, reason = result
        return bool(available), reason
    return bool(result), None


def _probe_torchao_fp8_accelerator(
    *,
    cuda_probe: Callable[[], bool] | None = None,
    hip_probe: Callable[[], bool] | None = None,
    cuda_capability_probe: Callable[[], tuple[int, int] | None] | None = None,
) -> tuple[bool, str | None]:
    """Validate the accelerator required by TorchAO's FP8 kernels."""

    if cuda_probe is None or hip_probe is None:
        from xfuser.envs import _is_cuda, _is_hip

        cuda_probe = cuda_probe or _is_cuda
        hip_probe = hip_probe or _is_hip
    if hip_probe():
        return True, None
    if not cuda_probe():
        return False, "TorchAO FP8 requires CUDA or HIP/ROCm"
    if cuda_capability_probe is None:
        torch = import_module("torch")
        cuda_capability_probe = torch.cuda.get_device_capability
    try:
        capability = cuda_capability_probe()
    except Exception as exc:
        return (
            False,
            f"cannot query CUDA capability for TorchAO FP8: "
            f"{type(exc).__name__}: {exc}",
        )
    if capability is None or capability < (8, 9):
        observed = (
            "unknown" if capability is None else f"{capability[0]}.{capability[1]}"
        )
        return (
            False,
            f"TorchAO FP8 requires CUDA capability >= 8.9; observed {observed}",
        )
    return True, None


def probe_fp8_backend_capabilities(
    *,
    aiter_probe: Callable[[], bool] | None = None,
    torchao_accelerator_probe: Callable[[], _ProbeResult] | None = None,
    torchao_probe: Callable[[], _ProbeResult] | None = None,
    torchao_diffusers_probe: Callable[[], _ProbeResult] | None = None,
    torchao_text_encoder_probe: Callable[[], _ProbeResult] | None = None,
    aiter_transformers_probe: Callable[[], _ProbeResult] | None = None,
    torchao_fsdp_probe: Callable[[], _ProbeResult] | None = None,
) -> Fp8BackendCapabilities:
    """Keep hardware/package probing outside adapters and injectable in tests."""

    if aiter_probe is None:
        from xfuser.core.utils.runner_utils import _use_aiter_fp8_rdna4

        aiter_probe = _use_aiter_fp8_rdna4
    if torchao_accelerator_probe is None:
        torchao_accelerator_probe = _probe_torchao_fp8_accelerator
    if torchao_probe is None:
        torchao_probe = _probe_torchao_fp8_conversion_api
    if torchao_diffusers_probe is None:
        torchao_diffusers_probe = _probe_torchao_diffusers_streaming
    if torchao_text_encoder_probe is None:
        torchao_text_encoder_probe = _probe_torchao_text_encoder_streaming
    if aiter_transformers_probe is None:
        aiter_transformers_probe = _probe_aiter_transformers_streaming
    if torchao_fsdp_probe is None:
        torchao_fsdp_probe = _probe_torchao_fsdp_patches
    aiter_available = bool(aiter_probe())
    if aiter_available:
        aiter_te_available, aiter_te_reason = _probe_result(aiter_transformers_probe())
    else:
        aiter_te_available, aiter_te_reason = (
            False,
            "AITER FP8 backend is unavailable",
        )
    accelerator_available, accelerator_reason = _probe_result(
        torchao_accelerator_probe()
    )
    if accelerator_available:
        torchao_available, torchao_reason = _probe_result(torchao_probe())
    else:
        torchao_available = False
        torchao_reason = accelerator_reason or "TorchAO FP8 requires CUDA or HIP/ROCm"
    if torchao_available:
        native_available, native_reason = _probe_result(torchao_diffusers_probe())
        te_available, te_reason = _probe_result(torchao_text_encoder_probe())
        fsdp_available, fsdp_reason = _probe_result(torchao_fsdp_probe())
    else:
        native_available, native_reason = False, torchao_reason
        te_available, te_reason = False, torchao_reason
        fsdp_available, fsdp_reason = False, torchao_reason
    return Fp8BackendCapabilities(
        aiter_block_scale=aiter_available,
        aiter_transformers_streaming=aiter_te_available,
        torchao_fp8=torchao_available,
        torchao_diffusers_streaming=native_available,
        torchao_text_encoder_streaming=te_available,
        torchao_fsdp_patches=fsdp_available,
        aiter_transformers_reason=aiter_te_reason,
        torchao_fp8_reason=torchao_reason,
        torchao_diffusers_reason=native_reason,
        torchao_text_encoder_reason=te_reason,
        torchao_fsdp_reason=fsdp_reason,
    )


@dataclass(frozen=True)
class TransformerFp8LoadDescriptor:
    requested_format: str
    selected_backend: str
    storage_semantics: str
    materialization_mode: str
    fallback_reason: str | None = None
    component_name: str = "transformer"

    def log_message(self) -> str:
        message = (
            f"{self.component_name} quantization: "
            f"requested={self.requested_format}, "
            f"backend={self.selected_backend}, "
            f"storage={self.storage_semantics}, "
            f"materialization={self.materialization_mode}"
        )
        if self.fallback_reason:
            message += f"; fallback={self.fallback_reason}"
        return message


@dataclass(frozen=True)
class PreparedTransformerFp8Load:
    descriptor: TransformerFp8LoadDescriptor
    quantization_config: object | None = None


class TargetMappingUnavailable(RuntimeError):
    """The config-built model cannot safely express xDiT target prefixes."""


def _is_linear_module(module) -> bool:
    from torch import nn

    return isinstance(module, nn.Linear)


def derive_untargeted_linear_exclusions(
    model,
    targets,
    *,
    is_linear: Callable[[object], bool] = _is_linear_module,
) -> list[str]:
    """Map positive xDiT prefixes to Diffusers' negative Linear list."""

    targets = tuple(dict.fromkeys(targets))
    for target in targets:
        try:
            model.get_submodule(target)
        except (AttributeError, KeyError) as exc:
            raise TargetMappingUnavailable(
                f"target mapping unavailable: model structure is missing '{target}'"
            ) from exc

    def targeted(name: str) -> bool:
        return any(
            name == target or name.startswith(f"{target}.") for target in targets
        )

    return [
        name
        for name, module in model.named_modules()
        if name and is_linear(module) and not targeted(name)
    ]


class Fp8BackendAdapter:
    """Shared conversion/config surface for ordinary, replicated, and FSDP loads."""

    storage_semantics = ""
    converts_before_device_move = False
    supports_text_encoder_post_load = True

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

    def transformer_stream_config(self, targets, *, model_factory=None):
        return None

    def convert_module(self, module, *, device, offload_to_cpu=False):
        raise NotImplementedError

    def convert_block(self, block, *, device, **kwargs):
        return self.convert_module(block, device=device, **kwargs)


class AiterFp8BackendAdapter(Fp8BackendAdapter):
    storage_semantics = "block_128_scaled"
    converts_before_device_move = True

    def _stream_config_factory(self, targets):
        from xfuser.model_executor.quant.aiter_load import stream_config

        return stream_config(list(targets))

    def transformer_stream_config(self, targets, *, model_factory=None):
        return self._stream_config_factory(targets)

    def convert_module(
        self,
        module,
        *,
        device,
        offload_to_cpu=False,
        filter_fn=None,
    ):
        from xfuser.core.utils.runner_utils import (
            quantize_linear_layers_to_fp8_blockscale,
        )

        return quantize_linear_layers_to_fp8_blockscale(
            module,
            device=device,
            offload_to_cpu=offload_to_cpu,
            filter_fn=filter_fn,
        )


class TorchaoFp8BackendAdapter(Fp8BackendAdapter):
    storage_semantics = "tensorwise_dynamic"

    def _stream_config_factory(self, exclusions):
        from diffusers import TorchAoConfig
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
        )
        from xfuser.core.utils.runner_utils import (
            FP8_ACTIVATION_SCALE_FLOOR,
            _get_fp8_kernel_preference,
        )
        from xfuser.model_executor.quant.torchao_quantizer import (
            register_torchao_fp32_policy,
        )

        register_torchao_fp32_policy()
        quant_type = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=False,
            kernel_preference=_get_fp8_kernel_preference(),
            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        )
        return TorchAoConfig(
            quant_type,
            modules_to_not_convert=list(exclusions),
        )

    def transformer_stream_config(self, targets, *, model_factory=None):
        if not self.uses_native_transformer_streaming:
            raise TargetMappingUnavailable(
                self.native_unavailable_reason
                or "Diffusers TorchAoConfig API is unavailable"
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
        exclusions = derive_untargeted_linear_exclusions(model, targets)
        try:
            return self._stream_config_factory(exclusions)
        except Exception as exc:
            raise TargetMappingUnavailable(
                "Diffusers TorchAoConfig API is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def convert_module(
        self,
        module,
        *,
        device,
        offload_to_cpu=False,
        filter_fn=None,
    ):
        if offload_to_cpu:
            raise ValueError(
                "torchao FP8 conversion does not support immediate CPU offload"
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp8

        return quantize_linear_layers_to_fp8(module, device=device, filter_fn=filter_fn)


def _unsupported_error(contract):
    module = import_module(contract.requested_format.__class__.__module__)
    return getattr(module, "UnsupportedLoadContract", ValueError)


def select_fp8_backend(contract, *, capabilities: Fp8BackendCapabilities):
    """Select exactly the backend the contract names, or fail explicitly."""

    format_value = contract.requested_format.value
    backend_value = contract.selected_backend.value
    if format_value != "fp8":
        return None

    error = _unsupported_error(contract)
    if backend_value == "aiter":
        if not capabilities.aiter_block_scale:
            raise error("AITER backend for FP8 is unavailable")
        return AiterFp8BackendAdapter(
            backend=contract.selected_backend,
            format_=contract.requested_format,
            native_transformer_streaming=True,
            native_text_encoder_streaming=(capabilities.aiter_transformers_streaming),
            text_encoder_unavailable_reason=(capabilities.aiter_transformers_reason),
        )
    if backend_value == "torchao":
        if not capabilities.torchao_fp8:
            reason = (
                f": {capabilities.torchao_fp8_reason}"
                if capabilities.torchao_fp8_reason
                else ""
            )
            raise error(f"TORCHAO backend for FP8 is unavailable{reason}")
        return TorchaoFp8BackendAdapter(
            backend=contract.selected_backend,
            format_=contract.requested_format,
            native_transformer_streaming=(capabilities.torchao_diffusers_streaming),
            native_unavailable_reason=(capabilities.torchao_diffusers_reason),
            native_text_encoder_streaming=(capabilities.torchao_text_encoder_streaming),
            text_encoder_unavailable_reason=(capabilities.torchao_text_encoder_reason),
        )
    raise error(f"{contract.selected_backend.name} backend cannot store requested FP8")


def select_blockwise_fp8_backend(
    contract,
    *,
    capabilities: Fp8BackendCapabilities,
):
    """Select FP8 independently for FP8, FP4, and FP8+FP4 contracts."""

    format_value = contract.requested_format.value
    if format_value not in {"fp8", "fp4", "fp8_fp4"}:
        return None
    if format_value == "fp8":
        fp8_contract = contract
    else:
        backend_enum = contract.selected_backend.__class__
        fp8_backend = (
            backend_enum.AITER
            if capabilities.aiter_block_scale
            else backend_enum.TORCHAO
        )
        fp8_contract = SimpleNamespace(
            requested_format=contract.requested_format.__class__.FP8,
            selected_backend=fp8_backend,
        )
    adapter = select_fp8_backend(fp8_contract, capabilities=capabilities)
    return adapter


def validate_torchao_fsdp2_patches(
    contract,
    *,
    capabilities: Fp8BackendCapabilities,
    required: bool,
) -> None:
    """Reject an FSDP2 placement that will contain TorchAO tensor subclasses."""

    if not required or capabilities.torchao_fsdp_patches:
        return
    reason = (
        f": {capabilities.torchao_fsdp_reason}"
        if capabilities.torchao_fsdp_reason
        else ""
    )
    raise _unsupported_error(contract)(
        "TorchAO tensor subclasses under FSDP2 require "
        f"Float8Tensor FSDP patches{reason}"
    )


def _descriptor(adapter, component_name, streaming, fallback=None):
    return TransformerFp8LoadDescriptor(
        requested_format=adapter.format.value,
        selected_backend=adapter.backend.value,
        storage_semantics=adapter.storage_semantics,
        materialization_mode="streaming" if streaming else "post_load",
        fallback_reason=fallback,
        component_name=component_name,
    )


def prepare_native_transformer_fp8_load(
    adapter,
    *,
    component_name: str,
    targets,
    stream_quant: bool,
    model_factory=None,
) -> PreparedTransformerFp8Load:
    """Build a native Diffusers config, or return an explicit fallback."""

    targets = tuple(targets)
    if not stream_quant:
        fallback = "streaming disabled by the runner"
    elif not targets:
        fallback = f"{component_name} has no FP8 targets"
    else:
        try:
            config = adapter.transformer_stream_config(
                targets,
                model_factory=model_factory,
            )
        except TargetMappingUnavailable as exc:
            fallback = str(exc)
        else:
            return PreparedTransformerFp8Load(
                descriptor=_descriptor(adapter, component_name, streaming=True),
                quantization_config=config,
            )
    return PreparedTransformerFp8Load(
        descriptor=_descriptor(
            adapter, component_name, streaming=False, fallback=fallback
        )
    )


def prepare_text_encoder_fp8_load(
    adapter,
    *,
    component_name: str,
    targets,
    model_factory=None,
    stream_quant: bool = True,
    supports_post_load: bool | None = None,
    framework_config_factory=None,
) -> PreparedTransformerFp8Load:
    """Plan one TE load, keeping framework construction behind its adapter."""

    targets = tuple(targets)
    if not targets:
        fallback = f"{component_name} has no FP8 targets"
    elif not stream_quant:
        fallback = "streaming disabled by the runner"
    elif not adapter.uses_native_text_encoder_streaming:
        fallback = (
            adapter.text_encoder_unavailable_reason
            or "text-encoder framework quantize-on-load API is unavailable"
        )
    else:
        try:
            exclusions = ()
            if adapter.backend.value == "torchao":
                if model_factory is None:
                    raise TargetMappingUnavailable(
                        "target mapping unavailable: no text-encoder "
                        "structure factory"
                    )
                try:
                    model = model_factory()
                except Exception as exc:
                    raise TargetMappingUnavailable(
                        "target mapping unavailable: " f"{type(exc).__name__}: {exc}"
                    ) from exc
                exclusions = tuple(derive_untargeted_linear_exclusions(model, targets))
            if framework_config_factory is None:
                from .text_encoder_adapter import (
                    TextEncoderFrameworkAdapter,
                )

                framework = TextEncoderFrameworkAdapter()
                framework_config_factory = lambda backend, targets, exclusions: (
                    framework.component_quantization_config(
                        backend=backend,
                        targets=targets,
                        exclusions=exclusions,
                    )
                )
            config = framework_config_factory(
                adapter.backend.value,
                targets,
                exclusions,
            )
        except TargetMappingUnavailable as exc:
            fallback = str(exc)
        except Exception as exc:
            fallback = (
                "text-encoder framework config unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            return PreparedTransformerFp8Load(
                descriptor=_descriptor(adapter, component_name, streaming=True),
                quantization_config=config,
            )

    if supports_post_load is None:
        supports_post_load = adapter.supports_text_encoder_post_load
    if targets and not supports_post_load:
        raise RuntimeError(
            f"{component_name} FP8 cannot fall back before allocation: " f"{fallback}"
        )
    return PreparedTransformerFp8Load(
        descriptor=_descriptor(
            adapter,
            component_name,
            streaming=False,
            fallback=fallback,
        )
    )


def plan_blockwise_transformer_fp8_load(
    adapter,
    *,
    component_name: str,
    targets,
    wrap_attrs,
) -> TransformerFp8LoadDescriptor:
    """Describe the existing FSDP/replicated per-block materializer."""

    targets = tuple(targets)
    wrap_attrs = tuple(wrap_attrs)
    if not targets:
        fallback = f"{component_name} has no FP8 targets"
    elif not wrap_attrs:
        fallback = "model has no streamed transformer blocks"
    elif not any(
        target == attr or target.startswith(f"{attr}.")
        for target in targets
        for attr in wrap_attrs
    ):
        fallback = "FP8 targets do not align with streamed transformer blocks"
    else:
        return _descriptor(adapter, component_name, streaming=True)
    return _descriptor(adapter, component_name, streaming=False, fallback=fallback)
