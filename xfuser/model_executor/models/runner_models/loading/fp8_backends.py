"""Dependency-light FP8 backend identities, capability checks, and load planning."""

from dataclasses import dataclass
from importlib import import_module

from .quant_adapter import (
    Capabilities,
    FormatCapability,
    LinearOwnership,
    PreparedQuantLoad,
    QuantAdapter,
    stores,
    TargetMappingUnavailable,
    derive_linear_ownership,
    descriptor_for,
    prepare_native_load,
)
from importlib.metadata import PackageNotFoundError, version
from packaging.version import InvalidVersion, Version
from types import SimpleNamespace
from typing import Callable

_MIN_TORCHAO_VERSION = Version("0.15.0")
_ProbeResult = bool | tuple[bool, str | None]


#: The two FP8 implementations, spelled as QuantizationBackend values so a
#: caller maps the answer onto its own enum without a second table.
AITER_FP8 = "aiter"
TORCHAO_FP8 = "torchao"


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
) -> Capabilities:
    """Keep hardware/package probing outside adapters and injectable in tests.

    AITER's ``gemm_a8w8_blockscale`` is a Triton kernel confirmed to reach FP8
    WMMA on gfx1200+ (RDNA4), and ``aiter_probe`` is gated to exactly that; on
    every other target -- CUDA, MI300, gfx950 -- it reports unavailable and the
    preference order falls through to torchao per-tensor dynamic scaling.
    """

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
    return Capabilities(
        {
            ("fp8", AITER_FP8): FormatCapability(
                available=aiter_available,
                reason=None if aiter_available else "AITER FP8 requires RDNA4",
                streams=aiter_available,
                te_streams=aiter_te_available,
                te_streams_reason=aiter_te_reason,
                # Block-128 scales live in a plain nn.Parameter and a buffer,
                # which is why this one needs no Float8Tensor FSDP patches.
                fsdp_safe=True,
            ),
            ("fp8", TORCHAO_FP8): FormatCapability(
                available=torchao_available,
                reason=torchao_reason,
                streams=native_available,
                streams_reason=native_reason,
                te_streams=te_available,
                te_streams_reason=te_reason,
                fsdp_safe=fsdp_available,
                fsdp_reason=fsdp_reason,
            ),
        }
    )


def _is_linear_module(module) -> bool:
    from torch import nn

    return isinstance(module, nn.Linear)


@stores("fp8", AITER_FP8)
class AiterFp8BackendAdapter(QuantAdapter):
    storage_semantics = "block_128_scaled"
    # A plain nn.Parameter, which is why this one needs no FSDP2 patches.
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "persistent_scale_buffer"
    serialization = "packed_state_supported_not_portable"
    converts_before_device_move = True
    loads_to_host_under_offload = True
    meta_layout_matches_storage = True
    streams_by_exclusion = False

    def _stream_config_factory(self, targets):
        from xfuser.model_executor.quant.aiter_load import stream_config

        return stream_config(list(targets))

    def transformer_stream_plan(
        self, targets, *, model_factory=None, residual_match=None
    ):
        """Built from the targets themselves; this one needs no model structure."""
        targets = tuple(targets)
        return self._stream_config_factory(targets), LinearOwnership(
            exclusions=(), streamed=targets
        )

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import packed_layer_factory
        from xfuser.model_executor.layers.fp8_linear import xFuserFP8BlockScaleLinear

        return packed_layer_factory(xFuserFP8BlockScaleLinear, device)

    def convert_module(
        self,
        module,
        *,
        device,
        offload_to_cpu=False,
        filter_fn=None,
        companion=None,
    ):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("fp8", TORCHAO_FP8)
class TorchaoFp8BackendAdapter(QuantAdapter):
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    storage_semantics = "tensorwise_dynamic"
    parameter_semantics = "torchao_float8_tensor_subclass"

    def _quant_config(self):
        """The torchao config this implementation quantizes with.

        One definition, whether it is applied to a whole tree on the way in
        from disk or to a single leaf as a hybrid companion.
        """
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
        )
        from xfuser.core.utils.runner_utils import (
            FP8_ACTIVATION_SCALE_FLOOR,
            _get_fp8_kernel_preference,
        )

        return Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=False,
            kernel_preference=_get_fp8_kernel_preference(),
            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        )

    def _stream_config_factory(self, exclusions):
        from diffusers import TorchAoConfig
        from xfuser.model_executor.quant.torchao_quantizer import (
            register_torchao_fp32_policy,
        )

        register_torchao_fp32_policy()
        return TorchAoConfig(
            self._quant_config(),
            modules_to_not_convert=list(exclusions),
        )

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(self._quant_config(), device)

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


def prepare_text_encoder_fp8_load(
    adapter,
    *,
    component_name: str,
    targets,
    model_factory=None,
    stream_quant: bool = True,
    supports_post_load: bool | None = None,
    framework_config_factory=None,
) -> PreparedQuantLoad:
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
            if adapter.streams_by_exclusion:
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
                exclusions = derive_linear_ownership(model, targets).exclusions
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
            return PreparedQuantLoad(
                descriptor=descriptor_for(adapter, component_name, "streaming"),
                quantization_config=config,
            )

    if supports_post_load is None:
        supports_post_load = adapter.supports_text_encoder_post_load
    if targets and not supports_post_load:
        raise RuntimeError(
            f"{component_name} FP8 cannot fall back before allocation: " f"{fallback}"
        )
    return PreparedQuantLoad(
        descriptor=descriptor_for(
            adapter, component_name, "post_load", fallback
        )
    )
