"""FP4/INT8 backend adapters and dependency-light materialization planning."""

import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from packaging.version import InvalidVersion, Version
from typing import Callable

_ProbeResult = bool | tuple[bool, str | None]
_MIN_TORCHAO_VERSION = Version("0.15.0")
MXFP4_STREAMING_FALLBACK = (
    "AITER MXFP4 conversion requires each full-precision weight before creating "
    "xFuserMXFP4Linear packed state; no safe Diffusers per-weight loader exists"
)
# The architectures AITER has FP4 kernels for, per its own arch_info.is_fp4_avail. Its build gate
# is wider than that (aiter/jit/core.py compiles -D__Float4_e2m1fn_x2 for anything but gfx942 when
# AITER_FP4x2 is enabled), but the kernels behind the define are narrower: the hand-written A4W4
# assembly covers gfx942 and gfx950, gemm_a4w4 then raises on gfx942, and only gfx950 carries tuned
# configs. RDNA4 has FP8 kernels and no FP4 ones. An arch off this list reaches an AITER_CHECK(false)
# that aborts the process, so it has to be refused here rather than caught at the call site.
_AITER_FP4_ARCHS = ("gfx950", "gfx1250")


def _result(value: _ProbeResult) -> tuple[bool, str | None]:
    if isinstance(value, tuple):
        return bool(value[0]), value[1]
    return bool(value), None


@dataclass(frozen=True)
class FormatBackendCapabilities:
    torchao_nvfp4: bool = False
    aiter_mxfp4: bool = False
    torchao_int8: bool = False
    torchao_nvfp4_streaming: bool = False
    torchao_int8_streaming: bool = False
    torchao_nvfp4_fsdp: bool = False
    torchao_int8_fsdp: bool = False
    aiter_mxfp4_fsdp: bool = False
    torchao_nvfp4_reason: str | None = None
    aiter_mxfp4_reason: str | None = None
    torchao_int8_reason: str | None = None
    torchao_nvfp4_streaming_reason: str | None = None
    torchao_int8_streaming_reason: str | None = None
    torchao_nvfp4_fsdp_reason: str | None = None
    torchao_int8_fsdp_reason: str | None = None
    aiter_mxfp4_fsdp_reason: str | None = None


def _probe_torchao_config(kind: str) -> tuple[bool, str | None]:
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
        if kind == "nvfp4":
            module = import_module("torchao.prototype.mx_formats.inference_workflow")
            config = module.NVFP4DynamicActivationNVFP4WeightConfig(
                use_dynamic_per_tensor_scale=True,
                use_triton_kernel=True,
            )
        else:
            quant = import_module("torchao.quantization.quant_api")
            granularity = import_module("torchao.quantization.granularity")
            primitives = import_module("torchao.quantization.quant_primitives")
            config = quant.Int8DynamicActivationInt8WeightConfig(
                granularity=granularity.PerRow(),
                act_mapping_type=primitives.MappingType.SYMMETRIC,
                set_inductor_config=False,
            )
        if config is None:
            return False, f"TorchAO {kind.upper()} config is unavailable"
    except Exception as exc:
        return (
            False,
            f"TorchAO {kind.upper()} API probe failed: {type(exc).__name__}: {exc}",
        )
    return True, None


def _probe_diffusers_config(kind: str) -> tuple[bool, str | None]:
    available, reason = _probe_torchao_config(kind)
    if not available:
        return False, reason
    try:
        diffusers = import_module("diffusers")
        quantizer = import_module("diffusers.quantizers.torchao.torchao_quantizer")
        config = _torchao_stream_config(kind, [])
        quantizer_cls = quantizer.TorchAoHfQuantizer
        valid = bool(
            config is not None
            and callable(getattr(quantizer_cls, "check_if_quantized_param", None))
            and callable(getattr(quantizer_cls, "create_quantized_param", None))
            and getattr(diffusers, "TorchAoConfig", None)
        )
        if not valid:
            return False, "Diffusers TorchAoConfig per-weight API is unavailable"
    except Exception as exc:
        return (
            False,
            f"Diffusers TorchAoConfig probe failed: {type(exc).__name__}: {exc}",
        )
    return True, None


def _gcn_arch_name() -> str | None:
    try:
        torch = import_module("torch")
        device = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device).gcnArchName
    except Exception:
        return None


def _probe_aiter_fp4_kernels(
    gcn_arch_probe: Callable[[], str | None] | None = None,
) -> tuple[bool, str | None]:
    """Accept only architectures AITER has FP4 kernels for."""

    if int(os.getenv("AITER_FP4x2", "1")) <= 0:
        return False, "AITER FP4 kernels are disabled by AITER_FP4x2=0"
    arch = (gcn_arch_probe or _gcn_arch_name)()
    if arch is None:
        return False, "cannot determine the ROCm architecture for AITER FP4 support"
    if not any(name in arch for name in _AITER_FP4_ARCHS):
        return (
            False,
            f"AITER builds no FP4 (Float4_e2m1fn_x2) kernels for {arch}; "
            f"FP4 on ROCm requires {' or '.join(_AITER_FP4_ARCHS)}",
        )
    return True, None


def _probe_aiter_mxfp4_apis() -> tuple[bool, str | None]:
    """Probe only the AITER symbols used by xFuserMXFP4Linear."""

    try:
        aiter = import_module("aiter")
        shuffle = import_module("aiter.ops.shuffle")
    except Exception as exc:
        return (
            False,
            f"AITER MXFP4 import probe failed: {type(exc).__name__}: {exc}",
        )

    required_callables = (
        ("aiter.get_hip_quant", getattr(aiter, "get_hip_quant", None)),
        ("aiter.gemm_a4w4", getattr(aiter, "gemm_a4w4", None)),
        (
            "aiter.ops.shuffle.shuffle_weight",
            getattr(shuffle, "shuffle_weight", None),
        ),
    )
    for name, value in required_callables:
        if not callable(value):
            return False, f"missing required AITER MXFP4 API: {name}"
    quant_type = getattr(aiter, "QuantType", None)
    if quant_type is None or not hasattr(quant_type, "per_1x32"):
        return (
            False,
            "missing required AITER MXFP4 API: aiter.QuantType.per_1x32",
        )
    # The symbols above exist on every ROCm arch; only the kernels behind them
    # are arch-gated, so the device check has to happen too.
    return _probe_aiter_fp4_kernels()


def _probe_fsdp_non_float_parameters() -> tuple[bool, str | None]:
    """Require an FSDP2 that can wrap the uint8 packed MXFP4 weight.

    Before pytorch/pytorch#177948 (torch 2.12) FSDP2 built the sharded parameter
    as ``nn.Parameter(dtensor)``, which defaults to ``requires_grad=True`` and
    therefore raises for any integer dtype, before setting the real flag on the
    next line. Detect the fix at its call site so a backport is honoured.
    """

    unsupported = (
        "this PyTorch cannot shard non-floating-point parameters under FSDP2 "
        "(needs the pytorch/pytorch#177948 fix, released in 2.12.0)"
    )
    try:
        inspect = import_module("inspect")
        param_module = import_module("torch.distributed.fsdp._fully_shard._fsdp_param")
        source = inspect.getsource(param_module.FSDPParam._init_sharded_param)
    except Exception:
        try:
            torch = import_module("torch")
            fixed = Version(torch.__version__.split("+")[0]) >= Version("2.12.0")
        except Exception as exc:
            return False, f"FSDP2 non-float parameter probe failed: {exc}"
        return (True, None) if fixed else (False, unsupported)
    if "requires_grad=" not in source:
        return False, unsupported
    return True, None


def _probe_fsdp_support(kind: str) -> tuple[bool, str | None]:
    """Require the exact tensor subclass to expose composable-FSDP gather hooks."""

    if kind == "nvfp4":
        return (
            False,
            "NVFP4 tensor-subclass FSDP gather/scatter support is not validated",
        )
    if kind == "mxfp4":
        return _probe_fsdp_non_float_parameters()
    try:
        torch = import_module("torch")
        quant = import_module("torchao.quantization.quant_api")
        granularity = import_module("torchao.quantization.granularity")
        primitives = import_module("torchao.quantization.quant_primitives")
        module = torch.nn.Linear(512, 512)
        quant.quantize_(
            module,
            quant.Int8DynamicActivationInt8WeightConfig(
                granularity=granularity.PerRow(),
                act_mapping_type=primitives.MappingType.SYMMETRIC,
                set_inductor_config=False,
            ),
        )
        weight = module.weight
        methods = ("fsdp_pre_all_gather", "fsdp_post_all_gather")
        missing = [
            name for name in methods if not callable(getattr(weight, name, None))
        ]
        if missing:
            return False, "INT8 tensor subclass is missing " + ", ".join(missing)
    except Exception as exc:
        return False, f"INT8 FSDP probe failed: {type(exc).__name__}: {exc}"
    return True, None


def probe_format_backend_capabilities(
    *,
    cuda_probe: Callable[[], bool] | None = None,
    hip_probe: Callable[[], bool] | None = None,
    cuda_capability_probe: Callable[[], tuple[int, int] | None] | None = None,
    aiter_probe: Callable[[], bool] | None = None,
    mxfp4_probe: Callable[[], _ProbeResult] | None = None,
    nvfp4_probe: Callable[[], _ProbeResult] | None = None,
    int8_probe: Callable[[], _ProbeResult] | None = None,
    diffusers_probe: Callable[[str], _ProbeResult] | None = None,
    fsdp_probe: Callable[[str], _ProbeResult] | None = None,
) -> FormatBackendCapabilities:
    """Probe packages/hardware with injectable seams for routing tests."""

    if cuda_probe is None or hip_probe is None:
        from xfuser.envs import _is_cuda, _is_hip

        cuda_probe = cuda_probe or _is_cuda
        hip_probe = hip_probe or _is_hip
    cuda = bool(cuda_probe())
    hip = bool(hip_probe())
    if cuda_capability_probe is None:
        if cuda:
            torch = import_module("torch")
            cuda_capability_probe = torch.cuda.get_device_capability
        else:

            def cuda_capability_probe():
                return None

    capability = cuda_capability_probe()
    blackwell = bool(cuda and capability is not None and capability >= (10, 0))
    if mxfp4_probe is None:
        if aiter_probe is None:
            mxfp4_probe = _probe_aiter_mxfp4_apis
        else:

            def mxfp4_probe():
                available = bool(aiter_probe())
                return (
                    available,
                    None if available else "AITER MXFP4 APIs are unavailable",
                )

    nvfp4_probe = nvfp4_probe or (lambda: _probe_torchao_config("nvfp4"))
    int8_probe = int8_probe or (lambda: _probe_torchao_config("int8"))
    diffusers_probe = diffusers_probe or _probe_diffusers_config
    fsdp_probe = fsdp_probe or _probe_fsdp_support

    if blackwell:
        nvfp4, nvfp4_reason = _result(nvfp4_probe())
    else:
        nvfp4 = False
        nvfp4_reason = "NVFP4 requires CUDA capability >= 10.0"
    if hip:
        mxfp4, mxfp4_reason = _result(mxfp4_probe())
    else:
        mxfp4 = False
        mxfp4_reason = "AITER MXFP4 requires ROCm"
    if cuda:
        int8, int8_reason = _result(int8_probe())
    else:
        int8 = False
        int8_reason = "TorchAO INT8 is supported only on CUDA"

    nv_stream, nv_stream_reason = (
        _result(diffusers_probe("nvfp4")) if nvfp4 else (False, nvfp4_reason)
    )
    int8_stream, int8_stream_reason = (
        _result(diffusers_probe("int8")) if int8 else (False, int8_reason)
    )
    nv_fsdp, nv_fsdp_reason = (
        _result(fsdp_probe("nvfp4")) if nvfp4 else (False, nvfp4_reason)
    )
    int8_fsdp, int8_fsdp_reason = (
        _result(fsdp_probe("int8")) if int8 else (False, int8_reason)
    )
    mx_fsdp, mx_fsdp_reason = (
        _result(fsdp_probe("mxfp4")) if mxfp4 else (False, mxfp4_reason)
    )
    return FormatBackendCapabilities(
        torchao_nvfp4=nvfp4,
        aiter_mxfp4=mxfp4,
        torchao_int8=int8,
        torchao_nvfp4_streaming=nv_stream,
        torchao_int8_streaming=int8_stream,
        torchao_nvfp4_fsdp=nv_fsdp,
        torchao_int8_fsdp=int8_fsdp,
        aiter_mxfp4_fsdp=mx_fsdp,
        torchao_nvfp4_reason=nvfp4_reason,
        aiter_mxfp4_reason=mxfp4_reason,
        torchao_int8_reason=int8_reason,
        torchao_nvfp4_streaming_reason=nv_stream_reason,
        torchao_int8_streaming_reason=int8_stream_reason,
        torchao_nvfp4_fsdp_reason=nv_fsdp_reason,
        torchao_int8_fsdp_reason=int8_fsdp_reason,
        aiter_mxfp4_fsdp_reason=mx_fsdp_reason,
    )


@dataclass(frozen=True)
class FormatLoadDescriptor:
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
class PreparedFormatLoad:
    descriptor: FormatLoadDescriptor
    quantization_config: object | None = None
    streamed_targets: tuple[str, ...] = ()
    residual_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinearOwnership:
    exclusions: tuple[str, ...]
    streamed: tuple[str, ...]
    residual: tuple[str, ...] = ()


@dataclass(frozen=True)
class EagerBlockwisePlan:
    enabled: bool
    reason: str | None = None


class FormatBackendAdapter:
    storage_semantics = ""
    parameter_semantics = "tensor_subclass_parameter"
    auxiliary_state_semantics = "backend_managed"
    trainability = "inference_only"
    serialization = "torchao_version_dependent"
    min_layer_size = 0
    supports_precision_overrides = False

    def __init__(
        self,
        *,
        backend,
        format_,
        native_transformer_streaming: bool = False,
        native_unavailable_reason: str | None = None,
    ):
        self.backend = backend
        self.format = format_
        self.uses_native_transformer_streaming = native_transformer_streaming
        self.native_unavailable_reason = native_unavailable_reason

    def transformer_stream_plan(
        self,
        targets,
        *,
        model_factory=None,
        residual_match: Callable[[str], bool] | None = None,
    ):
        if not self.uses_native_transformer_streaming:
            raise RuntimeError(
                self.native_unavailable_reason
                or "native Diffusers per-weight streaming is unavailable"
            )
        if model_factory is None:
            raise RuntimeError("target mapping unavailable: no model structure factory")
        model = model_factory()
        ownership = derive_linear_ownership(
            model,
            targets,
            min_layer_size=self.min_layer_size,
            residual_match=residual_match,
        )
        return self._stream_config_factory(ownership.exclusions), ownership

    def transformer_stream_config(self, targets, *, model_factory=None):
        config, _ = self.transformer_stream_plan(targets, model_factory=model_factory)
        return config

    def convert_module(
        self,
        module,
        *,
        device,
        fp8_layers=None,
        fp8_suffix_layers=None,
        hybrid=False,
        filter_fn=None,
    ):
        raise NotImplementedError

    def convert_block(self, block, *, device, **kwargs):
        return self.convert_module(block, device=device, **kwargs)


class TorchaoNvfp4BackendAdapter(FormatBackendAdapter):
    storage_semantics = "torchao_nvfp4_dynamic_per_tensor"
    parameter_semantics = "torchao_nvfp4_tensor_subclass"
    supports_precision_overrides = True

    def _stream_config_factory(self, exclusions):
        return _torchao_stream_config("nvfp4", exclusions)

    def convert_module(
        self,
        module,
        *,
        device,
        fp8_layers=None,
        fp8_suffix_layers=None,
        hybrid=False,
        filter_fn=None,
    ):
        if hybrid:
            raise RuntimeError(
                "CUDA NVFP4 does not implement the runtime hybrid FP8/FP4 " "schedule"
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_nvfp4

        return quantize_linear_layers_to_nvfp4(
            module,
            fp8_layers=fp8_layers,
            fp8_suffix_layers=fp8_suffix_layers,
            device=device,
            filter_fn=filter_fn,
        )


class AiterMxfp4BackendAdapter(FormatBackendAdapter):
    storage_semantics = "aiter_mxfp4_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "replicated_scale_buffer"
    serialization = "packed_state_supported_not_portable"
    supports_precision_overrides = True

    def convert_module(
        self,
        module,
        *,
        device,
        fp8_layers=None,
        fp8_suffix_layers=None,
        hybrid=False,
        filter_fn=None,
    ):
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp4

        return quantize_linear_layers_to_fp4(
            module,
            fp8_layers=fp8_layers,
            fp8_suffix_layers=fp8_suffix_layers,
            use_hybrid_schedule=hybrid,
            device=device,
            filter_fn=filter_fn,
        )


class TorchaoInt8BackendAdapter(FormatBackendAdapter):
    storage_semantics = "torchao_w8a8_dynamic_per_row_symmetric"
    parameter_semantics = "torchao_int8_tensor_subclass"
    min_layer_size = 512

    def _stream_config_factory(self, exclusions):
        return _torchao_stream_config("int8", exclusions)

    def convert_module(self, module, *, device, filter_fn=None, **kwargs):
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_int8

        return quantize_linear_layers_to_int8(
            module,
            device=device,
            min_layer_size=self.min_layer_size,
            filter_fn=filter_fn,
        )


def module_paths_overlap(left: str, right: str) -> bool:
    """Whether two module paths are equal or one is a dotted ancestor."""

    if not left or not right:
        return True
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def module_path_is_covered(path: str, owner: str) -> bool:
    """Whether ``owner`` is the same dotted path as, or an ancestor of, ``path``."""

    return not owner or path == owner or path.startswith(f"{owner}.")


def _torchao_stream_config(kind: str, exclusions):
    from diffusers import TorchAoConfig
    from xfuser.model_executor.quant.torchao_quantizer import (
        register_torchao_fp32_policy,
    )

    register_torchao_fp32_policy()
    if kind == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )

        config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
    else:
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quant_primitives import MappingType
        from torchao.quantization.quant_api import (
            Int8DynamicActivationInt8WeightConfig,
        )

        config = Int8DynamicActivationInt8WeightConfig(
            granularity=PerRow(),
            act_mapping_type=MappingType.SYMMETRIC,
            set_inductor_config=False,
        )
    return TorchAoConfig(config, modules_to_not_convert=list(exclusions))


def derive_linear_ownership(
    model,
    targets,
    *,
    min_layer_size: int = 0,
    residual_match: Callable[[str], bool] | None = None,
    is_linear=None,
) -> LinearOwnership:
    """Classify linear leaves for native streaming and post-load ownership."""

    if is_linear is None:
        from torch import nn

        def is_linear(module):
            return isinstance(module, nn.Linear)

    targets = tuple(dict.fromkeys(targets))
    for target in targets:
        try:
            model.get_submodule(target)
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
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


def derive_linear_exclusions(
    model,
    targets,
    *,
    min_layer_size: int = 0,
    is_linear=None,
) -> list[str]:
    """Translate positive target prefixes and size gates to Diffusers exclusions."""

    ownership = derive_linear_ownership(
        model,
        targets,
        min_layer_size=min_layer_size,
        is_linear=is_linear,
    )
    return list(ownership.exclusions)


def _precision_override_matcher(targets, prefixes, suffixes):
    targets = tuple(targets)
    prefixes = tuple(prefixes)
    suffixes = tuple(suffixes)

    def matches(name):
        for target in targets:
            if not module_path_is_covered(name, target):
                continue
            local_name = name if not target else name[len(target) :].lstrip(".")
            if prefixes and local_name.startswith(prefixes):
                return True
            if suffixes and local_name.endswith(suffixes):
                return True
        return False

    return matches


def _unsupported_error(contract):
    module = import_module(contract.requested_format.__class__.__module__)
    return getattr(module, "UnsupportedLoadContract", ValueError)


def select_format_backend(
    contract,
    *,
    capabilities: FormatBackendCapabilities,
    hybrid: bool = False,
):
    format_value = contract.requested_format.value
    backend_value = contract.selected_backend.value
    error = _unsupported_error(contract)
    if format_value not in {"fp4", "fp8_fp4", "int8"}:
        return None
    if format_value == "int8":
        if backend_value != "torchao" or not capabilities.torchao_int8:
            reason = capabilities.torchao_int8_reason or "backend unavailable"
            raise error(f"TorchAO INT8 backend is unavailable: {reason}")
        return TorchaoInt8BackendAdapter(
            backend=contract.selected_backend,
            format_=contract.requested_format,
            native_transformer_streaming=capabilities.torchao_int8_streaming,
            native_unavailable_reason=capabilities.torchao_int8_streaming_reason,
        )
    if backend_value == "torchao":
        if not capabilities.torchao_nvfp4:
            reason = capabilities.torchao_nvfp4_reason or "backend unavailable"
            raise error(f"NVFP4 backend is unavailable: {reason}")
        if hybrid:
            raise error(
                "CUDA NVFP4 does not implement the runtime hybrid FP8/FP4 "
                "schedule; disable --use_hybrid_gemm_schedule or use "
                "ROCm with AITER MXFP4"
            )
        return TorchaoNvfp4BackendAdapter(
            backend=contract.selected_backend,
            format_=contract.requested_format,
            native_transformer_streaming=capabilities.torchao_nvfp4_streaming,
            native_unavailable_reason=capabilities.torchao_nvfp4_streaming_reason,
        )
    if backend_value == "aiter":
        if not capabilities.aiter_mxfp4:
            reason = capabilities.aiter_mxfp4_reason or "backend unavailable"
            raise error(f"AITER MXFP4 backend is unavailable: {reason}")
        return AiterMxfp4BackendAdapter(
            backend=contract.selected_backend,
            format_=contract.requested_format,
            native_unavailable_reason=MXFP4_STREAMING_FALLBACK,
        )
    raise error(f"{backend_value} cannot store {format_value}")


def _descriptor(adapter, component_name, mode, fallback=None):
    return FormatLoadDescriptor(
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


def prepare_native_transformer_format_load(
    adapter,
    *,
    component_name,
    targets,
    stream_quant,
    model_factory=None,
    precision_prefixes=(),
    precision_suffixes=(),
    hybrid=False,
) -> PreparedFormatLoad:
    targets = tuple(targets)
    if not stream_quant:
        fallback = "streaming disabled by the runner"
    elif not targets:
        fallback = f"{component_name} has no {adapter.format.value.upper()} targets"
    elif isinstance(adapter, AiterMxfp4BackendAdapter):
        fallback = adapter.native_unavailable_reason or MXFP4_STREAMING_FALLBACK
    elif isinstance(adapter, TorchaoNvfp4BackendAdapter) and hybrid:
        fallback = "native NVFP4 streaming cannot preserve hybrid FP8/FP4 ownership"
    else:
        residual_match = (
            _precision_override_matcher(targets, precision_prefixes, precision_suffixes)
            if isinstance(adapter, TorchaoNvfp4BackendAdapter)
            and (precision_prefixes or precision_suffixes)
            else None
        )
        try:
            config, ownership = adapter.transformer_stream_plan(
                targets,
                model_factory=model_factory,
                residual_match=residual_match,
            )
        except Exception as exc:
            fallback = str(exc)
        else:
            return PreparedFormatLoad(
                descriptor=_descriptor(adapter, component_name, "streaming"),
                quantization_config=config,
                streamed_targets=(
                    ownership.streamed if ownership.residual else targets
                ),
                residual_targets=ownership.residual,
            )
    return PreparedFormatLoad(
        descriptor=_descriptor(adapter, component_name, "post_load", fallback)
    )


def describe_blockwise_format_load(
    adapter,
    *,
    component_name,
    targets,
    wrap_attrs,
) -> FormatLoadDescriptor:
    targets = tuple(targets)
    wrap_attrs = tuple(wrap_attrs)
    if not targets:
        return _descriptor(
            adapter,
            component_name,
            "post_load",
            f"{component_name} has no {adapter.format.value.upper()} targets",
        )
    if not wrap_attrs or not any(
        module_paths_overlap(target, attr) for target in targets for attr in wrap_attrs
    ):
        return _descriptor(
            adapter,
            component_name,
            "post_load",
            "quantization targets do not align with streamed transformer blocks",
        )
    return _descriptor(adapter, component_name, "blockwise")


def plan_eager_blockwise_fallback(
    *,
    prepared,
    targets,
    wrap_attrs,
    world_size: int,
    standard_loader: bool,
    offload_requested: bool,
) -> EagerBlockwisePlan:
    """Decide whether an eager post-load fallback can use local block filling."""

    if prepared.descriptor.materialization_mode != "post_load":
        return EagerBlockwisePlan(False, "native loading already owns materialization")
    if world_size != 1:
        return EagerBlockwisePlan(False, "local blockwise loading requires one rank")
    if not standard_loader:
        return EagerBlockwisePlan(
            False, "loader does not expose the standard checkpoint seam"
        )
    if offload_requested:
        return EagerBlockwisePlan(
            False, "local blockwise loading does not support offload"
        )
    targets = tuple(targets)
    wrap_attrs = tuple(wrap_attrs)
    if not targets or not wrap_attrs:
        return EagerBlockwisePlan(
            False, "quantization targets or wrap attributes are empty"
        )
    if not all(
        any(module_path_is_covered(target, attr) for attr in wrap_attrs)
        for target in targets
    ):
        return EagerBlockwisePlan(
            False,
            "quantization targets are not fully owned by streamed transformer blocks",
        )
    return EagerBlockwisePlan(True)


def validate_format_fsdp_placement(
    contract,
    adapter,
    *,
    capabilities: FormatBackendCapabilities,
    required: bool,
) -> None:
    if not required or adapter is None:
        return
    if isinstance(adapter, TorchaoNvfp4BackendAdapter):
        available = capabilities.torchao_nvfp4_fsdp
        reason = capabilities.torchao_nvfp4_fsdp_reason
        label = "TorchAO NVFP4 tensor subclass"
    elif isinstance(adapter, TorchaoInt8BackendAdapter):
        available = capabilities.torchao_int8_fsdp
        reason = capabilities.torchao_int8_fsdp_reason
        label = "TorchAO INT8 tensor subclass"
    elif isinstance(adapter, AiterMxfp4BackendAdapter):
        available = capabilities.aiter_mxfp4_fsdp
        reason = capabilities.aiter_mxfp4_fsdp_reason
        label = "AITER MXFP4 packed weight"
    else:
        return
    if not available:
        suffix = f": {reason}" if reason else ""
        raise _unsupported_error(contract)(
            f"{label} cannot be placed under FSDP2{suffix}"
        )
