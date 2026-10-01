"""FP4/INT8 backend adapters and dependency-light materialization planning."""

import os
from dataclasses import dataclass
from importlib import import_module
from .quant_adapter import (
    Capabilities,
    FormatCapability,
    LinearOwnership,
    PreparedQuantLoad,
    QuantAdapter,
    stores,
    QuantLoadDescriptor,
    TargetMappingUnavailable,
    derive_linear_ownership,
    descriptor_for,
    module_path_is_covered,
    module_paths_overlap,
)
from importlib.metadata import PackageNotFoundError, version
from packaging.version import InvalidVersion, Version
from typing import Callable

_ProbeResult = bool | tuple[bool, str | None]
_MIN_TORCHAO_VERSION = Version("0.15.0")
MXFP4_STREAMING_FALLBACK = (
    "AITER MXFP4 conversion requires each full-precision weight before creating "
    "xFuserMXFP4Linear packed state; no safe Diffusers per-weight loader exists"
)
MXFP6_STREAMING_FALLBACK = (
    "AITER MXFP6 conversion requires each full-precision weight before creating "
    "xFuserMXFP6Linear packed state; no native Diffusers per-weight loader exists"
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


def _probe_aiter_mxfp6_apis(
    gcn_arch_probe: Callable[[], str | None] | None = None,
) -> tuple[bool, str | None]:
    """Probe the exact gfx950 AITER A6W6 surface used by xFuserMXFP6Linear."""

    try:
        module = import_module("aiter")
    except Exception as exc:  # noqa: BLE001 - report every capability probe failure
        return (
            False,
            f"AITER MXFP6 import probe failed: {type(exc).__name__}: {exc}",
        )

    required = (
        "quant_mxfp6_gemm",
        "gemm_a6w6",
        "mxfp6_gemm_pack_size",
    )
    for name in required:
        if not callable(getattr(module, name, None)):
            return (
                False,
                f"missing required AITER MXFP6 API: aiter.{name}",
            )

    arch = (gcn_arch_probe or _gcn_arch_name)()
    if arch is None:
        return False, "cannot determine the ROCm architecture for AITER MXFP6 support"
    if "gfx950" not in arch:
        return (
            False,
            f"AITER MXFP6 ASM kernels require gfx950, detected {arch}",
        )
    return True, None


def _probe_fsdp_non_float_parameters() -> tuple[bool, str | None]:
    """Require an FSDP2 that can wrap uint8 packed MXFP4/MXFP6 weights.

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
    if kind in {"mxfp4", "mxfp6"}:
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
    mxfp6_probe: Callable[[], _ProbeResult] | None = None,
    require_mxfp6: bool = False,
    nvfp4_probe: Callable[[], _ProbeResult] | None = None,
    int8_probe: Callable[[], _ProbeResult] | None = None,
    diffusers_probe: Callable[[str], _ProbeResult] | None = None,
    fsdp_probe: Callable[[str], _ProbeResult] | None = None,
) -> Capabilities:
    """Probe packages/hardware with injectable seams for routing tests.

    Each record is gated to the hardware its kernels run on, so a caller
    choosing between two implementations of one format takes the first
    available one rather than repeating the hardware test.
    """

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

    probe_mxfp6 = require_mxfp6 or mxfp6_probe is not None
    if probe_mxfp6 and mxfp6_probe is None:
        if aiter_probe is None:
            mxfp6_probe = _probe_aiter_mxfp6_apis
        else:

            def mxfp6_probe():
                available = bool(aiter_probe())
                return (
                    available,
                    None if available else "AITER MXFP6 APIs are unavailable",
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
        if probe_mxfp6:
            mxfp6, mxfp6_reason = _result(mxfp6_probe())
        else:
            mxfp6 = False
            mxfp6_reason = "AITER MXFP6 was not requested"
    else:
        mxfp4 = False
        mxfp4_reason = "AITER MXFP4 requires ROCm"
        mxfp6 = False
        mxfp6_reason = "AITER MXFP6 requires ROCm"
    if cuda:
        int8, int8_reason = _result(int8_probe())
    else:
        int8 = False
        int8_reason = "TorchAO INT8 is supported only on CUDA"

    def derived(available, reason, kind, *, streaming):
        """Streaming and sharding for one record, probed only if it can be stored."""
        if not available:
            return FormatCapability(available=False, reason=reason)
        streams, streams_reason = (
            _result(diffusers_probe(kind)) if streaming else (False, None)
        )
        shards, shards_reason = _result(fsdp_probe(kind))
        return FormatCapability(
            available=True,
            reason=reason,
            streams=streams,
            streams_reason=streams_reason,
            fsdp_safe=shards,
            fsdp_reason=shards_reason,
        )

    return Capabilities(
        {
            ("fp4", "torchao"): derived(nvfp4, nvfp4_reason, "nvfp4", streaming=True),
            ("fp4", "aiter"): derived(mxfp4, mxfp4_reason, "mxfp4", streaming=False),
            ("fp6", "aiter"): derived(mxfp6, mxfp6_reason, "mxfp6", streaming=False),
            ("int8", "torchao"): derived(int8, int8_reason, "int8", streaming=True),
        }
    )


@dataclass(frozen=True)
class EagerBlockwisePlan:
    enabled: bool
    reason: str | None = None


@stores("fp4", "torchao")
class TorchaoNvfp4BackendAdapter(QuantAdapter):
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    # Streaming would own every targeted leaf, leaving the hybrid
    # wrapper nothing to build its second precision from.
    streams_under_hybrid = False
    storage_semantics = "torchao_nvfp4_dynamic_per_tensor"
    parameter_semantics = "torchao_nvfp4_tensor_subclass"
    supports_precision_overrides = True

    def _stream_config_factory(self, exclusions):
        return _torchao_stream_config("nvfp4", exclusions)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(_torchao_quant_config("nvfp4"), device)

    def convert_module(
        self, module, *, device, filter_fn=None, offload_to_cpu=False, companion=None
    ):
        if companion is not None:
            from xfuser.core.utils.runner_utils import replace_linears

            return replace_linears(
                module,
                self.layer_factory(device=device, companion=companion),
                filter_fn=filter_fn,
                offload_to_cpu=offload_to_cpu,
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_nvfp4

        return quantize_linear_layers_to_nvfp4(
            module, device=device, filter_fn=filter_fn
        )


@stores("fp4", "aiter")
class AiterMxfp4BackendAdapter(QuantAdapter):
    meta_layout_matches_storage = True
    streams_by_exclusion = False
    # Group offloading moves a module's parameters between host and device
    # around each call, and these weights survive neither leg. Both failures
    # land below Python -- one inside the hook, one as AITER's own abort with
    # no traceback -- so neither can be caught where it happens.
    group_offload_refusal = {
        "pinned": "torch cannot pin a Float4_e2m1fn_x2 tensor",
        "unpinned": (
            "AITER binds a device from the parameter it is given, and a host "
            "parameter resolves to an invalid ordinal"
        ),
    }
    storage_semantics = "aiter_mxfp4_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "replicated_scale_buffer"
    serialization = "packed_state_supported_not_portable"
    supports_precision_overrides = True

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import mxfp4_layer_factory

        return mxfp4_layer_factory(device)

    def convert_module(
        self, module, *, device, filter_fn=None, offload_to_cpu=False, companion=None
    ):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("fp6", "aiter")
class AiterMxfp6BackendAdapter(QuantAdapter):
    meta_layout_matches_storage = True
    streams_by_exclusion = False
    storage_semantics = "aiter_mxfp6_e2m3_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "persistent_scale_buffer"
    serialization = "packed_state_supported_not_portable"

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import packed_layer_factory
        from xfuser.model_executor.layers.mxfp6_linear import xFuserMXFP6Linear

        return packed_layer_factory(xFuserMXFP6Linear, device)

    def convert_module(
        self,
        module,
        *,
        device,
        filter_fn=None,
        offload_to_cpu=False,
        companion=None,
        **kwargs,
    ):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("int8", "torchao")
class TorchaoInt8BackendAdapter(QuantAdapter):
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    storage_semantics = "torchao_w8a8_dynamic_per_row_symmetric"
    parameter_semantics = "torchao_int8_tensor_subclass"
    min_layer_size = 512

    def _stream_config_factory(self, exclusions):
        return _torchao_stream_config("int8", exclusions)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(_torchao_quant_config("int8"), device)

    def convert_module(
        self, module, *, device, filter_fn=None, companion=None, **kwargs
    ):
        if companion is not None:
            from xfuser.core.utils.runner_utils import replace_linears

            return replace_linears(
                module,
                self.layer_factory(device=device, companion=companion),
                filter_fn=filter_fn,
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_int8

        return quantize_linear_layers_to_int8(
            module,
            device=device,
            min_layer_size=self.min_layer_size,
            filter_fn=filter_fn,
        )


def _torchao_stream_config(kind: str, exclusions):
    """The Diffusers wrapper around the quant config, for a module-wide load."""

    from diffusers import TorchAoConfig

    return TorchAoConfig(
        _torchao_quant_config(kind), modules_to_not_convert=list(exclusions)
    )


def _torchao_quant_config(kind: str):
    """The torchao config itself, which `quantize_` takes for one leaf or a tree.

    Split out so a single leaf can be built with exactly the settings a
    module-wide load would have used -- that is what gives NVFP4 and INT8 a
    per-leaf seam, and with it a place in a per-step pair.
    """

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
    return config


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
