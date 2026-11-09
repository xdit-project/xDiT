"""What this machine can store, for every GEMM format this build knows.

A backend is a ``(format, implementation)`` pair: "fp8 via aiter" is block-128
scales through ``gemm_a8w8_blockscale``, "fp8 via torchao" is per-tensor dynamic
scaling. This module holds the probe for each pair; ``backend_adapters`` holds
the ``QuantAdapter`` that does the storing, and which pair a run uses is
``backend_selection``'s question.

Every pair is probed the same way, in one table: the version gate, the
Diffusers per-weight API check and the FSDP2 check each exist once and serve
whichever format asks. A record says what was measured and why, so a refusal
names the probe that failed rather than calling a format "unavailable".

Probing is narrowed by ``wanted`` and gated by hardware, and nothing imports
torch at module scope: a caller that injects the probes it needs pays for no
import it will never consult.
"""

import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

from packaging.version import InvalidVersion, Version

from .contracts import AITER_FP8, TORCHAO_FP8
from .quant_adapter import Capabilities, FormatCapability

_ProbeResult = bool | tuple[bool, str | None]
_MIN_TORCHAO_VERSION = Version("0.15.0")

# The architectures AITER has FP4 kernels for, per its own arch_info.is_fp4_avail. Its build gate
# is wider than that (aiter/jit/core.py compiles -D__Float4_e2m1fn_x2 for anything but gfx942 when
# AITER_FP4x2 is enabled), but the kernels behind the define are narrower: the hand-written A4W4
# assembly covers gfx942 and gfx950, gemm_a4w4 then raises on gfx942, and only gfx950 carries tuned
# configs. RDNA4 has FP8 kernels and no FP4 ones. An arch off this list reaches an AITER_CHECK(false)
# that aborts the process, so it has to be refused here rather than caught at the call site.
_AITER_FP4_ARCHS = ("gfx950", "gfx1250")

#: How each torchao config names itself in a probe's reason, so one probe can
#: serve every format and still say which one could not be built.
_TORCHAO_LABELS = {"fp8": "FP8 conversion", "nvfp4": "NVFP4", "int8": "INT8"}


def _result(value: _ProbeResult) -> tuple[bool, str | None]:
    """A probe's answer as a pair, whether it returned one or a bare bool."""

    if isinstance(value, tuple):
        return bool(value[0]), value[1]
    return bool(value), None


# ---------------------------------------------------------------------------
# TorchAO
# ---------------------------------------------------------------------------


def _torchao_version_gate() -> tuple[bool, str | None]:
    """Whether an installed torchao is new enough for any of its configs."""

    try:
        installed = Version(version("torchao"))
    except PackageNotFoundError:
        return False, "torchao is not installed"
    except InvalidVersion as exc:
        return False, f"cannot parse installed torchao version: {exc}"
    if installed < _MIN_TORCHAO_VERSION:
        return (
            False,
            f"torchao {installed} is older than required {_MIN_TORCHAO_VERSION}",
        )
    return True, None


def _probe_torchao_config(kind: str) -> tuple[bool, str | None]:
    """Whether torchao here can build the config ``kind`` is stored through.

    One probe for every torchao format. The version gate is the same for all of
    them; what differs is the config each builds and, for FP8, the two extra
    symbols its conversion path goes through. ``kind`` names the torchao config
    rather than the GEMM format, because two formats could share one.
    """

    label = _TORCHAO_LABELS[kind]
    available, reason = _torchao_version_gate()
    if not available:
        return False, reason
    try:
        if kind == "fp8":
            quant = import_module("torchao.quantization.quant_api")
            granularity = import_module("torchao.quantization.granularity")
            common = import_module("torchao.quantization.quantize_.common")
            from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR

            config = quant.Float8DynamicActivationFloat8WeightConfig(
                granularity=granularity.PerTensor(),
                set_inductor_config=False,
                kernel_preference=common.KernelPreference.AUTO,
                # Probe what conversion will build, floor included. The config
                # is discarded, but every construction site carries the floor
                # -- an unfloored one turns an all-zero activation into NaN,
                # and `test_every_place_the_config_is_built_sets_the_floor`
                # holds the rule for all of them.
                activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
            )
            # Conversion goes through these two as well, so a build carrying
            # the config class but not them still cannot store FP8.
            if not all(callable(getattr(quant, name, None)) for name in ("quantize_", "_is_linear")):
                return False, "TorchAO FP8 conversion APIs are not callable"
        elif kind == "nvfp4":
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
            return False, f"TorchAO {label} config is unavailable"
    except Exception as exc:
        return (
            False,
            f"TorchAO {label} API probe failed: {type(exc).__name__}: {exc}",
        )
    return True, None


def _probe_torchao_fp8_conversion_api() -> tuple[bool, str | None]:
    """Import and instantiate the exact TorchAO APIs used by FP8 conversion."""

    return _probe_torchao_config("fp8")


def _probe_diffusers_config(kind: str) -> tuple[bool, str | None]:
    """Whether Diffusers can quantize this config per weight during a load."""

    available, reason = _probe_torchao_config(kind)
    if not available:
        return False, reason
    try:
        diffusers = import_module("diffusers")
        quantizer = import_module("diffusers.quantizers.torchao.torchao_quantizer")
        from .backend_adapters import torchao_stream_config

        config = torchao_stream_config(kind, [])
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
            f"cannot query CUDA capability for TorchAO FP8: {type(exc).__name__}: {exc}",
        )
    if capability is None or capability < (8, 9):
        observed = "unknown" if capability is None else f"{capability[0]}.{capability[1]}"
        return (
            False,
            f"TorchAO FP8 requires CUDA capability >= 8.9; observed {observed}",
        )
    return True, None


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
        from .text_encoder_adapter import TextEncoderFrameworkAdapter

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
            (None if available else "Transformers TorchAO quantize-on-load methods are unavailable"),
        )
    except Exception as exc:
        return (
            False,
            f"TorchAO text-encoder framework API probe failed: {type(exc).__name__}: {exc}",
        )


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
            f"TorchAO FSDP patch validation failed: {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# AITER
# ---------------------------------------------------------------------------


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


def _probe_aiter_a6w4_apis(
    gcn_arch_probe: Callable[[], str | None] | None = None,
) -> tuple[bool, str | None]:
    """Probe the exact gfx950 AITER A6W4 surface used by xFuserA6W4Linear.

    A6W4 reads an FP4 weight through one AITER module and an FP6 activation
    through another, so both submodules and both packed layouts are checked
    here rather than trusting the single ``import aiter`` the MX formats need.
    """

    if os.getenv("AITER_TRITON_ONLY", "0") == "1":
        return False, "AITER A6W4 requires AITER ASM, not AITER_TRITON_ONLY"
    try:
        a6w4 = import_module("aiter.ops.gemm_op_a6w4")
        a6w6 = import_module("aiter.ops.gemm_op_a6w6")
    except Exception as exc:  # noqa: BLE001 - report every capability probe failure
        return (
            False,
            f"AITER A6W4 import probe failed: {type(exc).__name__}: {exc}",
        )

    required = (
        ("gemm_a6w4", getattr(a6w4, "gemm_a6w4", None)),
        ("mxfp4_gemm_pack_size", getattr(a6w4, "mxfp4_gemm_pack_size", None)),
        ("quant_mxfp4_gemm_hip_out", getattr(a6w4, "quant_mxfp4_gemm_hip_out", None)),
        ("mxfp6_gemm_pack_size", getattr(a6w6, "mxfp6_gemm_pack_size", None)),
        ("quant_mxfp6_gemm", getattr(a6w6, "quant_mxfp6_gemm", None)),
    )
    missing = [name for name, value in required if not callable(value)]
    if missing:
        return (
            False,
            "missing required AITER mixed-MXFP APIs: " + ", ".join(missing),
        )

    expected_fp4 = (3 * 16384, 3 * 1024)
    expected_fp6 = (3 * 24576, 3 * 1024)
    try:
        actual_fp4 = tuple(int(value) for value in a6w4.mxfp4_gemm_pack_size(256, 128))
        actual_fp6 = tuple(int(value) for value in a6w6.mxfp6_gemm_pack_size(256, 128))
    except Exception as exc:  # noqa: BLE001
        return False, f"AITER mixed-MXFP pack-size probe failed: {exc}"
    if actual_fp4 != expected_fp4 or actual_fp6 != expected_fp6:
        return (
            False,
            f"AITER mixed-MXFP packed-layout ABI mismatch: fp4={actual_fp4}, fp6={actual_fp6}",
        )

    arch = (gcn_arch_probe or _gcn_arch_name)()
    if arch is None:
        return False, "cannot determine the ROCm architecture for AITER A6W4 support"
    if "gfx950" not in arch:
        return False, f"AITER A6W4 ASM kernels require gfx950, detected {arch}"
    return True, None


def _probe_aiter_transformers_streaming() -> tuple[bool, str | None]:
    from .text_encoder_adapter import probe_transformers_streaming_loader

    support = probe_transformers_streaming_loader()
    return support.available, support.reason


# ---------------------------------------------------------------------------
# FSDP2
# ---------------------------------------------------------------------------


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
    """Whether what one implementation leaves in a block survives FSDP2 here.

    One probe for every pair, keyed by what the implementation stores rather
    than by its format: a packed parameter and a tensor subclass raise entirely
    different questions, and two formats stored the same way share an answer.
    """

    if kind == "nvfp4":
        return (
            False,
            "NVFP4 tensor-subclass FSDP gather/scatter support is not validated",
        )
    if kind in {"mxfp4", "mxfp6"}:
        return _probe_fsdp_non_float_parameters()
    if kind == "fp8":
        return _probe_torchao_fsdp_patches()
    if kind == "aiter_fp8":
        # Block-128 scales live in a plain nn.Parameter and a buffer, which is
        # why this one needs no Float8Tensor FSDP patches to be validated.
        return True, None
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
        missing = [name for name in methods if not callable(getattr(weight, name, None))]
        if missing:
            return False, "INT8 tensor subclass is missing " + ", ".join(missing)
    except Exception as exc:
        return False, f"INT8 FSDP probe failed: {type(exc).__name__}: {exc}"
    return True, None


# ---------------------------------------------------------------------------
# The one capability probe
# ---------------------------------------------------------------------------


def probe_backend_capabilities(
    *,
    wanted=None,
    cuda_probe: Callable[[], bool] | None = None,
    hip_probe: Callable[[], bool] | None = None,
    cuda_capability_probe: Callable[[], tuple[int, int] | None] | None = None,
    aiter_probe: Callable[[], bool] | None = None,
    aiter_fp8_probe: Callable[[], _ProbeResult] | None = None,
    aiter_transformers_probe: Callable[[], _ProbeResult] | None = None,
    mxfp4_probe: Callable[[], _ProbeResult] | None = None,
    mxfp6_probe: Callable[[], _ProbeResult] | None = None,
    a6w4_probe: Callable[[], _ProbeResult] | None = None,
    nvfp4_probe: Callable[[], _ProbeResult] | None = None,
    int8_probe: Callable[[], _ProbeResult] | None = None,
    torchao_fp8_probe: Callable[[], _ProbeResult] | None = None,
    torchao_fp8_accelerator_probe: Callable[[], _ProbeResult] | None = None,
    torchao_text_encoder_probe: Callable[[], _ProbeResult] | None = None,
    diffusers_probe: Callable[[str], _ProbeResult] | None = None,
    fsdp_probe: Callable[[str], _ProbeResult] | None = None,
) -> Capabilities:
    """Probe every ``(format, implementation)`` pair, with injectable seams.

    Each record is gated to the hardware its kernels run on, so a caller
    choosing between two implementations of one format takes the first
    available one rather than repeating the hardware test.

    ``wanted`` narrows the probing to the formats a run actually named; None
    probes everything. Some probes are not free -- MXFP6's imports AITER
    kernels, FP8's imports torchao, Diffusers and Transformers -- and a run
    that will never place a format has no reason to pay for it. Every format is
    narrowed the same way: when the narrowing covered only some of them, the
    formats left out were probed on every run no matter what it asked for.

    ``aiter_probe`` is the generic "are the AITER MX APIs here" seam that
    stands in for the MXFP4 and MXFP6 probes; ``aiter_fp8_probe`` is the
    separate RDNA4 block-scale gate. They were one argument name in two
    functions meaning two different things.
    """

    def is_wanted(format_name):
        return wanted is None or format_name in wanted

    accelerator = {}

    def accelerators():
        """CUDA/HIP, resolved once and only if a probe in play needs them.

        Lazily, because the default probes live in ``xfuser.envs``, which
        imports torch: a caller that injects everything it consults -- which is
        every routing test -- must not pay for that import.
        """
        nonlocal cuda_probe, hip_probe
        if not accelerator:
            if cuda_probe is None or hip_probe is None:
                from xfuser.envs import _is_cuda, _is_hip

                cuda_probe = cuda_probe or _is_cuda
                hip_probe = hip_probe or _is_hip
            accelerator["cuda"] = bool(cuda_probe())
            accelerator["hip"] = bool(hip_probe())
        return accelerator

    def on_cuda():
        return accelerators()["cuda"]

    def on_hip():
        return accelerators()["hip"]

    def on_blackwell():
        nonlocal cuda_capability_probe
        if not on_cuda():
            return False
        if cuda_capability_probe is None:
            torch = import_module("torch")
            cuda_capability_probe = torch.cuda.get_device_capability
        capability = cuda_capability_probe()
        return capability is not None and capability >= (10, 0)

    def from_aiter(label):
        def probe():
            available = bool(aiter_probe())
            return available, None if available else f"AITER {label} APIs are unavailable"

        return probe

    if mxfp4_probe is None:
        mxfp4_probe = from_aiter("MXFP4") if aiter_probe else _probe_aiter_mxfp4_apis
    if mxfp6_probe is None:
        mxfp6_probe = from_aiter("MXFP6") if aiter_probe else _probe_aiter_mxfp6_apis
    if a6w4_probe is None:
        a6w4_probe = from_aiter("A6W4") if aiter_probe else _probe_aiter_a6w4_apis
    if aiter_fp8_probe is None:

        def aiter_fp8_probe():
            from xfuser.core.utils.runner_utils import _use_aiter_fp8_rdna4

            return bool(_use_aiter_fp8_rdna4())

    nvfp4_probe = nvfp4_probe or (lambda: _probe_torchao_config("nvfp4"))
    int8_probe = int8_probe or (lambda: _probe_torchao_config("int8"))
    torchao_fp8_probe = torchao_fp8_probe or _probe_torchao_fp8_conversion_api
    torchao_fp8_accelerator_probe = torchao_fp8_accelerator_probe or _probe_torchao_fp8_accelerator
    torchao_text_encoder_probe = torchao_text_encoder_probe or _probe_torchao_text_encoder_streaming
    aiter_transformers_probe = aiter_transformers_probe or _probe_aiter_transformers_streaming
    diffusers_probe = diffusers_probe or _probe_diffusers_config
    fsdp_probe = fsdp_probe or _probe_fsdp_support

    def derived(available, reason, *, streams=None, te=None, fsdp=None, unmet=None):
        """One record, built the same way whatever pair it describes.

        ``streams``, ``te`` and ``fsdp`` are thunks returning a probe result,
        run in that order and only once the pair is known to be storable at
        all; None means the question does not arise and the field stays False.
        ``unmet`` is the reason the sub-fields carry when it is not storable.
        """
        if not available:
            return FormatCapability(
                available=False,
                reason=reason,
                streams_reason=unmet,
                te_streams_reason=unmet,
                fsdp_reason=unmet,
            )

        def ask(probe):
            return _result(probe()) if probe is not None else (False, None)

        streams_ok, streams_reason = ask(streams)
        te_ok, te_reason = ask(te)
        shards, shards_reason = ask(fsdp)
        return FormatCapability(
            available=True,
            reason=reason,
            streams=streams_ok,
            streams_reason=streams_reason,
            te_streams=te_ok,
            te_streams_reason=te_reason,
            fsdp_safe=shards,
            fsdp_reason=shards_reason,
        )

    # One block per format, gated by its own hardware and its own place in
    # `wanted`, and nothing nested inside another format's branch: MXFP6 used
    # to be resolved inside FP4's, so `--gemm_quantization fp6` on CUDA was
    # told "AITER MXFP6 was not requested" about a format the run had named.

    if not is_wanted("fp4"):
        nvfp4, nvfp4_reason = False, "fp4 was not requested"
        mxfp4, mxfp4_reason = False, "fp4 was not requested"
    else:
        if on_blackwell():
            nvfp4, nvfp4_reason = _result(nvfp4_probe())
        else:
            nvfp4 = False
            nvfp4_reason = "NVFP4 requires CUDA capability >= 10.0"
        if on_hip():
            mxfp4, mxfp4_reason = _result(mxfp4_probe())
        else:
            mxfp4 = False
            mxfp4_reason = "AITER MXFP4 requires ROCm"

    if not is_wanted("fp6"):
        mxfp6, mxfp6_reason = False, "fp6 was not requested"
    elif on_hip():
        mxfp6, mxfp6_reason = _result(mxfp6_probe())
    else:
        mxfp6 = False
        mxfp6_reason = "AITER MXFP6 requires ROCm"

    if not is_wanted("a6w4"):
        a6w4, a6w4_reason = False, "a6w4 was not requested"
    elif on_hip():
        a6w4, a6w4_reason = _result(a6w4_probe())
    else:
        a6w4 = False
        a6w4_reason = "AITER A6W4 requires ROCm"

    if not is_wanted("int8"):
        int8, int8_reason = False, "int8 was not requested"
    elif on_cuda():
        int8, int8_reason = _result(int8_probe())
    else:
        int8 = False
        int8_reason = "TorchAO INT8 is supported only on CUDA"

    if not is_wanted("fp8"):
        aiter_fp8 = torchao_fp8 = False
        aiter_fp8_reason = torchao_fp8_reason = "fp8 was not requested"
        aiter_te = (False, "fp8 was not requested")
    else:
        aiter_fp8 = bool(_result(aiter_fp8_probe())[0])
        aiter_fp8_reason = None if aiter_fp8 else "AITER FP8 requires RDNA4"
        aiter_te = _result(aiter_transformers_probe()) if aiter_fp8 else (False, "AITER FP8 backend is unavailable")
        accelerator_ok, accelerator_reason = _result(torchao_fp8_accelerator_probe())
        if accelerator_ok:
            torchao_fp8, torchao_fp8_reason = _result(torchao_fp8_probe())
        else:
            torchao_fp8 = False
            torchao_fp8_reason = accelerator_reason or "TorchAO FP8 requires CUDA or HIP/ROCm"

    return Capabilities(
        {
            ("fp8", AITER_FP8): derived(
                aiter_fp8,
                aiter_fp8_reason,
                streams=lambda: True,
                te=lambda: aiter_te,
                fsdp=lambda: fsdp_probe("aiter_fp8"),
                unmet="AITER FP8 backend is unavailable",
            ),
            ("fp8", TORCHAO_FP8): derived(
                torchao_fp8,
                torchao_fp8_reason,
                streams=lambda: diffusers_probe("fp8"),
                te=torchao_text_encoder_probe,
                fsdp=lambda: fsdp_probe("fp8"),
                unmet=torchao_fp8_reason,
            ),
            ("fp4", "torchao"): derived(
                nvfp4,
                nvfp4_reason,
                streams=lambda: diffusers_probe("nvfp4"),
                fsdp=lambda: fsdp_probe("nvfp4"),
            ),
            ("fp4", "aiter"): derived(mxfp4, mxfp4_reason, fsdp=lambda: fsdp_probe("mxfp4")),
            ("fp6", "aiter"): derived(mxfp6, mxfp6_reason, fsdp=lambda: fsdp_probe("mxfp6")),
            ("a6w4", "aiter"): derived(
                a6w4,
                a6w4_reason,
                # Not asked of FSDP2 at all: the converter runs after the
                # component is built, so there is no packed parameter for a
                # wrapped block to shard in the first place.
                fsdp=lambda: (False, "A6W4 currently requires eager materialization"),
            ),
            ("int8", "torchao"): derived(
                int8,
                int8_reason,
                streams=lambda: diffusers_probe("int8"),
                fsdp=lambda: fsdp_probe("int8"),
            ),
        }
    )
