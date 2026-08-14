"""Dependency-light contracts for FP4 and INT8 materialization backends."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKENDS_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/format_backends.py"
)
CONTRACTS_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/contracts.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    contracts = _load_module(CONTRACTS_PATH, "format_adapter_contracts")
    backends = _load_module(BACKENDS_PATH, "format_adapter_backends")
    return SimpleNamespace(contracts=contracts, backends=backends)


def _contract(modules, format_name, backend_name, mode_name="EAGER"):
    c = modules.contracts
    return SimpleNamespace(
        requested_format=getattr(c.QuantizationFormat, format_name),
        selected_backend=getattr(c.QuantizationBackend, backend_name),
        materialization_mode=getattr(c.MaterializationMode, mode_name),
    )


@pytest.mark.parametrize(
    (
        "cuda",
        "hip",
        "capability",
        "aiter",
        "format_name",
        "backend_name",
        "adapter_name",
    ),
    [
        (True, False, (10, 0), False, "FP4", "TORCHAO", "TorchaoNvfp4BackendAdapter"),
        (False, True, None, True, "FP4", "AITER", "AiterMxfp4BackendAdapter"),
        (True, False, (8, 9), False, "INT8", "TORCHAO", "TorchaoInt8BackendAdapter"),
    ],
)
def test_hardware_routing_matrix_is_injectable(
    modules,
    cuda,
    hip,
    capability,
    aiter,
    format_name,
    backend_name,
    adapter_name,
):
    b = modules.backends
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: cuda,
        hip_probe=lambda: hip,
        cuda_capability_probe=lambda: capability,
        aiter_probe=lambda: aiter,
        nvfp4_probe=lambda: (True, None),
        int8_probe=lambda: (True, None),
        diffusers_probe=lambda config_kind: (True, None),
        fsdp_probe=lambda config_kind: (True, None),
    )

    adapter = b.select_format_backend(
        _contract(modules, format_name, backend_name),
        capabilities=capabilities,
    )

    assert type(adapter).__name__ == adapter_name


def test_nvfp4_requires_blackwell_before_adapter_selection(modules):
    b = modules.backends
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (9, 0),
        aiter_probe=lambda: False,
        nvfp4_probe=lambda: pytest.fail("must not import NVFP4 APIs"),
        int8_probe=lambda: (True, None),
        diffusers_probe=lambda config_kind: (True, None),
        fsdp_probe=lambda config_kind: (True, None),
    )

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match=r"NVFP4.*capability.*10.0",
    ):
        b.select_format_backend(
            _contract(modules, "FP4", "TORCHAO"),
            capabilities=capabilities,
        )


def test_cuda_nvfp4_hybrid_is_rejected_before_backend_use(modules):
    b = modules.backends
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (10, 0),
        mxfp4_probe=lambda: pytest.fail("must not probe AITER on CUDA"),
        nvfp4_probe=lambda: (True, None),
        int8_probe=lambda: (True, None),
        diffusers_probe=lambda config_kind: (True, None),
        fsdp_probe=lambda config_kind: (True, None),
    )

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match=r"CUDA NVFP4.*hybrid.*ROCm.*AITER",
    ):
        b.select_format_backend(
            _contract(modules, "FP4", "TORCHAO"),
            capabilities=capabilities,
            hybrid=True,
        )


def test_nvfp4_adapter_never_silently_ignores_hybrid(modules):
    b = modules.backends
    adapter = b.TorchaoNvfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.FP4,
    )

    with pytest.raises(RuntimeError, match=r"NVFP4.*hybrid"):
        adapter.convert_module(object(), device="cuda:0", hybrid=True)


def test_rocm_aiter_mxfp4_hybrid_remains_supported(modules):
    b = modules.backends
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: False,
        hip_probe=lambda: True,
        cuda_capability_probe=lambda: None,
        mxfp4_probe=lambda: (True, None),
        nvfp4_probe=lambda: pytest.fail("must not probe NVFP4 on ROCm"),
        int8_probe=lambda: pytest.fail("must not probe INT8 on ROCm"),
        diffusers_probe=lambda config_kind: pytest.fail(
            "must not probe Diffusers for MXFP4"
        ),
        fsdp_probe=lambda config_kind: (True, None)
        if config_kind == "mxfp4"
        else pytest.fail(f"must not probe {config_kind} FSDP on ROCm"),
    )

    adapter = b.select_format_backend(
        _contract(modules, "FP4", "AITER"),
        capabilities=capabilities,
        hybrid=True,
    )

    assert isinstance(adapter, b.AiterMxfp4BackendAdapter)


@pytest.mark.parametrize(
    ("missing", "expected_reason"),
    [
        ("get_hip_quant", "aiter.get_hip_quant"),
        ("per_1x32", "aiter.QuantType.per_1x32"),
        ("gemm_a4w4", "aiter.gemm_a4w4"),
        ("shuffle_weight", "aiter.ops.shuffle.shuffle_weight"),
    ],
)
def test_aiter_mxfp4_probe_requires_exact_runtime_symbols(
    modules,
    monkeypatch,
    missing,
    expected_reason,
):
    b = modules.backends
    quant_type = SimpleNamespace(per_1x32=object())
    aiter = SimpleNamespace(
        get_hip_quant=lambda quant: object(),
        QuantType=quant_type,
        gemm_a4w4=lambda *args, **kwargs: object(),
    )
    shuffle = SimpleNamespace(shuffle_weight=lambda weight, layout: weight)
    if missing == "get_hip_quant":
        aiter.get_hip_quant = None
    elif missing == "per_1x32":
        aiter.QuantType = SimpleNamespace()
    elif missing == "gemm_a4w4":
        aiter.gemm_a4w4 = None
    else:
        shuffle.shuffle_weight = None

    monkeypatch.setattr(
        b,
        "import_module",
        lambda name: {
            "aiter": aiter,
            "aiter.ops.shuffle": shuffle,
        }[name],
    )

    available, reason = b._probe_aiter_mxfp4_apis()

    assert not available
    assert expected_reason in reason


def test_aiter_mxfp4_probe_rejects_architectures_without_fp4_kernels(
    modules, monkeypatch
):
    b = modules.backends
    aiter = SimpleNamespace(
        get_hip_quant=lambda quant: object(),
        QuantType=SimpleNamespace(per_1x32=object()),
        gemm_a4w4=lambda *args, **kwargs: object(),
    )
    shuffle = SimpleNamespace(shuffle_weight=lambda weight, layout: weight)
    monkeypatch.setattr(
        b,
        "import_module",
        lambda name: {"aiter": aiter, "aiter.ops.shuffle": shuffle}[name],
    )
    monkeypatch.setattr(b, "_gcn_arch_name", lambda: "gfx942:sramecc+:xnack-")

    available, reason = b._probe_aiter_mxfp4_apis()

    assert not available
    assert "gfx942" in reason


@pytest.mark.parametrize(
    ("arch", "fp4x2", "expected_available", "expected_reason"),
    [
        ("gfx950:sramecc+:xnack-", None, True, None),
        ("gfx1250", None, True, None),
        # RDNA4 runs AITER FP8 and has no FP4 kernels, so asking for FP4 there has to be
        # refused in preflight; reaching AITER aborts the process instead of raising.
        ("gfx1201", None, False, "gfx1201"),
        ("gfx1200", None, False, "gfx1200"),
        ("gfx1100", None, False, "gfx1100"),
        ("gfx942:sramecc+:xnack-", None, False, "gfx942"),
        ("gfx950", "0", False, "AITER_FP4x2=0"),
        (None, None, False, "cannot determine the ROCm architecture"),
    ],
)
def test_aiter_fp4_kernel_probe_accepts_only_archs_with_fp4_kernels(
    modules,
    monkeypatch,
    arch,
    fp4x2,
    expected_available,
    expected_reason,
):
    b = modules.backends
    if fp4x2 is None:
        monkeypatch.delenv("AITER_FP4x2", raising=False)
    else:
        monkeypatch.setenv("AITER_FP4x2", fp4x2)

    available, reason = b._probe_aiter_fp4_kernels(gcn_arch_probe=lambda: arch)

    assert available is expected_available
    if expected_reason is None:
        assert reason is None
    else:
        assert expected_reason in reason


def test_aiter_mxfp4_capability_preserves_symbol_probe_reason(modules):
    b = modules.backends
    calls = []
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: False,
        hip_probe=lambda: True,
        cuda_capability_probe=lambda: None,
        mxfp4_probe=lambda: calls.append("mxfp4")
        or (
            False,
            "missing required AITER MXFP4 API: aiter.gemm_a4w4",
        ),
        nvfp4_probe=lambda: pytest.fail("must not probe NVFP4"),
        int8_probe=lambda: pytest.fail("must not probe INT8"),
        diffusers_probe=lambda config_kind: pytest.fail("must not probe Diffusers"),
        fsdp_probe=lambda config_kind: pytest.fail("must not probe TorchAO FSDP"),
    )

    assert calls == ["mxfp4"]
    assert capabilities.aiter_mxfp4 is False
    assert (
        capabilities.aiter_mxfp4_reason
        == "missing required AITER MXFP4 API: aiter.gemm_a4w4"
    )


def test_int8_exclusions_preserve_targets_and_minimum_layer_size(modules):
    b = modules.backends

    class Linear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features

    class FakeModel:
        def named_modules(self):
            return [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.large", Linear(1024, 512)),
                ("blocks.0.small", Linear(1024, 256)),
                ("input_proj", Linear(1024, 1024)),
            ]

        def get_submodule(self, name):
            if name == "blocks":
                return object()
            raise AttributeError(name)

    exclusions = b.derive_linear_exclusions(
        FakeModel(),
        ("blocks",),
        min_layer_size=512,
        is_linear=lambda module: isinstance(module, Linear),
    )

    assert exclusions == ["blocks.0.small", "input_proj"]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("transformer", "transformer.blocks", True),
        ("transformer.blocks.attn", "transformer.blocks", True),
        ("transformer.blocks", "transformer.blocks", True),
        ("foo.bar", "foo.barn", False),
        ("foo.barn", "foo.bar", False),
    ],
)
def test_module_path_overlap_is_ancestor_aware_and_boundary_safe(
    modules,
    left,
    right,
    expected,
):
    assert modules.backends.module_paths_overlap(left, right) is expected


def test_component_root_target_preserves_int8_minimum_size_exclusions(modules):
    b = modules.backends

    class Linear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features

    class FakeModel:
        def named_modules(self):
            return [
                ("", self),
                ("large", Linear(512, 512)),
                ("small", Linear(512, 256)),
            ]

        def get_submodule(self, name):
            if name == "":
                return self
            raise AttributeError(name)

    exclusions = b.derive_linear_exclusions(
        FakeModel(),
        ("",),
        min_layer_size=512,
        is_linear=lambda module: isinstance(module, Linear),
    )

    assert exclusions == ["small"]


def test_component_root_target_aligns_with_blockwise_wrapped_paths(modules):
    b = modules.backends
    adapter = b.TorchaoInt8BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.INT8,
    )

    descriptor = b.describe_blockwise_format_load(
        adapter,
        component_name="transformer",
        targets=("",),
        wrap_attrs=("blocks",),
    )

    assert descriptor.materialization_mode == "blockwise"
    assert descriptor.fallback_reason is None


def test_eager_blockwise_fallback_accepts_owned_single_rank_target(modules):
    b = modules.backends
    prepared = SimpleNamespace(
        descriptor=SimpleNamespace(materialization_mode="post_load")
    )

    plan = b.plan_eager_blockwise_fallback(
        prepared=prepared,
        targets=("blocks",),
        wrap_attrs=("blocks",),
        world_size=1,
        standard_loader=True,
        offload_requested=False,
    )

    assert plan.enabled is True
    assert plan.reason is None


@pytest.mark.parametrize(
    ("prepared_mode", "targets", "wrap_attrs", "world_size", "standard", "offload"),
    [
        ("streaming", ("blocks",), ("blocks",), 1, True, False),
        ("post_load", ("tail",), ("blocks",), 1, True, False),
        ("post_load", ("blocks",), ("blocks",), 2, True, False),
        ("post_load", ("blocks",), ("blocks",), 1, False, False),
        ("post_load", ("blocks",), ("blocks",), 1, True, True),
    ],
)
def test_eager_blockwise_fallback_rejects_unsafe_plan(
    modules,
    prepared_mode,
    targets,
    wrap_attrs,
    world_size,
    standard,
    offload,
):
    plan = modules.backends.plan_eager_blockwise_fallback(
        prepared=SimpleNamespace(
            descriptor=SimpleNamespace(materialization_mode=prepared_mode)
        ),
        targets=targets,
        wrap_attrs=wrap_attrs,
        world_size=world_size,
        standard_loader=standard,
        offload_requested=offload,
    )

    assert plan.enabled is False
    assert plan.reason


def test_nvfp4_native_streaming_excludes_only_precision_overrides(modules, monkeypatch):
    b = modules.backends
    adapter = b.TorchaoNvfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.FP4,
        native_transformer_streaming=True,
    )
    sentinel = object()
    captured = []

    class Linear:
        in_features = 512
        out_features = 512

    class FakeModel:
        def named_modules(self):
            return [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.attn.q_proj", Linear()),
                ("blocks.0.mlp", Linear()),
                ("blocks.1.attn.out_proj", Linear()),
                ("blocks.1.mlp", Linear()),
                ("input_proj", Linear()),
            ]

        def get_submodule(self, name):
            if name == "blocks":
                return object()
            raise AttributeError(name)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(nn=SimpleNamespace(Linear=Linear)),
    )
    monkeypatch.setattr(
        adapter,
        "_stream_config_factory",
        lambda exclusions: captured.append(list(exclusions)) or sentinel,
    )

    prepared = b.prepare_native_transformer_format_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        precision_prefixes=("0.attn",),
        precision_suffixes=(".out_proj",),
        hybrid=False,
        model_factory=FakeModel,
    )

    assert prepared.quantization_config is sentinel, prepared.descriptor.fallback_reason
    assert captured == [
        [
            "blocks.0.attn.q_proj",
            "blocks.1.attn.out_proj",
            "input_proj",
        ]
    ]
    assert prepared.streamed_targets == ("blocks.0.mlp", "blocks.1.mlp")
    assert prepared.residual_targets == (
        "blocks.0.attn.q_proj",
        "blocks.1.attn.out_proj",
    )


def test_nvfp4_native_streaming_rejects_hybrid_ownership(modules):
    b = modules.backends
    adapter = b.TorchaoNvfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.FP8_FP4,
        native_transformer_streaming=True,
    )

    prepared = b.prepare_native_transformer_format_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        precision_prefixes=(),
        precision_suffixes=(),
        hybrid=True,
        model_factory=lambda: pytest.fail("must not inspect structure"),
    )

    assert prepared.descriptor.materialization_mode == "post_load"
    assert "hybrid" in prepared.descriptor.fallback_reason


def test_mxfp4_never_claims_per_weight_streaming(modules):
    b = modules.backends
    adapter = b.AiterMxfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.AITER,
        format_=modules.contracts.QuantizationFormat.FP4,
        native_unavailable_reason=b.MXFP4_STREAMING_FALLBACK,
    )

    prepared = b.prepare_native_transformer_format_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        model_factory=lambda: pytest.fail("must not inspect structure"),
    )

    assert prepared.quantization_config is None
    assert prepared.descriptor.materialization_mode == "post_load"
    assert "full-precision weight" in prepared.descriptor.fallback_reason


def test_descriptor_declares_storage_sharding_and_trainability(modules):
    b = modules.backends
    adapter = b.AiterMxfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.AITER,
        format_=modules.contracts.QuantizationFormat.FP4,
    )

    descriptor = b.describe_blockwise_format_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        wrap_attrs=("blocks",),
    )

    assert descriptor.storage_semantics == "aiter_mxfp4_per_1x32"
    assert descriptor.parameter_semantics == "packed_weight_parameter"
    assert descriptor.auxiliary_state_semantics == "replicated_scale_buffer"
    assert descriptor.trainability == "inference_only"
    assert descriptor.serialization == "packed_state_supported_not_portable"
    assert descriptor.materialization_mode == "blockwise"
    message = descriptor.log_message()
    assert "backend=aiter" in message
    assert "storage=aiter_mxfp4_per_1x32" in message


@pytest.mark.parametrize("format_name", ["FP4", "INT8"])
def test_fsdp_tensor_subclass_validation_is_format_specific(modules, format_name):
    b = modules.backends
    capabilities = b.FormatBackendCapabilities(
        torchao_nvfp4=True,
        torchao_int8=True,
        torchao_nvfp4_fsdp=False,
        torchao_int8_fsdp=(format_name == "INT8"),
        torchao_nvfp4_fsdp_reason="NVFP4 lacks gather support",
        torchao_int8_fsdp_reason="INT8 lacks gather support",
    )
    adapter_cls = (
        b.TorchaoNvfp4BackendAdapter
        if format_name == "FP4"
        else b.TorchaoInt8BackendAdapter
    )
    adapter = adapter_cls(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=getattr(modules.contracts.QuantizationFormat, format_name),
    )

    if format_name == "FP4":
        with pytest.raises(
            modules.contracts.UnsupportedLoadContract,
            match=r"NVFP4.*gather support",
        ):
            b.validate_format_fsdp_placement(
                _contract(modules, format_name, "TORCHAO", "FSDP_META"),
                adapter,
                capabilities=capabilities,
                required=True,
            )
    else:
        b.validate_format_fsdp_placement(
            _contract(modules, format_name, "TORCHAO", "FSDP_META"),
            adapter,
            capabilities=capabilities,
            required=True,
        )


@pytest.mark.parametrize("fsdp_supported", [False, True])
def test_mxfp4_fsdp_placement_follows_the_packed_parameter_capability(
    modules, fsdp_supported
):
    b = modules.backends
    capabilities = b.FormatBackendCapabilities(
        aiter_mxfp4=True,
        aiter_mxfp4_fsdp=fsdp_supported,
        aiter_mxfp4_fsdp_reason=None if fsdp_supported else "torch is too old",
    )
    adapter = b.AiterMxfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.AITER,
        format_=modules.contracts.QuantizationFormat.FP4,
    )
    contract = _contract(modules, "FP4", "AITER", "FSDP_META")

    if fsdp_supported:
        b.validate_format_fsdp_placement(
            contract, adapter, capabilities=capabilities, required=True
        )
        return

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match=r"AITER MXFP4 packed weight cannot be placed under FSDP2: torch is too old",
    ):
        b.validate_format_fsdp_placement(
            contract, adapter, capabilities=capabilities, required=True
        )


def test_mxfp4_fsdp_placement_is_skipped_when_fsdp_is_not_required(modules):
    b = modules.backends
    capabilities = b.FormatBackendCapabilities(aiter_mxfp4=True, aiter_mxfp4_fsdp=False)
    adapter = b.AiterMxfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.AITER,
        format_=modules.contracts.QuantizationFormat.FP4,
    )

    b.validate_format_fsdp_placement(
        _contract(modules, "FP4", "AITER"),
        adapter,
        capabilities=capabilities,
        required=False,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("self.sharded_param = nn.Parameter(self.to_sharded_dtensor(p))", False),
        (
            "self.sharded_param = nn.Parameter(\n"
            "    self.to_sharded_dtensor(p), requires_grad=param.requires_grad\n)",
            True,
        ),
    ],
)
def test_fsdp_non_float_probe_reads_the_sharded_parameter_call_site(
    modules, monkeypatch, source, expected
):
    b = modules.backends
    fake_inspect = SimpleNamespace(getsource=lambda fn: source)
    fake_param_module = SimpleNamespace(
        FSDPParam=SimpleNamespace(_init_sharded_param=object())
    )
    monkeypatch.setattr(
        b,
        "import_module",
        lambda name: {
            "inspect": fake_inspect,
            "torch.distributed.fsdp._fully_shard._fsdp_param": fake_param_module,
        }[name],
    )

    available, reason = b._probe_fsdp_non_float_parameters()

    assert available is expected
    if expected:
        assert reason is None
    else:
        assert "177948" in reason


@pytest.mark.parametrize(
    ("torch_version", "expected"),
    [("2.9.1+gitff65f5b", False), ("2.12.0", True)],
)
def test_fsdp_non_float_probe_falls_back_to_the_torch_version(
    modules, monkeypatch, torch_version, expected
):
    b = modules.backends

    def import_module(name):
        if name == "torch":
            return SimpleNamespace(__version__=torch_version)
        raise ImportError(name)

    monkeypatch.setattr(b, "import_module", import_module)

    available, reason = b._probe_fsdp_non_float_parameters()

    assert available is expected
    assert (reason is None) is expected


def test_installed_diffusers_accepts_exact_nvfp4_and_int8_configs(modules):
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    b = modules.backends

    nvfp4 = b.TorchaoNvfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.FP4,
        native_transformer_streaming=True,
    )._stream_config_factory([])
    int8 = b.TorchaoInt8BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.INT8,
        native_transformer_streaming=True,
    )._stream_config_factory([])

    assert type(nvfp4.quant_type).__name__ == (
        "NVFP4DynamicActivationNVFP4WeightConfig"
    )
    assert nvfp4.quant_type.use_dynamic_per_tensor_scale is True
    assert nvfp4.quant_type.use_triton_kernel is True
    assert type(int8.quant_type).__name__ == ("Int8DynamicActivationInt8WeightConfig")
    assert int8.quant_type.set_inductor_config is False


@pytest.mark.parametrize("format_name", ["INT8", "FP4"])
def test_native_diffusers_load_preserves_format_targets(
    modules,
    tmp_path,
    format_name,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    if not torch.cuda.is_available():
        pytest.skip("native TorchAO load requires a CUDA accelerator")
    if format_name == "FP4" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("NVFP4 integration requires Blackwell")

    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
    from torchao.utils import TorchAOBaseTensor

    class TinyTransformer(ModelMixin, ConfigMixin):
        @register_to_config
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(512, 512)])
            self.small = torch.nn.Linear(512, 256)
            self.input_proj = torch.nn.Linear(512, 512)

    b = modules.backends
    adapter_cls = (
        b.TorchaoNvfp4BackendAdapter
        if format_name == "FP4"
        else b.TorchaoInt8BackendAdapter
    )
    adapter = adapter_cls(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=getattr(modules.contracts.QuantizationFormat, format_name),
        native_transformer_streaming=True,
    )
    original = TinyTransformer().to(torch.bfloat16)
    original.save_pretrained(tmp_path)
    config = adapter.transformer_stream_config(
        ("blocks",), model_factory=TinyTransformer
    )

    loaded = TinyTransformer.from_pretrained(
        tmp_path,
        torch_dtype=torch.bfloat16,
        quantization_config=config,
        device_map={"": 0},
    )

    assert isinstance(loaded.blocks[0].weight, TorchAOBaseTensor)
    assert not isinstance(loaded.input_proj.weight, TorchAOBaseTensor)
    assert not isinstance(loaded.small.weight, TorchAOBaseTensor)
