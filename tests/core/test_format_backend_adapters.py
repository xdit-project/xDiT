"""Dependency-light contracts for FP4 and INT8 materialization backends."""

import importlib.machinery
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
ADAPTER_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/quant_adapter.py"
)


#: A synthetic package over the loading directory. These modules are meant to
#: be importable without the rest of the app, and they share a dependency-light
#: base, so the loader gives relative imports somewhere to resolve rather than
#: forcing the shared code to be duplicated or imported absolutely.
_PKG = "format_adapter_pkg"


def _package():
    if _PKG not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(_PKG, None, is_package=True)
        )
        package.__path__ = [str(BACKENDS_PATH.parent)]
        sys.modules[_PKG] = package
    return sys.modules[_PKG]


def _load_module(path, name):
    _package()
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{_PKG}.{name}"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    contracts = _load_module(CONTRACTS_PATH, "contracts")
    adapter = _load_module(ADAPTER_PATH, "quant_adapter")
    backends = _load_module(BACKENDS_PATH, "format_adapter_backends")
    return SimpleNamespace(contracts=contracts, backends=backends, adapter=adapter)


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

    fmt, impl = format_name.lower(), backend_name.lower()
    record = capabilities.of(fmt, impl)
    assert record.available, record.reason
    adapter = modules.adapter.build_adapter(fmt, impl, capability=record)

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
        match=r"capability.*10.0",
    ):
        modules.adapter.build_adapter(
            "fp4", "torchao", capability=capabilities.of("fp4", "torchao")
        )


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

    adapter = modules.adapter.build_adapter(
        "fp4", "aiter", capability=capabilities.of("fp4", "aiter"), hybrid=True
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
    record = capabilities.of("fp4", "aiter")
    assert record.available is False
    assert record.reason == "missing required AITER MXFP4 API: aiter.gemm_a4w4"


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

    descriptor = modules.adapter.describe_blockwise_load(
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

    prepared = modules.adapter.prepare_native_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        residual_match=lambda name: (
            name.startswith("blocks.0.attn") or name.endswith(".out_proj")
        ),
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
        format_=modules.contracts.QuantizationFormat.FP4,
        native_transformer_streaming=True,
    )

    prepared = modules.adapter.prepare_native_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
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

    prepared = modules.adapter.prepare_native_load(
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

    descriptor = modules.adapter.describe_blockwise_load(
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


@pytest.mark.parametrize(
    ("format_name", "impl", "shardable"),
    [
        ("fp4", "torchao", False),
        ("int8", "torchao", True),
        ("fp4", "aiter", False),
        ("fp6", "aiter", True),
    ],
)
def test_fsdp_placement_follows_the_record_for_every_pair(
    modules, format_name, impl, shardable
):
    """One rule, no ladder: the record says, the adapter names what it stores."""
    a = modules.adapter
    capability = a.FormatCapability(
        available=True,
        fsdp_safe=shardable,
        fsdp_reason=None if shardable else "torch is too old",
    )
    adapter = a.build_adapter(format_name, impl, capability=capability)

    if shardable:
        a.validate_fsdp_placement(adapter, capability=capability, required=True)
        return

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match=rf"{adapter.parameter_semantics}.*{impl} {format_name}.*"
        r"FSDP2: torch is too old",
    ):
        a.validate_fsdp_placement(adapter, capability=capability, required=True)


def test_fsdp_placement_is_skipped_when_nothing_shards_it(modules):
    a = modules.adapter
    capability = a.FormatCapability(available=True, fsdp_safe=False)
    adapter = a.build_adapter("fp4", "aiter", capability=capability)

    a.validate_fsdp_placement(adapter, capability=capability, required=False)


@pytest.mark.parametrize(
    ("format_name", "impl"),
    [
        ("fp4", "aiter"),
        ("fp6", "aiter"),
        ("fp4", "torchao"),
        ("int8", "torchao"),
    ],
)
def test_every_implementation_can_be_either_half_of_a_per_step_pair(
    modules, format_name, impl
):
    """Composing is the base class's job, so no format is excluded by age.

    The pairing used to live inside the MXFP4 factory, which made MXFP4 the
    only possible low side -- not because another kernel could not hold two
    precisions at a leaf, but because no other factory was ever handed the
    companion.
    """
    a = modules.adapter
    capability = a.FormatCapability(available=True)

    adapter = a.build_adapter(format_name, impl, capability=capability)
    assert adapter.builds_one_layer()

    # Accepted as the low side rather than refused.
    a.build_adapter(format_name, impl, capability=capability, hybrid=True)


def test_an_implementation_with_no_single_leaf_seam_is_still_refused(modules):
    """The refusal remains for a future adapter that can only convert a tree."""
    a = modules.adapter

    class TreeOnly(a.QuantAdapter):
        pass

    assert not TreeOnly.builds_one_layer()
    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match="cannot install one layer at a time",
    ):
        TreeOnly(backend=None, format_=None)._single_layer_factory(device=None)


def test_a_companion_reaches_the_converter_rather_than_being_dropped(modules):
    """A dropped companion would silently lose the run's second precision."""
    a, b = modules.adapter, modules.backends
    adapter = b.TorchaoNvfp4BackendAdapter(
        backend=modules.contracts.QuantizationBackend.TORCHAO,
        format_=modules.contracts.QuantizationFormat.FP4,
    )
    seen = []
    adapter._single_layer_factory = lambda *, device: (
        lambda spec: seen.append("low")
    )

    paired = adapter.layer_factory(
        device=None, companion=lambda spec: seen.append("high")
    )
    paired(object())

    # Low first, as it was when only MXFP4 composed.
    assert seen == ["low", "high"]
