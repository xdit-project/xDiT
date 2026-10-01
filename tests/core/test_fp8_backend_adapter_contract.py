"""Dependency-light contracts for generalized transformer FP8 load backends."""

import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKENDS_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/fp8_backends.py"
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
_PKG = "fp8_adapter_pkg"


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
    backends = _load_module(BACKENDS_PATH, "fp8_adapter_backends")
    return SimpleNamespace(contracts=contracts, backends=backends, adapter=adapter)


def _adapter(modules, impl, **measured):
    """The adapter one implementation gives, from a measured record."""
    a = modules.adapter
    return a.build_adapter(
        "fp8", impl, capability=a.FormatCapability(available=True, **measured)
    )


def test_an_adapter_is_built_from_its_measured_record(modules):
    c, a = modules.contracts, modules.adapter

    selected = _adapter(modules, "aiter")

    assert selected.backend is c.QuantizationBackend.AITER
    assert selected.format is c.QuantizationFormat.FP8
    assert selected.impl == "aiter"
    assert selected.format_name == "fp8"
    assert selected.storage_semantics == "block_128_scaled"


def test_an_unavailable_record_refuses_with_what_the_probe_measured(modules):
    a = modules.adapter

    with pytest.raises(modules.contracts.UnsupportedLoadContract, match="no GPU"):
        a.build_adapter(
            "fp8",
            "torchao",
            capability=a.FormatCapability(available=False, reason="no GPU"),
        )


def test_a_pair_with_no_registered_class_is_refused(modules):
    a = modules.adapter

    with pytest.raises(modules.contracts.UnsupportedLoadContract, match="no aiter"):
        a.build_adapter(
            "int8", "aiter", capability=a.FormatCapability(available=True)
        )


def test_hardware_and_package_probes_are_injectable(modules):
    b = modules.backends
    calls = []

    capabilities = b.probe_fp8_backend_capabilities(
        aiter_probe=lambda: calls.append("aiter") or False,
        torchao_accelerator_probe=lambda: calls.append("accelerator") or True,
        torchao_probe=lambda: calls.append("torchao") or True,
        torchao_diffusers_probe=lambda: calls.append("diffusers") or False,
        torchao_text_encoder_probe=lambda: calls.append("text_encoder")
        or (
            False,
            "Transformers TorchAO unavailable",
        ),
        torchao_fsdp_probe=lambda: calls.append("fsdp") or True,
    )

    assert calls == [
        "aiter",
        "accelerator",
        "torchao",
        "diffusers",
        "text_encoder",
        "fsdp",
    ]
    assert capabilities.of("fp8", "aiter").available is False
    torchao = capabilities.of("fp8", "torchao")
    assert torchao.available is True
    assert torchao.streams is False
    assert torchao.te_streams is False
    assert torchao.te_streams_reason == "Transformers TorchAO unavailable"
    assert torchao.fsdp_safe is True


def test_text_encoder_probe_does_not_require_diffusers_transformer_quantizer(
    modules, monkeypatch
):
    b = modules.backends
    monkeypatch.setattr(
        b,
        "_probe_torchao_diffusers_streaming",
        lambda: pytest.fail(
            "text-encoder routing must probe PipelineQuantizationConfig directly"
        ),
    )
    monkeypatch.setattr(b, "_probe_torchao_fp8_conversion_api", lambda: (True, None))

    def missing_transformers(name):
        raise ImportError(f"isolated missing API: {name}")

    monkeypatch.setattr(b, "import_module", missing_transformers)

    available, reason = b._probe_torchao_text_encoder_streaming()

    assert not available
    assert "isolated missing API" in reason


@pytest.mark.parametrize(
    ("methods", "streams"),
    [
        # Transformers 5's op-based surface
        (("get_quantize_ops", "param_needs_quantization"), True),
        # What Transformers 4 and Diffusers carry
        (("create_quantized_param", "check_if_quantized_param"), True),
        # Transformers 4.57 has neither pair whole, and must fall back
        (("create_quantized_param", "param_needs_quantization"), False),
        ((), False),
    ],
)
def test_either_parameter_quantization_surface_counts_as_streaming(
    modules, methods, streams
):
    """Requiring only the older pair meant no installed Transformers ever matched.

    Every text encoder then took the post-load fallback, silently, which is the
    memory saving the flag exists for not happening.
    """
    quantizer = type("Quantizer", (), {name: lambda self: None for name in methods})

    assert modules.backends._quantizes_parameter_by_parameter(quantizer) is streams


def test_supported_rocm_runs_torchao_api_preflight(modules):
    b = modules.backends
    calls = []

    capabilities = b.probe_fp8_backend_capabilities(
        aiter_probe=lambda: False,
        torchao_accelerator_probe=lambda: True,
        torchao_probe=lambda: calls.append("torchao") or True,
        torchao_diffusers_probe=lambda: False,
        torchao_fsdp_probe=lambda: True,
    )

    assert calls == ["torchao"]
    assert capabilities.of("fp8", "torchao").available is True


def test_cuda_fp8_requires_capability_89(modules):
    available, reason = modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (8, 6),
    )

    assert available is False
    assert "8.9" in reason


def test_cuda_fp8_accepts_capability_89(modules):
    assert modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (8, 9),
    ) == (True, None)


def test_rocm_fp8_eligibility_does_not_query_cuda_capability(modules):
    assert modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: False,
        hip_probe=lambda: True,
        cuda_capability_probe=lambda: pytest.fail("CUDA capability probed on ROCm"),
    ) == (True, None)


def test_unsupported_accelerator_skips_torchao_import_probe(modules):
    b = modules.backends

    capabilities = b.probe_fp8_backend_capabilities(
        aiter_probe=lambda: False,
        torchao_accelerator_probe=lambda: False,
        torchao_probe=lambda: pytest.fail("must not import TorchAO"),
        torchao_diffusers_probe=lambda: pytest.fail("must not probe Diffusers"),
        torchao_fsdp_probe=lambda: pytest.fail("must not inspect patches"),
    )

    torchao = capabilities.of("fp8", "torchao")
    assert torchao.available is False
    assert "CUDA or HIP/ROCm" in torchao.reason


def test_torchao_preflight_preserves_exact_unavailability_reason(modules):
    b = modules.backends

    capabilities = b.probe_fp8_backend_capabilities(
        aiter_probe=lambda: False,
        torchao_accelerator_probe=lambda: True,
        torchao_probe=lambda: (
            False,
            "torchao 0.14.0 is older than required 0.15.0",
        ),
        torchao_diffusers_probe=lambda: (
            False,
            "Diffusers TorchAoConfig unavailable",
        ),
        torchao_fsdp_probe=lambda: pytest.fail(
            "FSDP probe must not run when TorchAO is unavailable"
        ),
    )

    torchao = capabilities.of("fp8", "torchao")
    assert torchao.available is False
    assert torchao.reason == "torchao 0.14.0 is older than required 0.15.0"


def test_installed_torchao_conversion_api_preflight(modules):
    pytest.importorskip("torchao")
    available, reason = modules.backends._probe_torchao_fp8_conversion_api()

    assert available, reason
    assert reason is None


def test_installed_torchao_fsdp_patch_preflight(modules):
    pytest.importorskip("torchao")
    available, reason = modules.backends._probe_torchao_fsdp_patches()

    assert available, reason
    assert reason is None


@pytest.mark.parametrize(
    "materialization_mode",
    ["EAGER", "REPLICATED_META"],
)
def test_unshardable_storage_is_fine_where_nothing_shards_it(
    modules,
    materialization_mode,
):
    a = modules.adapter
    capability = a.FormatCapability(
        available=True,
        fsdp_safe=False,
        fsdp_reason="FSDP patches unavailable",
    )
    adapter = a.build_adapter("fp8", "torchao", capability=capability)

    a.validate_fsdp_placement(adapter, capability=capability, required=False)

    assert adapter.backend is modules.contracts.QuantizationBackend.TORCHAO


def test_unshardable_storage_under_fsdp_is_refused_with_the_measured_reason(
    modules,
):
    a = modules.adapter
    capability = a.FormatCapability(
        available=True,
        fsdp_safe=False,
        fsdp_reason="missing TorchAO FSDP patches: fsdp_post_all_gather",
    )
    adapter = a.build_adapter("fp8", "torchao", capability=capability)

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match=r"FSDP2.*fsdp_post_all_gather",
    ):
        a.validate_fsdp_placement(adapter, capability=capability, required=True)


def test_a_shardable_parameter_kind_passes_the_same_check(modules):
    """One rule, and AITER's plain packed parameter satisfies it."""
    a = modules.adapter
    capability = a.FormatCapability(available=True, fsdp_safe=True)
    adapter = a.build_adapter("fp8", "aiter", capability=capability)

    a.validate_fsdp_placement(adapter, capability=capability, required=True)

    assert adapter.parameter_semantics == "packed_weight_parameter"


def test_derive_exclusions_preserves_only_declared_target_prefixes(modules):
    b = modules.backends

    class FakeModel:
        def named_modules(self):
            return [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.proj", "linear"),
                ("blocks_extra.proj", "linear"),
                ("input_proj", "linear"),
                ("norm", object()),
            ]

        def get_submodule(self, name):
            if name == "blocks":
                return object()
            raise AttributeError(name)

    ownership = modules.adapter.derive_linear_ownership(
        FakeModel(),
        ("blocks",),
        is_linear=lambda module: module == "linear",
    )

    assert ownership.exclusions == ("blocks_extra.proj", "input_proj")
    assert ownership.streamed == ("blocks.0.proj",)


def test_missing_target_makes_native_mapping_unavailable(modules):
    b = modules.backends

    class FakeModel:
        def named_modules(self):
            return [("", object()), ("input_proj", "linear")]

        def get_submodule(self, name):
            raise AttributeError(name)

    with pytest.raises(b.TargetMappingUnavailable, match="missing"):
        modules.adapter.derive_linear_ownership(
            FakeModel(),
            ("missing",),
            is_linear=lambda module: module == "linear",
        )


def test_torchao_native_config_uses_structure_derived_exclusions(modules, monkeypatch):
    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao", streams=True)
    sentinel = object()
    captured = []

    class FakeModel:
        def named_modules(self):
            return [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.proj", "linear"),
                ("input_proj", "linear"),
            ]

        def get_submodule(self, name):
            if name == "blocks":
                return object()
            raise AttributeError(name)

    monkeypatch.setattr(
        adapter,
        "_stream_config_factory",
        lambda exclusions: captured.append(exclusions) or sentinel,
    )
    monkeypatch.setattr(
        modules.adapter,
        "derive_linear_ownership",
        lambda model, targets, **kwargs: modules.adapter.LinearOwnership(
            exclusions=("input_proj",), streamed=("blocks.0.proj",)
        ),
    )

    prepared = modules.adapter.prepare_native_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        model_factory=FakeModel,
    )

    assert prepared.quantization_config is sentinel
    assert captured == [("input_proj",)]
    assert prepared.descriptor.materialization_mode == "streaming"
    assert prepared.descriptor.storage_semantics == "tensorwise_dynamic"
    assert prepared.descriptor.fallback_reason is None


def test_torchao_without_native_diffusers_api_is_explicit_fallback(modules):
    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao")

    prepared = modules.adapter.prepare_native_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        model_factory=lambda: pytest.fail("must not inspect model"),
    )

    assert prepared.quantization_config is None
    assert prepared.descriptor.materialization_mode == "post_load"
    assert "Diffusers TorchAoConfig API" in prepared.descriptor.fallback_reason
    assert "torchao" in prepared.descriptor.log_message()
    assert "post_load" in prepared.descriptor.log_message()


def test_untargeted_transformer_does_not_claim_streaming(modules):
    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao")

    prepared = modules.adapter.prepare_native_load(
        adapter,
        component_name="transformer_2",
        targets=(),
        stream_quant=True,
        model_factory=lambda: pytest.fail("must not inspect model"),
    )

    assert prepared.descriptor.materialization_mode == "post_load"
    assert "no FP8 targets" in prepared.descriptor.fallback_reason


def test_blockwise_paths_quantize_on_the_way_in(modules):
    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao")

    descriptor = modules.adapter.describe_blockwise_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        wrap_attrs=("blocks",),
    )

    # Per block on the way in from disk, which the descriptor names as its own
    # mode rather than borrowing the per-weight one.
    assert descriptor.materialization_mode == "blockwise"
    assert descriptor.fallback_reason is None


def test_aiter_keeps_native_quantize_on_load_config(modules, monkeypatch):
    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "aiter", streams=True)
    sentinel = object()
    monkeypatch.setattr(
        adapter,
        "_stream_config_factory",
        lambda targets: (sentinel, tuple(targets)),
    )

    assert adapter.transformer_stream_config(("blocks",)) == (
        sentinel,
        ("blocks",),
    )


def test_installed_native_config_matches_existing_torchao_fp8_semantics(
    modules,
):
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    from torchao.quantization.granularity import PerTensor

    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao", streams=True)

    config = adapter._stream_config_factory(["input_proj"])
    quant_type = config.quant_type

    assert config.modules_to_not_convert == ["input_proj"]
    assert quant_type.set_inductor_config is False
    assert all(isinstance(value, PerTensor) for value in quant_type.granularity)


def test_native_diffusers_load_quantizes_only_targeted_linears(modules, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    if not torch.cuda.is_available():
        pytest.skip("TorchAO float8 runtime needs a supported accelerator")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        pytest.skip("TorchAO float8 runtime needs CUDA capability >= 8.9")

    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
    from torchao.utils import TorchAOBaseTensor

    # TorchAO silently leaves a linear in bf16 unless both dimensions are a
    # multiple of 16, which _scaled_mm requires.
    class TinyTransformer(ModelMixin, ConfigMixin):
        @register_to_config
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(32, 32)])
            self.input_proj = torch.nn.Linear(32, 32)

    c, b = modules.contracts, modules.backends
    adapter = _adapter(modules, "torchao", streams=True)
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
    assert loaded.input_proj.weight.dtype is torch.bfloat16
