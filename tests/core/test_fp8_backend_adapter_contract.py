"""Dependency-light contracts for generalized transformer FP8 load backends."""

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


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    contracts = _load_module(CONTRACTS_PATH, "fp8_adapter_contracts")
    backends = _load_module(BACKENDS_PATH, "fp8_adapter_backends")
    return SimpleNamespace(contracts=contracts, backends=backends)


def _contract(modules, backend):
    return SimpleNamespace(
        requested_format=modules.contracts.QuantizationFormat.FP8,
        selected_backend=backend,
    )


def test_backend_selection_uses_injected_capabilities(modules):
    c, b = modules.contracts, modules.backends
    capabilities = b.Fp8BackendCapabilities(
        aiter_block_scale=True,
        torchao_fp8=False,
    )

    selected = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.AITER),
        capabilities=capabilities,
    )

    assert selected.backend is c.QuantizationBackend.AITER
    assert selected.format is c.QuantizationFormat.FP8
    assert selected.storage_semantics == "block_128_scaled"


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
    assert capabilities == b.Fp8BackendCapabilities(
        aiter_block_scale=False,
        torchao_fp8=True,
        torchao_diffusers_streaming=False,
        torchao_text_encoder_streaming=False,
        torchao_fsdp_patches=True,
        aiter_transformers_reason="AITER FP8 backend is unavailable",
        torchao_text_encoder_reason="Transformers TorchAO unavailable",
    )


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
    assert capabilities.torchao_fp8 is True


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

    assert capabilities.torchao_fp8 is False
    assert "CUDA or HIP/ROCm" in capabilities.torchao_fp8_reason


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

    assert capabilities.torchao_fp8 is False
    assert (
        capabilities.torchao_fp8_reason
        == "torchao 0.14.0 is older than required 0.15.0"
    )


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


def test_unavailable_selected_backend_fails_without_format_change(modules):
    c, b = modules.contracts, modules.backends

    with pytest.raises(
        c.UnsupportedLoadContract,
        match=r"TORCHAO.*FP8.*unavailable",
    ):
        b.select_fp8_backend(
            _contract(modules, c.QuantizationBackend.TORCHAO),
            capabilities=b.Fp8BackendCapabilities(
                aiter_block_scale=True,
                torchao_fp8=False,
            ),
        )


def test_hybrid_contract_projects_to_torchao_for_blockwise_fp8(modules):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP8_FP4,
        selected_backend=c.QuantizationBackend.TORCHAO,
    )

    adapter = b.select_blockwise_fp8_backend(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=True,
        ),
    )

    assert adapter.backend is c.QuantizationBackend.TORCHAO
    assert adapter.format is c.QuantizationFormat.FP8
    assert adapter.storage_semantics == "tensorwise_dynamic"


def test_pure_fp4_contract_projects_for_fp8_only_blockwise_component(modules):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP4,
        selected_backend=c.QuantizationBackend.TORCHAO,
    )

    adapter = b.select_blockwise_fp8_backend(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=True,
        ),
    )

    assert adapter.backend is c.QuantizationBackend.TORCHAO
    assert adapter.format is c.QuantizationFormat.FP8


@pytest.mark.parametrize(
    (
        "platform",
        "fp4_backend",
        "aiter_fp8",
        "torchao_fp8",
        "expected_fp8_backend",
    ),
    [
        ("cuda", "TORCHAO", False, True, "TORCHAO"),
        ("rdna4_rocm", "AITER", True, True, "AITER"),
        ("other_rocm", "AITER", False, True, "TORCHAO"),
    ],
)
def test_fp4_contract_selects_fp8_backend_from_hardware_matrix(
    modules,
    platform,
    fp4_backend,
    aiter_fp8,
    torchao_fp8,
    expected_fp8_backend,
):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP4,
        selected_backend=getattr(c.QuantizationBackend, fp4_backend),
    )

    adapter = b.select_blockwise_fp8_backend(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            aiter_block_scale=aiter_fp8,
            torchao_fp8=torchao_fp8,
            torchao_fsdp_patches=torchao_fp8,
        ),
    )

    assert adapter.backend is getattr(
        c.QuantizationBackend, expected_fp8_backend
    ), platform


@pytest.mark.parametrize(
    "materialization_mode",
    ["EAGER", "REPLICATED_META"],
)
def test_non_fsdp_torchao_allows_missing_fsdp_tensor_patches(
    modules,
    materialization_mode,
):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP4,
        selected_backend=c.QuantizationBackend.TORCHAO,
        materialization_mode=getattr(c.MaterializationMode, materialization_mode),
    )

    adapter = b.select_blockwise_fp8_backend(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="FSDP patches unavailable",
        ),
    )
    b.validate_torchao_fsdp2_patches(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="FSDP patches unavailable",
        ),
        required=False,
    )

    assert adapter.backend is c.QuantizationBackend.TORCHAO


def test_fsdp_torchao_rejects_missing_fsdp_tensor_patches(modules):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP4,
        selected_backend=c.QuantizationBackend.TORCHAO,
        materialization_mode=c.MaterializationMode.FSDP_META,
    )

    with pytest.raises(
        c.UnsupportedLoadContract,
        match=r"FSDP.*patch.*fsdp_post_all_gather",
    ):
        b.validate_torchao_fsdp2_patches(
            contract,
            capabilities=b.Fp8BackendCapabilities(
                torchao_fp8=True,
                torchao_fsdp_patches=False,
                torchao_fsdp_reason=(
                    "missing TorchAO FSDP patches: fsdp_post_all_gather"
                ),
            ),
            required=True,
        )


def test_blockwise_aiter_ignores_torchao_fsdp_patch_state(modules):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP4,
        selected_backend=c.QuantizationBackend.AITER,
    )

    adapter = b.select_blockwise_fp8_backend(
        contract,
        capabilities=b.Fp8BackendCapabilities(
            aiter_block_scale=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="TorchAO patches unavailable",
        ),
    )

    assert adapter.backend is c.QuantizationBackend.AITER


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

    exclusions = b.derive_untargeted_linear_exclusions(
        FakeModel(),
        ("blocks",),
        is_linear=lambda module: module == "linear",
    )

    assert exclusions == ["blocks_extra.proj", "input_proj"]


def test_missing_target_makes_native_mapping_unavailable(modules):
    b = modules.backends

    class FakeModel:
        def named_modules(self):
            return [("", object()), ("input_proj", "linear")]

        def get_submodule(self, name):
            raise AttributeError(name)

    with pytest.raises(b.TargetMappingUnavailable, match="missing"):
        b.derive_untargeted_linear_exclusions(
            FakeModel(),
            ("missing",),
            is_linear=lambda module: module == "linear",
        )


def test_torchao_native_config_uses_structure_derived_exclusions(modules, monkeypatch):
    c, b = modules.contracts, modules.backends
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_diffusers_streaming=True,
        ),
    )
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
        b,
        "derive_untargeted_linear_exclusions",
        lambda model, targets: ["input_proj"],
    )

    prepared = b.prepare_native_transformer_fp8_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        stream_quant=True,
        model_factory=FakeModel,
    )

    assert prepared.quantization_config is sentinel
    assert captured == [["input_proj"]]
    assert prepared.descriptor.materialization_mode == "streaming"
    assert prepared.descriptor.storage_semantics == "tensorwise_dynamic"
    assert prepared.descriptor.fallback_reason is None


def test_torchao_without_native_diffusers_api_is_explicit_fallback(modules):
    c, b = modules.contracts, modules.backends
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(torchao_fp8=True),
    )

    prepared = b.prepare_native_transformer_fp8_load(
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
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(torchao_fp8=True),
    )

    prepared = b.prepare_native_transformer_fp8_load(
        adapter,
        component_name="transformer_2",
        targets=(),
        stream_quant=True,
        model_factory=lambda: pytest.fail("must not inspect model"),
    )

    assert prepared.descriptor.materialization_mode == "post_load"
    assert "no FP8 targets" in prepared.descriptor.fallback_reason


def test_blockwise_paths_keep_streaming_through_backend_adapter(modules):
    c, b = modules.contracts, modules.backends
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(torchao_fp8=True),
    )

    descriptor = b.plan_blockwise_transformer_fp8_load(
        adapter,
        component_name="transformer",
        targets=("blocks",),
        wrap_attrs=("blocks",),
    )

    assert descriptor.materialization_mode == "streaming"
    assert descriptor.fallback_reason is None


def test_aiter_keeps_native_quantize_on_load_config(modules, monkeypatch):
    c, b = modules.contracts, modules.backends
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.AITER),
        capabilities=b.Fp8BackendCapabilities(aiter_block_scale=True),
    )
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
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_diffusers_streaming=True,
        ),
    )

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
    adapter = b.select_fp8_backend(
        _contract(modules, c.QuantizationBackend.TORCHAO),
        capabilities=b.Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_diffusers_streaming=True,
        ),
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
    assert loaded.input_proj.weight.dtype is torch.bfloat16
