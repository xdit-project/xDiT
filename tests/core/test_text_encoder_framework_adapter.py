"""Dependency-light contracts for text-encoder framework integration."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/text_encoder_adapter.py"
)
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
    return SimpleNamespace(
        adapter=_load_module(ADAPTER_PATH, "te_framework_adapter"),
        backends=_load_module(BACKENDS_PATH, "te_framework_backends"),
        contracts=_load_module(CONTRACTS_PATH, "te_framework_contracts"),
    )


def _backend(modules, name, **capabilities):
    c, b = modules.contracts, modules.backends
    contract = SimpleNamespace(
        requested_format=c.QuantizationFormat.FP8,
        selected_backend=getattr(c.QuantizationBackend, name),
    )
    defaults = {
        "aiter_block_scale": name == "AITER",
        "torchao_fp8": name == "TORCHAO",
    }
    defaults.update(capabilities)
    return b.select_fp8_backend(
        contract, capabilities=b.Fp8BackendCapabilities(**defaults)
    )


def test_import_has_no_diffusers_or_transformers_dependency(monkeypatch):
    blocked = {"diffusers", "transformers", "torchao"}
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"eager framework import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    _load_module(ADAPTER_PATH, "te_adapter_import_probe")


def test_transformers_5_probe_is_lazy_and_actionable(modules):
    a = modules.adapter

    supported = a.probe_transformers_streaming_loader(find_spec=lambda name: object())
    unsupported = a.probe_transformers_streaming_loader(find_spec=lambda name: None)

    assert supported.available
    assert supported.reason is None
    assert not unsupported.available
    assert "transformers>=5.0" in unsupported.reason
    assert "core_model_loading" in unsupported.reason


def test_torchao_pipeline_mapping_uses_exact_exclusions_for_multiple_encoders(
    modules,
):
    a = modules.adapter
    calls = []
    framework = a.TextEncoderFrameworkAdapter(
        pipeline_config_factory=lambda mapping: SimpleNamespace(quant_mapping=mapping),
        torchao_config_factory=lambda exclusions: calls.append(exclusions)
        or ("torchao", tuple(exclusions)),
    )

    config = framework.pipeline_quantization_config(
        {
            "text_encoder": framework.component_quantization_config(
                backend="torchao",
                targets=("encoder.block",),
                exclusions=("shared", "lm_head"),
            ),
            "text_encoder_2": framework.component_quantization_config(
                backend="torchao",
                targets=("model.layers",),
                exclusions=("embed_tokens", "lm_head"),
            ),
        }
    )

    assert calls == [
        ["shared", "lm_head"],
        ["embed_tokens", "lm_head"],
    ]
    assert config.quant_mapping == {
        "text_encoder": ("torchao", ("shared", "lm_head")),
        "text_encoder_2": ("torchao", ("embed_tokens", "lm_head")),
    }


def test_aiter_transformers_5_mapping_preserves_targets(modules):
    a = modules.adapter
    framework = a.TextEncoderFrameworkAdapter(
        pipeline_config_factory=lambda mapping: SimpleNamespace(quant_mapping=mapping),
        aiter_config_factory=lambda targets: ("aiter", tuple(targets)),
    )

    config = framework.pipeline_quantization_config(
        {
            "text_encoder": framework.component_quantization_config(
                backend="aiter",
                targets=("model.language_model.layers",),
            )
        }
    )

    assert config.quant_mapping == {
        "text_encoder": (
            "aiter",
            ("model.language_model.layers",),
        )
    }


def test_combined_pipeline_mapping_preserves_transformer_config(modules):
    a = modules.adapter
    transformer = object()
    text_encoder = object()
    existing = SimpleNamespace(quant_mapping={"transformer": transformer})
    framework = a.TextEncoderFrameworkAdapter(
        pipeline_config_factory=lambda mapping: SimpleNamespace(quant_mapping=mapping)
    )

    combined = framework.pipeline_quantization_config(
        {"text_encoder": text_encoder},
        existing=existing,
    )

    assert combined.quant_mapping == {
        "transformer": transformer,
        "text_encoder": text_encoder,
    }
    assert existing.quant_mapping == {"transformer": transformer}


def test_combined_mapping_rejects_component_overwrite(modules):
    a = modules.adapter
    framework = a.TextEncoderFrameworkAdapter(
        pipeline_config_factory=lambda mapping: mapping
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        framework.pipeline_quantization_config(
            {"transformer": object()},
            existing=SimpleNamespace(quant_mapping={"transformer": object()}),
        )


def test_torchao_te_native_plan_derives_safe_negative_mapping(modules, monkeypatch):
    b = modules.backends
    adapter = _backend(
        modules,
        "TORCHAO",
        torchao_diffusers_streaming=True,
        torchao_text_encoder_streaming=True,
    )
    sentinel = object()
    captured = []

    monkeypatch.setattr(
        b,
        "derive_untargeted_linear_exclusions",
        lambda model, targets: ["shared", "lm_head"],
    )

    prepared = b.prepare_text_encoder_fp8_load(
        adapter,
        component_name="text_encoder",
        targets=("encoder.block",),
        model_factory=lambda: object(),
        framework_config_factory=lambda backend, targets, exclusions: (
            captured.append((backend, targets, exclusions)) or sentinel
        ),
    )

    assert prepared.quantization_config is sentinel
    assert captured == [
        (
            "torchao",
            ("encoder.block",),
            ("shared", "lm_head"),
        )
    ]
    assert prepared.descriptor.materialization_mode == "streaming"


def test_transformers_4_aiter_falls_back_to_post_load(modules):
    b = modules.backends
    adapter = _backend(
        modules,
        "AITER",
        aiter_transformers_streaming=False,
        aiter_transformers_reason=("transformers>=5.0 streaming loader is unavailable"),
    )

    prepared = b.prepare_text_encoder_fp8_load(
        adapter,
        component_name="text_encoder",
        targets=("encoder.block",),
        model_factory=lambda: pytest.fail("must not inspect structure"),
    )

    assert prepared.quantization_config is None
    assert prepared.descriptor.materialization_mode == "post_load"
    assert "transformers>=5.0" in prepared.descriptor.fallback_reason
    assert "text_encoder" in prepared.descriptor.log_message()


def test_transformers_4_raises_before_allocation_without_post_load(modules):
    b = modules.backends
    adapter = _backend(
        modules,
        "AITER",
        aiter_transformers_streaming=False,
        aiter_transformers_reason="transformers>=5.0 is required",
    )
    with pytest.raises(RuntimeError, match="before allocation"):
        b.prepare_text_encoder_fp8_load(
            adapter,
            component_name="text_encoder",
            targets=("encoder.block",),
            supports_post_load=False,
            model_factory=lambda: pytest.fail("must not allocate"),
        )


def test_missing_te_target_never_quantizes_all_linears(modules, monkeypatch):
    b = modules.backends
    adapter = _backend(
        modules,
        "TORCHAO",
        torchao_text_encoder_streaming=True,
    )

    def unavailable(model, targets):
        raise b.TargetMappingUnavailable("target mapping unavailable: missing")

    monkeypatch.setattr(b, "derive_untargeted_linear_exclusions", unavailable)
    prepared = b.prepare_text_encoder_fp8_load(
        adapter,
        component_name="text_encoder",
        targets=("missing",),
        model_factory=lambda: object(),
    )

    assert prepared.quantization_config is None
    assert prepared.descriptor.materialization_mode == "post_load"
    assert "target mapping unavailable" in prepared.descriptor.fallback_reason


def test_installed_torchao_te_pipeline_uses_transformers_config(modules):
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("torchao")
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    framework = modules.adapter.TextEncoderFrameworkAdapter()
    te_config = framework.component_quantization_config(
        backend="torchao",
        targets=("encoder.block",),
        exclusions=("shared", "lm_head"),
    )
    pipeline_config = framework.pipeline_quantization_config(
        {
            "text_encoder": te_config,
            "text_encoder_2": framework.component_quantization_config(
                backend="torchao",
                targets=("model.layers",),
                exclusions=("embed_tokens", "lm_head"),
            ),
        }
    )

    assert isinstance(te_config, TransformersTorchAoConfig)
    assert te_config.modules_to_not_convert == ["shared", "lm_head"]
    assert set(pipeline_config.quant_mapping) == {
        "text_encoder",
        "text_encoder_2",
    }
    assert (
        pipeline_config._resolve_quant_config(
            is_diffusers=False, module_name="text_encoder"
        )
        is te_config
    )


def test_installed_pipeline_accepts_combined_transformer_and_te_mapping(
    modules,
):
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("torchao")
    from diffusers import TorchAoConfig as DiffusersTorchAoConfig

    framework = modules.adapter.TextEncoderFrameworkAdapter()
    transformer_config = DiffusersTorchAoConfig(
        modules.adapter._torchao_quant_type(),
        modules_to_not_convert=["input_proj"],
    )
    existing = framework.pipeline_quantization_config(
        {"transformer": transformer_config}
    )
    te_config = framework.component_quantization_config(
        backend="torchao",
        targets=("encoder.block",),
        exclusions=("shared",),
    )

    combined = framework.pipeline_quantization_config(
        {"text_encoder": te_config},
        existing=existing,
    )

    assert (
        combined._resolve_quant_config(is_diffusers=True, module_name="transformer")
        is transformer_config
    )
    assert (
        combined._resolve_quant_config(is_diffusers=False, module_name="text_encoder")
        is te_config
    )
