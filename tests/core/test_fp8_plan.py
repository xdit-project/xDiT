"""Unit tests for Fp8Plan, which decides what a run quantizes to FP8.

Every FP8 consumer (the post-load walks on any hardware, the per-block FSDP quantize, the streaming
quantize-on-load, the meta-init paths) reads its target list from here, so the
--use_fp8_text_encoder opt-in and the prefix matching are pinned here rather than left to a GPU run
to discover.

Run with:
    pytest tests/core/test_fp8_plan.py -v
"""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.loading import fp8_plan
from xfuser.model_executor.models.runner_models.loading.fp8_plan import Fp8Plan


def make_plan(
    monkeypatch,
    *,
    transformer_targets=None,
    te_targets=None,
    use_fp8_gemms=True,
    use_fp8_text_encoder=False,
    on_rdna4=True,
):
    """An Fp8Plan over a stand-in model: the plan only reads settings and config."""
    monkeypatch.setattr(fp8_plan, "_use_aiter_fp8_rdna4", lambda: on_rdna4)
    model = SimpleNamespace(
        settings=SimpleNamespace(
            fp8_gemm_module_list=transformer_targets,
            fp8_text_encoder_module_list=te_targets,
        ),
        config=SimpleNamespace(
            use_fp8_gemms=use_fp8_gemms,
            use_fp8_text_encoder=use_fp8_text_encoder,
        ),
    )
    return Fp8Plan(model)


# ============================================================================
# aiter_active
# ============================================================================


def test_inactive_without_fp8_gemms(monkeypatch):
    assert not make_plan(monkeypatch, use_fp8_gemms=False).aiter_active


def test_inactive_off_rdna4(monkeypatch):
    """The AITER block-scale kernels are the only implementation, so FP8 is off elsewhere."""
    assert not make_plan(monkeypatch, on_rdna4=False).aiter_active


# ============================================================================
# module_list: the --use_fp8_text_encoder opt-in
# ============================================================================


def test_text_encoder_targets_excluded_by_default(monkeypatch):
    """Quantizing a text encoder is an output-quality trade-off, so it takes an explicit flag."""
    plan = make_plan(
        monkeypatch,
        transformer_targets=["transformer.blocks"],
        te_targets=["text_encoder.encoder.block"],
    )
    assert plan.module_list() == ["transformer.blocks"]


def test_text_encoder_targets_included_when_flag_set(monkeypatch):
    plan = make_plan(
        monkeypatch,
        transformer_targets=["transformer.blocks"],
        te_targets=["text_encoder.encoder.block"],
        use_fp8_text_encoder=True,
    )
    assert plan.module_list() == ["transformer.blocks", "text_encoder.encoder.block"]


def test_flag_without_declared_targets_is_inert(monkeypatch):
    """A model that declares no text-encoder targets is unaffected by the flag."""
    plan = make_plan(
        monkeypatch,
        transformer_targets=["transformer.blocks"],
        use_fp8_text_encoder=True,
    )
    assert plan.module_list() == ["transformer.blocks"]


def test_module_list_empty_when_model_declares_nothing(monkeypatch):
    assert make_plan(monkeypatch).module_list() == []


def test_module_list_does_not_alias_settings(monkeypatch):
    """Consumers mutating the returned list must not edit the model's declared targets."""
    targets = ["transformer.blocks"]
    plan = make_plan(monkeypatch, transformer_targets=targets)
    plan.module_list().append("transformer.extra")
    assert targets == ["transformer.blocks"]


# ============================================================================
# targets_for / aiter_covers: per-component prefix matching
# ============================================================================


def test_targets_are_stripped_of_the_component_prefix(monkeypatch):
    """Loaders take component-relative paths, while the model declares pipe-level ones."""
    plan = make_plan(
        monkeypatch,
        te_targets=["text_encoder.model.language_model.layers"],
        use_fp8_text_encoder=True,
    )
    assert plan.targets_for("text_encoder") == ["model.language_model.layers"]


def test_prefix_match_does_not_leak_across_sibling_components(monkeypatch):
    """ "transformer_2.blocks" must not count as a target of "transformer"."""
    plan = make_plan(
        monkeypatch, transformer_targets=["transformer.blocks", "transformer_2.blocks"]
    )
    assert plan.targets_for("transformer") == ["blocks"]
    assert plan.targets_for("transformer_2") == ["blocks"]


def test_aiter_covers_is_false_for_an_untargeted_component(monkeypatch):
    plan = make_plan(monkeypatch, transformer_targets=["transformer.blocks"])
    assert plan.aiter_covers("transformer")
    assert not plan.aiter_covers("vae")


def test_aiter_covers_is_false_for_a_text_encoder_without_the_flag(monkeypatch):
    plan = make_plan(monkeypatch, te_targets=["text_encoder.encoder.block"])
    assert not plan.aiter_covers("text_encoder")


def test_aiter_covers_is_false_when_the_aiter_path_is_off(monkeypatch):
    plan = make_plan(
        monkeypatch, transformer_targets=["transformer.blocks"], use_fp8_gemms=False
    )
    assert not plan.aiter_covers("transformer")


# ============================================================================
# loader configs
# ============================================================================


def test_no_stream_quant_config_when_inactive(monkeypatch):
    plan = make_plan(
        monkeypatch, transformer_targets=["transformer.blocks"], use_fp8_gemms=False
    )
    assert plan.aiter_stream_config("transformer") is None


def test_no_stream_quant_config_for_an_untargeted_component(monkeypatch):
    plan = make_plan(monkeypatch, transformer_targets=["transformer.blocks"])
    assert plan.aiter_stream_config("transformer_2") is None


def test_no_te_quant_config_without_the_flag(monkeypatch):
    """Belt and braces: the TE streaming config is gated on the flag as well as the target list."""
    plan = make_plan(monkeypatch, te_targets=["text_encoder.encoder.block"])
    assert plan.aiter_te_pipeline_config() is None


def test_no_te_quant_config_when_inactive(monkeypatch):
    plan = make_plan(
        monkeypatch,
        te_targets=["text_encoder.encoder.block"],
        use_fp8_text_encoder=True,
        use_fp8_gemms=False,
    )
    assert plan.aiter_te_pipeline_config() is None


def test_no_te_quant_config_for_a_bare_component_name(monkeypatch):
    """An entry with no sub-module path ("text_encoder") targets nothing the loader can route."""
    plan = make_plan(
        monkeypatch, te_targets=["text_encoder"], use_fp8_text_encoder=True
    )
    assert plan.aiter_te_pipeline_config() is None


def test_te_quant_config_groups_targets_by_component(monkeypatch):
    quantizers = pytest.importorskip("diffusers.quantizers")
    if not hasattr(quantizers, "PipelineQuantizationConfig"):
        pytest.skip("installed diffusers has no PipelineQuantizationConfig")
    plan = make_plan(
        monkeypatch,
        te_targets=["text_encoder.encoder.block", "text_encoder_2.layers"],
        use_fp8_text_encoder=True,
    )
    config = plan.aiter_te_pipeline_config()
    assert set(config.quant_mapping) == {"text_encoder", "text_encoder_2"}
    assert config.quant_mapping["text_encoder"].target_modules == ["encoder.block"]
    assert config.quant_mapping["text_encoder_2"].target_modules == ["layers"]


# ============================================================================
# Every runner declares its targets in the right list
# ============================================================================


def test_no_runner_hides_a_text_encoder_in_the_always_on_list():
    """A text-encoder path left in fp8_gemm_module_list is quantized unconditionally, which breaks
    two ways: on CUDA the torchao walk silently quantizes a text encoder the user never opted into,
    and on the replicated broadcast path aiter_covers() turns True while aiter_te_pipeline_config()
    stays None, so peers fp8-swap a component rank0 loads bf16 and the load hangs on mismatched
    tensor counts. Checked over the registry because the split is per-runner and easy to miss (flux
    was missed once, in the exact configuration the feature targets).

    A denoiser is not always the component literally named "transformer": Ideogram 4 carries a second
    unconditional_transformer and MiniMax-H3-Ref2VA names its own transformer_ref, so the component
    is matched on containing "transformer" rather than starting with it."""
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    leaks = {}
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        stray = [
            entry
            for entry in (cls.settings.fp8_gemm_module_list or [])
            if "transformer" not in entry.partition(".")[0]
        ]
        if stray:
            leaks[cls.__name__] = stray

    assert not leaks, (
        "these runners list non-transformer targets in fp8_gemm_module_list; move them to "
        f"fp8_text_encoder_module_list so --use_fp8_text_encoder gates them: {leaks}"
    )
