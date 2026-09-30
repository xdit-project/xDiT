"""Unit tests for backend-neutral quantization target planning.

Every FP8 consumer (the post-load walks on any hardware, the per-block FSDP quantize, the streaming
quantize-on-load, the meta-init paths) reads its target list from here, so the
--quantize_text_encoder opt-in and the prefix matching are pinned here rather than left to a GPU run
to discover.

Run with:
    pytest tests/core/test_fp8_plan.py -v
"""

from types import SimpleNamespace

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)
from xfuser.model_executor.quant.targets import GemmTargets, Select


def make_plan(
    monkeypatch,
    *,
    transformer_targets=None,
    te_targets=None,
    raw="fp8",
    quantize_text_encoder=False,
):
    """A QuantizationPlan over a stand-in runner."""
    model = SimpleNamespace(
        settings=SimpleNamespace(
            gemm_targets=GemmTargets(
                transformer=Select(modules=tuple(transformer_targets or ())),
                text_encoder=Select(modules=tuple(te_targets or ())),
            ),
        ),
        config=SimpleNamespace(
            gemm_quantization_spec=GemmQuantizationSpec.parse(raw),
            quantize_text_encoder=quantize_text_encoder,
            ulysses_degree=1,
            ring_degree=1,
        ),
    )
    return QuantizationPlan(model)


# ============================================================================
# module_list: the --quantize_text_encoder opt-in
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
        quantize_text_encoder=True,
    )
    assert plan.module_list() == ["transformer.blocks", "text_encoder.encoder.block"]


def test_flag_without_declared_targets_is_inert(monkeypatch):
    """A model that declares no text-encoder targets is unaffected by the flag."""
    plan = make_plan(
        monkeypatch,
        transformer_targets=["transformer.blocks"],
        quantize_text_encoder=True,
    )
    assert plan.module_list() == ["transformer.blocks"]


def test_module_list_empty_when_model_declares_nothing(monkeypatch):
    assert make_plan(monkeypatch).module_list() == []


def test_module_list_does_not_alias_the_declaration(monkeypatch):
    """Consumers mutating the returned list must not edit the declared targets."""
    plan = make_plan(monkeypatch, transformer_targets=["transformer.blocks"])
    plan.module_list().append("transformer.extra")
    assert plan.module_list() == ["transformer.blocks"]


def test_every_format_sees_the_same_declared_targets(monkeypatch):
    """A model declares which modules, never which modules per format."""
    plan = make_plan(monkeypatch, transformer_targets=["transformer.blocks"])

    for format_name in ("fp8", "fp4", "fp6", "int8"):
        assert plan.targets_for("transformer", format_name) == ["blocks"]


def test_model_loader_materialization_uses_current_shard_degree(monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import placement, shard
    from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader

    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(use_fp4_gemms=False, fully_shard_degree=2)
    )
    loader = SimpleNamespace(
        model=model,
        quantization_plan=SimpleNamespace(log_gemm_plan=lambda: None),
    )
    monkeypatch.setattr(
        shard, "shard_pipeline_components", lambda value: calls.append(("shard", value))
    )
    monkeypatch.setattr(
        placement,
        "place_pipeline_components",
        lambda value: calls.append(("place", value)),
    )

    ModelLoader.materialize_pipeline(loader)
    model.config.fully_shard_degree = 1
    ModelLoader.materialize_pipeline(loader)

    assert calls == [("shard", loader), ("place", loader)]


# ============================================================================
# targets_for: per-component prefix matching
# ============================================================================


def test_targets_are_stripped_of_the_component_prefix(monkeypatch):
    """Loaders take component-relative paths, while the model declares pipe-level ones."""
    plan = make_plan(
        monkeypatch,
        te_targets=["text_encoder.model.language_model.layers"],
        quantize_text_encoder=True,
    )
    assert plan.targets_for("text_encoder") == ["model.language_model.layers"]


def test_prefix_match_does_not_leak_across_sibling_components(monkeypatch):
    """ "transformer_2.blocks" must not count as a target of "transformer"."""
    plan = make_plan(
        monkeypatch, transformer_targets=["transformer.blocks", "transformer_2.blocks"]
    )
    assert plan.targets_for("transformer") == ["blocks"]
    assert plan.targets_for("transformer_2") == ["blocks"]


# ============================================================================
# Every runner declares its targets in the right list
# ============================================================================


def test_registered_runner_text_encoder_capability_matches_declared_targets():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    def declared(cls):
        """This runner's text-encoder targets."""
        targets = cls.settings.gemm_targets
        return list(targets.text_encoder.roots()) if targets is not None else []

    mismatches = {
        cls.__name__: {
            "capability": cls.capabilities.quantize_text_encoder,
            "targets": declared(cls),
        }
        for cls in dict.fromkeys(MODEL_REGISTRY.values())
        if cls.capabilities.quantize_text_encoder != bool(declared(cls))
    }

    assert not mismatches


def test_no_runner_hides_a_text_encoder_in_the_always_on_list():
    """A text-encoder path declared as a transformer target is quantized
    unconditionally, which breaks two ways: on CUDA the walk silently quantizes
    an encoder the user never opted into, and on the replicated broadcast path
    the plan can claim coverage while the encoder load stays bf16, so peers swap
    a different layout and hang on mismatched tensor counts. Checked over the
    registry because the split is per-runner and easy to miss (flux was missed
    once, in the exact configuration the feature targets).
    """
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    leaks = {}
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        targets = cls.settings.gemm_targets
        if targets is None:
            continue
        stray = [
            entry
            for entry in targets.transformer.roots()
            if "text_encoder" in entry.partition(".")[0]
        ]
        if stray:
            leaks[cls.__name__] = stray

    assert not leaks, (
        "these runners declare a text encoder among their transformer targets; "
        f"move it to gemm_targets.text_encoder so --quantize_text_encoder gates it: {leaks}"
    )
