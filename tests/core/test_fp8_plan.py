"""Unit tests for backend-neutral quantization target planning.

Every FP8 consumer (the post-load walks on any hardware, the per-block FSDP quantize, the streaming
quantize-on-load, the meta-init paths) reads its target list from here, so the
--use_fp8_text_encoder opt-in and the prefix matching are pinned here rather than left to a GPU run
to discover.

Run with:
    pytest tests/core/test_fp8_plan.py -v
"""

from types import SimpleNamespace

from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
    apply_fp8_override_cli_to_settings,
)


def make_plan(
    monkeypatch,
    *,
    transformer_targets=None,
    te_targets=None,
    use_fp8_gemms=True,
    use_fp8_text_encoder=False,
):
    """A QuantizationPlan over a stand-in runner."""
    model = SimpleNamespace(
        settings=SimpleNamespace(
            fp8_gemm_module_list=transformer_targets,
            fp8_text_encoder_module_list=te_targets,
            fp4_gemm_module_list=["transformer.fp4_blocks"],
            int8_gemm_module_list=["transformer.int8_blocks"],
        ),
        config=SimpleNamespace(
            use_fp8_gemms=use_fp8_gemms,
            use_fp8_text_encoder=use_fp8_text_encoder,
        ),
    )
    return QuantizationPlan(model)


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


def test_backend_neutral_targets_cover_fp4_and_int8(monkeypatch):
    plan = make_plan(monkeypatch)

    assert plan.targets_for("transformer", "fp4") == ["fp4_blocks"]
    assert plan.targets_for("transformer", "int8") == ["int8_blocks"]


def test_fp8_override_cli_patterns_update_settings_per_slot():
    settings = SimpleNamespace(
        fp8_precision_overrides=("declared-prefix",),
        fp8_precision_override_suffixes=("declared-suffix",),
    )
    config = SimpleNamespace(
        fp8_precision_override_prefix_patterns=" blocks.0, ,blocks.2 ",
        fp8_precision_override_suffix_patterns=None,
    )

    apply_fp8_override_cli_to_settings(config, settings)

    assert settings.fp8_precision_overrides == ("blocks.0", "blocks.2")
    assert settings.fp8_precision_override_suffixes == ("declared-suffix",)


def test_model_loader_materialization_logs_fp4_overrides_and_places(monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import (
        placement,
        quantization_plan,
    )
    from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader

    messages = []
    placed = []
    model = SimpleNamespace(
        config=SimpleNamespace(use_fp4_gemms=True, fully_shard_degree=1),
        settings=SimpleNamespace(
            fp8_precision_overrides=("blocks.0",),
            fp8_precision_override_suffixes=(".proj",),
        ),
    )
    loader = SimpleNamespace(
        model=model,
        quantization_plan=QuantizationPlan(model),
    )
    monkeypatch.setattr(quantization_plan, "log", messages.append)
    monkeypatch.setattr(placement, "place_pipeline_components", placed.append)

    ModelLoader.materialize_pipeline(loader)

    assert placed == [loader]
    assert messages == [
        "The following layers will be quantized to FP8, to maintain output quality: "
        "('blocks.0',) (prefix match)",
        "The following layers will be quantized to FP8, to maintain output quality: "
        "('.proj',) (suffix match)",
    ]


def test_model_loader_materialization_uses_current_shard_degree(monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import placement, shard
    from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader

    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(use_fp4_gemms=False, fully_shard_degree=2)
    )
    loader = SimpleNamespace(model=model)
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


# ============================================================================
# Every runner declares its targets in the right list
# ============================================================================


def test_registered_runner_text_encoder_capability_matches_declared_targets():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatches = {
        cls.__name__: {
            "capability": cls.capabilities.use_fp8_text_encoder,
            "targets": cls.settings.fp8_text_encoder_module_list,
        }
        for cls in dict.fromkeys(MODEL_REGISTRY.values())
        if cls.capabilities.use_fp8_text_encoder
        != bool(cls.settings.fp8_text_encoder_module_list)
    }

    assert not mismatches


def test_no_runner_hides_a_text_encoder_in_the_always_on_list():
    """A text-encoder path left in fp8_gemm_module_list is quantized unconditionally, which breaks
    two ways: on CUDA the torchao walk silently quantizes a text encoder the user never opted into,
    and on the replicated broadcast path the generic FP8 target plan can claim coverage while the
    text-encoder load remains bf16, so peers swap a different layout and hang on mismatched tensor
    counts. Checked over the registry because the split is per-runner and easy to miss (flux was
    missed once, in the exact configuration the feature targets).

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
