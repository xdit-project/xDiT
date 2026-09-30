"""FLUX.2-dev: the resolver must decide what the legacy fields decided.

The oracle that made deleting the legacy fields safe. Those fields and the
code that read them are gone, so the comparison is now against the recorded
declaration and the rule the old code applied to it, both written out below --
a specification cannot drift with the code it checks.
"""

import copy
from types import SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.flux import xFuserFlux2Model
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)
from xfuser.model_executor.quant.targets import resolve

TRANSFORMER = (
    "transformer.transformer_blocks",
    "transformer.single_transformer_blocks",
)
TEXT_ENCODER = "text_encoder.model.language_model.layers"

# FLUX.2-dev declared no precision patterns, so tiering inside a block was a
# no-op for it and the module lists carried the whole decision.
#: What FLUX.2-dev declared before it was migrated, recorded here so the
#: comparison outlives the fields themselves. Every test below that says
#: "the legacy path" means these values through the unchanged legacy code.
LEGACY_FP8_GEMM = ["transformer.transformer_blocks", "transformer.single_transformer_blocks"]
LEGACY_FP4_GEMM = ["transformer.transformer_blocks", "transformer.single_transformer_blocks"]
LEGACY_TEXT_ENCODER = ["text_encoder.model.language_model.layers"]
assert LEGACY_FP8_GEMM == LEGACY_FP4_GEMM  # the split was never by format


def _legacy_listed(spec, format_name, *, text_encoder):
    """The list the legacy fields gave one format, before any subtraction.

    The rule, stated: a pure fp4 run folded the fp8 targets into the fp4 list
    and emptied the fp8 one; the text encoder rode on the fp8 list, and only
    when the run asked for it.
    """
    fp8 = list(LEGACY_FP8_GEMM)
    fp4 = list(LEGACY_FP4_GEMM)
    if spec.is_pure("fp4"):
        fp4 = list(dict.fromkeys(fp4 + fp8))
        fp8 = []

    if format_name == "fp8":
        return fp8 + (list(LEGACY_TEXT_ENCODER) if text_encoder else [])
    if format_name == "fp4":
        return fp4
    if format_name == "fp6":
        return list(dict.fromkeys(fp4 + fp8))
    raise ValueError(format_name)


def _legacy(spec: GemmQuantizationSpec, *, text_encoder: bool) -> dict:
    """What the legacy fields targeted per format, by the rule they followed."""
    targets = {
        fmt: _legacy_listed(spec, fmt, text_encoder=text_encoder)
        for fmt in spec.formats
        if fmt != "none"
    }
    if spec.is_tiered:
        # A tiered run listed every eligible module under *both* formats; only
        # the modules appearing under the high format alone were actually held
        # there. That is the subtraction its consumers performed.
        low = targets[spec.low]
        targets[spec.high] = [
            t
            for t in targets[spec.high]
            if not any(t == m or t.startswith(f"{m}.") for m in low)
        ]
    return {fmt: sorted(t) for fmt, t in targets.items()}


def _resolved(spec: GemmQuantizationSpec, *, text_encoder: bool) -> dict:
    enable = ("transformer", "text_encoder") if text_encoder else ("transformer",)
    plan = resolve(xFuserFlux2Model.settings.gemm_targets, spec, enable=enable)
    return {
        fmt: sorted(plan.roots(fmt))
        for fmt in spec.formats
        if fmt != "none"
    }


# FLUX.2-dev's capabilities allow fp8, fp4 and the fp4/fp8 tier; the text
# encoder needs fp8 in the profile, which args.py enforces separately.
CASES = [
    ("fp8", False),
    ("fp8", True),
    ("fp4", False),
    ("low=fp4,high=fp8", False),
    ("low=fp4,high=fp8", True),
]


@pytest.mark.parametrize("raw, text_encoder", CASES)
def test_the_resolver_agrees_with_the_legacy_fields(raw, text_encoder):
    spec = GemmQuantizationSpec.parse(raw)
    assert _resolved(spec, text_encoder=text_encoder) == _legacy(
        spec, text_encoder=text_encoder
    )


def test_none_quantizes_nothing():
    plan = resolve(
        xFuserFlux2Model.settings.gemm_targets, GemmQuantizationSpec.parse("none")
    )
    assert not plan.quantizes
    assert plan.format_for(f"{TRANSFORMER[0]}.3.attn.to_qkv") is None


@pytest.mark.parametrize("raw", ["fp8", "fp4", "low=fp4,high=fp8"])
def test_every_transformer_block_is_quantized(raw):
    """Leaf paths, not just the declared roots: this is what the loaders walk."""
    plan = resolve(xFuserFlux2Model.settings.gemm_targets, GemmQuantizationSpec.parse(raw))
    spec = GemmQuantizationSpec.parse(raw)
    for root in TRANSFORMER:
        assert plan.format_for(f"{root}.0.attn.to_qkv") == spec.low
    assert plan.format_for("transformer.norm_out") is None
    assert plan.format_for("transformer.transformer_blocks_2.0") is None


def test_the_text_encoder_stays_fp8_in_every_profile_that_allows_it():
    """Today it is always fp8; keep_high is what preserves that under a tier."""
    leaf = f"{TEXT_ENCODER}.3.mlp.gate_proj"
    enable = ("transformer", "text_encoder")

    targets = xFuserFlux2Model.settings.gemm_targets
    pure = resolve(targets, GemmQuantizationSpec.parse("fp8"), enable=enable)
    tiered = resolve(targets, GemmQuantizationSpec.parse("low=fp4,high=fp8"), enable=enable)

    assert pure.format_for(leaf) == "fp8"
    assert tiered.format_for(leaf) == "fp8"
    # ... while the DiT drops to fp4 around it
    assert tiered.format_for(f"{TRANSFORMER[0]}.0.attn.to_qkv") == "fp4"


def test_the_text_encoder_is_untouched_unless_the_run_asks():
    plan = resolve(
        xFuserFlux2Model.settings.gemm_targets, GemmQuantizationSpec.parse("fp8")
    )
    assert plan.format_for(f"{TEXT_ENCODER}.3.mlp.gate_proj") is None


# ---------------------------------------------------------------------------
# phase 2: the shim feeding the existing consumers
# ---------------------------------------------------------------------------

FORMATS = ("fp8", "fp4", "fp6")


def _plan_for(raw: str, *, text_encoder: bool) -> QuantizationPlan:
    """A QuantizationPlan over FLUX.2-dev's declaration."""
    spec = GemmQuantizationSpec.parse(raw)
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        _gemm_config_loaded=False,
        quantize_text_encoder=text_encoder,
        use_hybrid_gemm_schedule=False,
        gemm_high_precision_targets="model",
        ulysses_degree=1,
        ring_degree=1,
    )
    return QuantizationPlan(
        SimpleNamespace(
            settings=copy.deepcopy(xFuserFlux2Model.settings),
            config=config,
            capabilities=xFuserFlux2Model.capabilities,
        )
    )


# The per-format shim is gone: consumers ask the plan directly, so there
# is no list-handing to compare. What each format owns is checked by
# test_the_resolver_agrees_with_the_legacy_fields above.


def test_the_high_tier_is_what_consumers_recover_by_subtracting():
    """What the consumers' fp8_list - fp4_list used to recover, read off the plan."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True).gemm_plan
    assert list(plan.roots(plan.high)) == [TEXT_ENCODER]
    assert set(plan.roots(plan.low)) == set(TRANSFORMER)


def test_keep_high_on_a_transformer_module_survives_the_subtraction():
    """FLUX.2 holds only its text encoder high; a DiT carve-out must work too."""
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    holder = _plan_for("low=fp4,high=fp8", text_encoder=False)
    holder.model.settings.gemm_targets = GemmTargets(
        transformer=Select(modules=TRANSFORMER),
        keep_high=Select(modules=(TRANSFORMER[1],)),
    )
    plan = holder.gemm_plan
    assert set(plan.roots(plan.low)) == {TRANSFORMER[0]}
    assert list(plan.roots(plan.high)) == [TRANSFORMER[1]]


@pytest.mark.parametrize("unsupported", ["fp6", "int8"])
def test_an_unsupported_format_is_refused_before_any_target_is_read(unsupported):
    """Targets are format-agnostic, so the refusal has to happen upstream.

    Without it, asking for int8 would happily claim the DiT that FLUX.2
    declares no int8 support for -- the declaration names modules, and says
    nothing about which formats may be applied to them.
    """
    from xfuser.config.args import xFuserArgs

    supported = xFuserFlux2Model.capabilities.supported_gemm_formats()
    assert unsupported not in supported

    model = object.__new__(xFuserFlux2Model)
    model.settings = copy.deepcopy(xFuserFlux2Model.settings)
    config = xFuserArgs(
        model=model.settings.model_name, gemm_quantization=unsupported
    )

    with pytest.raises(ValueError, match="does not support GEMM format"):
        model._validate_config(config)


# ---------------------------------------------------------------------------
# phase 3: consumers rewritten onto GemmPlan
# ---------------------------------------------------------------------------

#: The high tier was placed only under the fp4/fp6 backends. A pure fp8 run
#: converted through a different walk and never computed a subtraction, so
#: those profiles are not an oracle for anything here.
FORMAT_PATH_CASES = [c for c in CASES if GemmQuantizationSpec.parse(c[0]).low == "fp4"]


@pytest.mark.parametrize("raw, text_encoder", FORMAT_PATH_CASES)
def test_the_high_tier_survives_the_rewrite(raw, text_encoder):
    """`roots(high)` must name what the subtraction named, profile by profile."""
    spec = GemmQuantizationSpec.parse(raw)
    gemm_plan = _plan_for(raw, text_encoder=text_encoder).gemm_plan
    rewritten = list(gemm_plan.roots(gemm_plan.high)) if gemm_plan.high else []
    expected = (
        _legacy(spec, text_encoder=text_encoder)[spec.high] if spec.high else []
    )
    assert sorted(rewritten) == expected


def test_the_high_tier_is_the_text_encoder_under_a_tier():
    """The one profile where FLUX.2 actually has a high tier to place."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True).gemm_plan
    assert plan.roots(plan.high) == (TEXT_ENCODER,)
    assert sorted(plan.roots(plan.low)) == sorted(TRANSFORMER)


@pytest.mark.parametrize("raw", ["fp8", "fp4"])
def test_a_pure_profile_places_no_high_tier(raw):
    """`setup_high_tier_gemms` must stay a no-op, as it is today."""
    plan = _plan_for(raw, text_encoder=False).gemm_plan
    assert plan.high is None


def test_the_high_format_drives_the_adapter_choice():
    """It was a pair of capability booleans; now any tier names its own."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True).gemm_plan
    assert plan.high == "fp8"


# ---------------------------------------------------------------------------
# phase 4, step 1: the log says what the plan decided
# ---------------------------------------------------------------------------

def _logged(plan_obj, monkeypatch):
    lines = []
    from xfuser.model_executor.models.runner_models.loading import quantization_plan
    monkeypatch.setattr(quantization_plan, "log", lines.append)
    plan_obj.log_gemm_plan()
    return lines


def _mapping(lines, *, component="transformer"):
    """The target -> format decisions for one component, sans tier header."""
    return sorted(
        l
        for l in lines
        if l.startswith(f"GEMM quantization: {component}")
    )


def test_the_log_now_mentions_the_text_encoder(monkeypatch):
    """A deliberate addition: it is quantized, so the log should say so.

    The legacy logger reported "the resolved transformer target-to-format
    mapping" and stopped there, leaving the encoder's format invisible in the
    one place a run tells you what it did.
    """
    lines = _logged(_plan_for("fp8", text_encoder=True), monkeypatch)

    assert _mapping(lines, component="text_encoder") == [
        f"GEMM quantization: {TEXT_ENCODER} -> FP8"
    ]


def test_the_tier_log_names_the_high_modules(monkeypatch):
    lines = _logged(
        _plan_for("low=fp4,high=fp8", text_encoder=True), monkeypatch
    )
    header = [l for l in lines if l.startswith("GEMM high-precision tier:")]
    assert len(header) == 1
    assert "format=fp8" in header[0] and TEXT_ENCODER in header[0]
    assert f"GEMM quantization: {TEXT_ENCODER} -> FP8" in lines
    assert f"GEMM quantization: {TRANSFORMER[0]} -> FP4" in lines


def test_an_unquantized_run_logs_nothing(monkeypatch):
    assert _logged(_plan_for("none", text_encoder=False), monkeypatch) == []


# ---------------------------------------------------------------------------
# phase 4, step 2: the text-encoder readers
# ---------------------------------------------------------------------------

def test_declared_components_reads_the_declaration_not_the_run():
    """It lists encoders to plan for, which the run's format cannot change."""
    from xfuser.model_executor.models.runner_models.loading.text_encoder_plan import (
        _declared_components,
    )

    declared = SimpleNamespace(settings=xFuserFlux2Model.settings)

    assert _declared_components(declared) == ("text_encoder",)


def test_declared_components_is_empty_without_text_encoder_targets():
    from xfuser.model_executor.models.runner_models.loading.text_encoder_plan import (
        _declared_components,
    )
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    settings = copy.deepcopy(xFuserFlux2Model.settings)
    settings.gemm_targets = GemmTargets(transformer=Select(modules=TRANSFORMER))
    settings.fp8_text_encoder_module_list = None
    assert _declared_components(SimpleNamespace(settings=settings)) == ()


# ---------------------------------------------------------------------------
# the advanced GEMM config overrides keep_high
# ---------------------------------------------------------------------------

def _configured(raw, **yaml):
    """A plan with an advanced GEMM config file applied."""
    plan = _plan_for(raw, text_encoder=True)
    plan.model.config._gemm_config_loaded = True
    plan.model.config.gemm_high_precision_targets = yaml.get("targets", "model")
    plan.model.config.gemm_high_precision_module_patterns = yaml.get("modules")
    plan.model.config.gemm_high_precision_prefix_patterns = yaml.get("prefixes")
    plan.model.config.gemm_high_precision_suffix_patterns = yaml.get("suffixes")
    return plan.gemm_plan


def test_a_config_file_can_hold_extra_modules_high():
    plan = _configured("low=fp4,high=fp8", modules=TRANSFORMER[1])
    assert plan.format_for(f"{TRANSFORMER[1]}.0.attn.to_qkv") == "fp8"
    assert plan.format_for(f"{TRANSFORMER[0]}.0.attn.to_qkv") == "fp4"
    # the model's own keep_high still applies
    assert plan.format_for(f"{TEXT_ENCODER}.3.mlp") == "fp8"


def test_a_config_file_can_clear_the_models_keep_high():
    plan = _configured("low=fp4,high=fp8", targets="none")
    assert plan.format_for(f"{TEXT_ENCODER}.3.mlp") == "fp4"
    assert plan.format_for(f"{TRANSFORMER[0]}.0.attn.to_qkv") == "fp4"


def test_config_suffixes_hold_matching_leaves_high():
    plan = _configured("low=fp4,high=fp8", suffixes="attn.to_out.0")
    assert plan.format_for(f"{TRANSFORMER[0]}.3.attn.to_out.0") == "fp8"
    assert plan.format_for(f"{TRANSFORMER[0]}.3.attn.to_qkv") == "fp4"


def test_a_pattern_outside_the_targets_is_refused_not_ignored():
    """The legacy path raised on unknown modules; silence would be worse."""
    with pytest.raises(ValueError, match="outside the target set"):
        _configured("low=fp4,high=fp8", modules="transformer.norm_out")


def test_no_config_file_leaves_the_declaration_alone():
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True).gemm_plan
    assert plan.format_for(f"{TRANSFORMER[0]}.0.attn.to_qkv") == "fp4"
    assert plan.format_for(f"{TEXT_ENCODER}.3.mlp") == "fp8"
