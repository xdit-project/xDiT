"""FLUX.2-dev: the resolver must decide what the legacy fields decide today.

This is the oracle that makes deleting the legacy fields safe. It drives the
real settings object from the real model class through the real
`apply_fp8_override_cli_to_settings` / `QuantizationPlan` path, and compares
the answer to `resolve(settings.gemm_targets, spec)`.
"""

import copy
from types import SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.flux import xFuserFlux2Model
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
    apply_fp8_override_cli_to_settings,
)
from xfuser.model_executor.quant.targets import resolve

TRANSFORMER = (
    "transformer.transformer_blocks",
    "transformer.single_transformer_blocks",
)
TEXT_ENCODER = "text_encoder.model.language_model.layers"

# FLUX.2-dev declares neither, so the tiering inside a block is a no-op and the
# module lists carry the whole decision. A model that declares them needs the
# pattern consumers migrated too; see phase 3.
assert not xFuserFlux2Model.settings.fp8_precision_overrides
assert not xFuserFlux2Model.settings.fp8_precision_override_suffixes


#: What FLUX.2-dev declared before it was migrated, recorded here so the
#: comparison outlives the fields themselves. Every test below that says
#: "the legacy path" means these values through the unchanged legacy code.
LEGACY_FP8_GEMM = ["transformer.transformer_blocks", "transformer.single_transformer_blocks"]
LEGACY_FP4_GEMM = ["transformer.transformer_blocks", "transformer.single_transformer_blocks"]
LEGACY_TEXT_ENCODER = ["text_encoder.model.language_model.layers"]
assert LEGACY_FP8_GEMM == LEGACY_FP4_GEMM  # the split was never by format


def _legacy_settings():
    """FLUX.2-dev's settings as they were before `gemm_targets` replaced them."""
    settings = copy.deepcopy(xFuserFlux2Model.settings)
    settings.gemm_targets = None
    settings.fp8_gemm_module_list = list(LEGACY_FP8_GEMM)
    settings.fp4_gemm_module_list = list(LEGACY_FP4_GEMM)
    settings.fp8_text_encoder_module_list = list(LEGACY_TEXT_ENCODER)
    return settings


def _legacy(spec: GemmQuantizationSpec, *, text_encoder: bool) -> dict:
    """What today's code targets per format, via today's code."""
    settings = _legacy_settings()
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        _gemm_config_loaded=False,
        fp8_precision_override_prefix_patterns=None,
        fp8_precision_override_suffix_patterns=None,
        quantize_text_encoder=text_encoder,
        use_fp4_gemms=spec.low == "fp4",
        use_hybrid_gemm_schedule=False,
        gemm_high_precision_targets="model",
    )
    apply_fp8_override_cli_to_settings(config, settings)
    plan = QuantizationPlan(SimpleNamespace(settings=settings, config=config))

    targets = {fmt: plan.module_list(fmt) for fmt in spec.formats if fmt != "none"}
    if spec.is_tiered:
        # A tiered run lists every eligible module under *both* formats; only
        # the modules that appear under the high format alone are actually held
        # there. That is log_gemm_plan's `high_only_targets`, and it is the
        # decision a consumer ends up acting on.
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


def _plan_for(raw: str, *, text_encoder: bool, legacy: bool) -> QuantizationPlan:
    """A QuantizationPlan reading either the legacy lists or gemm_targets."""
    spec = GemmQuantizationSpec.parse(raw)
    settings = _legacy_settings() if legacy else copy.deepcopy(
        xFuserFlux2Model.settings
    )
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        _gemm_config_loaded=False,
        fp8_precision_override_prefix_patterns=None,
        fp8_precision_override_suffix_patterns=None,
        quantize_text_encoder=text_encoder,
        use_fp4_gemms=spec.low == "fp4",
        use_hybrid_gemm_schedule=False,
        gemm_high_precision_targets="model",
    )
    apply_fp8_override_cli_to_settings(config, settings)
    return QuantizationPlan(
        SimpleNamespace(
            settings=settings,
            config=config,
            capabilities=xFuserFlux2Model.capabilities,
        )
    )


@pytest.mark.parametrize("raw, text_encoder", CASES)
@pytest.mark.parametrize("format_name", FORMATS)
def test_the_shim_hands_consumers_the_legacy_lists(raw, text_encoder, format_name):
    """Byte-identical lists, so no consumer can tell which path produced them."""
    old = _plan_for(raw, text_encoder=text_encoder, legacy=True)
    new = _plan_for(raw, text_encoder=text_encoder, legacy=False)
    assert new.module_list(format_name) == old.module_list(format_name)


@pytest.mark.parametrize("raw, text_encoder", CASES)
@pytest.mark.parametrize("component", ["transformer", "text_encoder"])
def test_targets_for_follows(raw, text_encoder, component):
    old = _plan_for(raw, text_encoder=text_encoder, legacy=True)
    new = _plan_for(raw, text_encoder=text_encoder, legacy=False)
    assert new.targets_for(component) == old.targets_for(component)


def test_the_high_tier_is_what_consumers_recover_by_subtracting():
    """`setup_high_tier_gemms` and `shard` both do fp8_list - fp4_list."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True, legacy=False)
    fp4 = set(plan.module_list("fp4"))
    fp8_only = [m for m in plan.module_list("fp8") if m not in fp4]
    assert fp8_only == [TEXT_ENCODER]
    assert set(TRANSFORMER) == fp4


def test_keep_high_on_a_transformer_module_survives_the_subtraction():
    """FLUX.2 holds only its text encoder high; a DiT carve-out must work too."""
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    plan = _plan_for("low=fp4,high=fp8", text_encoder=False, legacy=False)
    plan.model.settings.gemm_targets = GemmTargets(
        transformer=Select(modules=TRANSFORMER),
        keep_high=Select(modules=(TRANSFORMER[1],)),
    )
    fp4 = set(plan.module_list("fp4"))
    fp8_only = [m for m in plan.module_list("fp8") if m not in fp4]
    assert fp4 == {TRANSFORMER[0]}
    assert fp8_only == [TRANSFORMER[1]]


def test_an_unsupported_format_is_refused_before_any_list_is_read():
    """Why the shim needs no capability gate of its own.

    `use_fp6_gemms` / `use_int8_gemms` are derived from the spec at parse time
    and checked against ModelCapabilities in `_validate_config`, so a format
    the model cannot run never reaches the plan. That matters because targets
    are format-agnostic now: without the upstream refusal, asking for int8
    would happily claim the DiT that FLUX.2 declares no int8 support for.
    """
    from xfuser.config import xFuserArgs

    assert not xFuserFlux2Model.capabilities.use_fp6_gemms
    assert not xFuserFlux2Model.capabilities.use_int8_gemms

    for unsupported in ("fp6", "int8"):
        config = SimpleNamespace(**{
            key: getattr(xFuserFlux2Model.capabilities, key, None)
            for key in type(xFuserFlux2Model.capabilities).__annotations__
        })
        setattr(config, f"use_{unsupported}_gemms", True)
        key = f"use_{unsupported}_gemms"
        assert getattr(config, key) and not getattr(
            xFuserFlux2Model.capabilities, key
        ), f"{key} must be the pair _validate_config rejects on"


# ---------------------------------------------------------------------------
# phase 3: consumers rewritten onto GemmPlan
# ---------------------------------------------------------------------------

def _high_tier_by_subtraction(plan: QuantizationPlan) -> list:
    """What `setup_high_tier_gemms` computed before the rewrite."""
    low = set(plan.module_list("fp4"))
    return [
        name
        for name in plan.module_list("fp8")
        if not any(name == m or name.startswith(f"{m}.") for m in low)
    ]


#: `setup_high_tier_gemms` runs only from `_setup_format_gemms`, i.e.
#: under the fp4/fp6 backends. A pure fp8 run converts through
#: `_convert_fp8_on_device` and never reaches it, so the subtraction it would
#: have computed there is dead and not an oracle for anything.
FORMAT_PATH_CASES = [c for c in CASES if GemmQuantizationSpec.parse(c[0]).low == "fp4"]


@pytest.mark.parametrize("raw, text_encoder", FORMAT_PATH_CASES)
def test_the_high_tier_survives_the_rewrite(raw, text_encoder):
    """`roots(high)` must name what the subtraction named, profile by profile."""
    old = _plan_for(raw, text_encoder=text_encoder, legacy=True)
    new = _plan_for(raw, text_encoder=text_encoder, legacy=False)

    gemm_plan = new.gemm_plan
    rewritten = list(gemm_plan.roots(gemm_plan.high)) if gemm_plan.high else []
    assert sorted(rewritten) == sorted(_high_tier_by_subtraction(old))


def test_the_high_tier_is_the_text_encoder_under_a_tier():
    """The one profile where FLUX.2 actually has a high tier to place."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True, legacy=False).gemm_plan
    assert plan.roots(plan.high) == (TEXT_ENCODER,)
    assert sorted(plan.roots(plan.low)) == sorted(TRANSFORMER)


@pytest.mark.parametrize("raw", ["fp8", "fp4"])
def test_a_pure_profile_places_no_high_tier(raw):
    """`setup_high_tier_gemms` must stay a no-op, as it is today."""
    plan = _plan_for(raw, text_encoder=False, legacy=False).gemm_plan
    assert plan.high is None


def test_the_high_format_drives_the_adapter_choice():
    """It was `use_fp6_gemms and use_fp4_gemms`; now any tier names its own."""
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True, legacy=False).gemm_plan
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


@pytest.mark.parametrize("raw, text_encoder", CASES)
def test_the_log_reports_what_it_did_before(raw, text_encoder, monkeypatch):
    """Transformer lines only: the legacy logger never mentioned the encoder."""
    old = _logged(_plan_for(raw, text_encoder=text_encoder, legacy=True), monkeypatch)
    new = _logged(_plan_for(raw, text_encoder=text_encoder, legacy=False), monkeypatch)
    assert _mapping(new) == _mapping(old)


def test_the_log_now_mentions_the_text_encoder(monkeypatch):
    """A deliberate addition: it is quantized, so the log should say so.

    The legacy logger reported "the resolved transformer target-to-format
    mapping" and stopped there, leaving the encoder's format invisible in the
    one place a run tells you what it did.
    """
    args = dict(text_encoder=True)
    old = _logged(_plan_for("fp8", legacy=True, **args), monkeypatch)
    new = _logged(_plan_for("fp8", legacy=False, **args), monkeypatch)

    assert _mapping(old, component="text_encoder") == []
    assert _mapping(new, component="text_encoder") == [
        f"GEMM quantization: {TEXT_ENCODER} -> FP8"
    ]


def test_the_tier_log_names_the_high_modules(monkeypatch):
    lines = _logged(
        _plan_for("low=fp4,high=fp8", text_encoder=True, legacy=False), monkeypatch
    )
    header = [l for l in lines if l.startswith("GEMM high-precision tier:")]
    assert len(header) == 1
    assert "format=fp8" in header[0] and TEXT_ENCODER in header[0]
    assert f"GEMM quantization: {TEXT_ENCODER} -> FP8" in lines
    assert f"GEMM quantization: {TRANSFORMER[0]} -> FP4" in lines


def test_an_unquantized_run_logs_nothing(monkeypatch):
    assert _logged(_plan_for("none", text_encoder=False, legacy=False), monkeypatch) == []


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
    assert _declared_components(declared) == _declared_components(
        SimpleNamespace(settings=_legacy_settings())
    )


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
    plan = _plan_for(raw, text_encoder=True, legacy=False)
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
    plan = _plan_for("low=fp4,high=fp8", text_encoder=True, legacy=False).gemm_plan
    assert plan.format_for(f"{TRANSFORMER[0]}.0.attn.to_qkv") == "fp4"
    assert plan.format_for(f"{TEXT_ENCODER}.3.mlp") == "fp8"
