"""Declared GEMM targets, and how a run binds formats to them.

No GPU, no model, no vendor library: this is the declaration and the resolver.
"""

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.quant.targets import (
    GemmTargets,
    Select,
    resolve,
)

BLOCKS = "transformer.blocks"
EDGE = tuple(f"{BLOCKS}.{i}" for i in (0, 1, 38, 39))

TARGETS = GemmTargets(
    transformer=Select(modules=(BLOCKS,)),
    keep_high=Select(prefixes=EDGE),
    text_encoder=Select(modules=("text_encoder.encoder.block",)),
)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def test_a_module_matches_itself_and_its_descendants():
    select = Select(modules=(BLOCKS,))
    assert select.matches(BLOCKS)
    assert select.matches(f"{BLOCKS}.7.attn.to_qkv")


def test_matching_respects_segment_boundaries():
    """A bare startswith would take "transformer.blocks_extra" too, quantizing
    a module the model never named."""
    select = Select(modules=(BLOCKS,))
    assert not select.matches("transformer.blocks_extra")
    assert not select.matches("transformer.blocks_extra.0")


def test_suffixes_match_whole_segments():
    select = Select(suffixes=("attn.to_qkv",))
    assert select.matches("transformer.blocks.7.attn.to_qkv")
    assert not select.matches("transformer.blocks.7.xattn.to_qkv")


def test_prefixes_do_not_need_a_trailing_dot_to_be_exact():
    """"3." vs "30." is the trap absolute segment matching removes."""
    select = Select(prefixes=(f"{BLOCKS}.3",))
    assert select.matches(f"{BLOCKS}.3.attn.to_qkv")
    assert not select.matches(f"{BLOCKS}.30.attn.to_qkv")


def test_keep_high_must_select_within_the_targets():
    with pytest.raises(ValueError, match="outside the target set"):
        GemmTargets(
            transformer=Select(modules=(BLOCKS,)),
            keep_high=Select(modules=("text_encoder.layers",)),
        )


def test_keep_high_may_name_any_targeted_component():
    targets = GemmTargets(
        transformer=Select(modules=(BLOCKS,)),
        text_encoder=Select(modules=("text_encoder.layers",)),
        keep_high=Select(modules=("text_encoder.layers",)),
    )
    assert targets.keep_high.matches("text_encoder.layers.3.mlp")


# ---------------------------------------------------------------------------
# resolving
# ---------------------------------------------------------------------------

def test_one_format_quantizes_every_target():
    """`--gemm_quantization fp8`: the high/low split has nothing to say."""
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("fp8"))
    assert plan.format_for(f"{BLOCKS}.7.attn.to_qkv") == "fp8"
    assert plan.format_for(f"{BLOCKS}.0.attn.to_qkv") == "fp8"


def test_two_formats_split_the_targets():
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp8"))
    assert plan.format_for(f"{BLOCKS}.7.attn.to_qkv") == "fp4"
    assert plan.format_for(f"{BLOCKS}.0.attn.to_qkv") == "fp8"
    assert plan.format_for(f"{BLOCKS}.39.ff.net.2") == "fp8"


def test_the_split_follows_the_formats_the_run_names():
    """The model declared no format, so a different pair needs no model change."""
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp6"))
    assert plan.format_for(f"{BLOCKS}.7.attn.to_qkv") == "fp4"
    assert plan.format_for(f"{BLOCKS}.0.attn.to_qkv") == "fp6"


def test_none_quantizes_nothing():
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("none"))
    assert not plan.quantizes
    assert plan.format_for(f"{BLOCKS}.0.attn.to_qkv") is None


def test_modules_outside_the_targets_are_left_alone():
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("fp8"))
    assert plan.format_for("transformer.norm_out") is None


def test_the_text_encoder_is_quantized_only_when_asked():
    path = "text_encoder.encoder.block.3.mlp"
    assert resolve(TARGETS, GemmQuantizationSpec.parse("fp8")).format_for(path) is None

    plan = resolve(
        TARGETS, GemmQuantizationSpec.parse("fp8"), enable=("transformer", "text_encoder")
    )
    assert plan.format_for(path) == "fp8"


def test_the_text_encoder_follows_the_same_rule_as_everything_else():
    """No component gets a precision rule of its own: low unless carved out."""
    tiered = GemmQuantizationSpec.parse("low=fp4,high=fp8")
    path = "text_encoder.encoder.block.3.mlp"

    plan = resolve(TARGETS, tiered, enable=("transformer", "text_encoder"))
    assert plan.format_for(path) == "fp4"

    held = GemmTargets(
        transformer=TARGETS.transformer,
        text_encoder=TARGETS.text_encoder,
        keep_high=Select(modules=("text_encoder.encoder.block",)),
    )
    plan = resolve(held, tiered, enable=("transformer", "text_encoder"))
    assert plan.format_for(path) == "fp8"


def test_keep_high_is_inert_while_a_component_is_not_enabled():
    """Carving it out says where it sits, not that it should be quantized."""
    held = GemmTargets(
        transformer=TARGETS.transformer,
        text_encoder=TARGETS.text_encoder,
        keep_high=Select(modules=("text_encoder.encoder.block",)),
    )
    plan = resolve(held, GemmQuantizationSpec.parse("low=fp4,high=fp8"))
    assert plan.format_for("text_encoder.encoder.block.3.mlp") is None
    assert "text_encoder.encoder.block" not in plan.roots("fp8")


def test_roots_report_the_declared_paths_per_format():
    """Consumers that walk a subtree need the declared roots, not every leaf."""
    plan = resolve(TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp8"))
    assert BLOCKS in plan.roots("fp4")
    assert set(EDGE) <= set(plan.roots("fp8"))


def test_resolving_does_not_touch_the_declaration():
    """The model's targets stay readable next to what the run decided."""
    before = TARGETS
    resolve(TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp8"))
    assert TARGETS == before


def test_a_component_is_quantized_only_when_enabled():
    """The transformer is not special: name it or it sits out too."""
    tiered = GemmQuantizationSpec.parse("low=fp4,high=fp8")
    block = f"{BLOCKS}.7.attn.to_qkv"

    assert resolve(TARGETS, tiered, enable=()).format_for(block) is None
    assert resolve(TARGETS, tiered, enable=("text_encoder",)).format_for(block) is None
    assert resolve(TARGETS, tiered, enable=("transformer",)).format_for(block) == "fp4"


def test_an_unknown_component_is_a_typo_not_a_silent_no_op():
    with pytest.raises(ValueError, match="unknown GEMM component"):
        resolve(TARGETS, GemmQuantizationSpec.parse("fp8"), enable=("text_enocder",))


def test_roots_narrow_to_one_component_for_loaders_that_split_them():
    plan = resolve(
        TARGETS,
        GemmQuantizationSpec.parse("fp8"),
        enable=("transformer", "text_encoder"),
    )
    assert plan.roots("fp8", component="text_encoder") == ("text_encoder.encoder.block",)
    assert BLOCKS in plan.roots("fp8", component="transformer")
    assert plan.roots("fp8", component="vae") == ()


# ---------------------------------------------------------------------------
# the rules, stated directly
# ---------------------------------------------------------------------------

RULES_TARGETS = GemmTargets(
    transformer=Select(modules=("transformer.blocks",)),
    text_encoder=Select(modules=("text_encoder.layers",)),
    keep_high=Select(
        modules=("transformer.blocks.0", "text_encoder.layers.0"),
    ),
)
BOTH = ("transformer", "text_encoder")
PLAIN = ("transformer.blocks.7.mlp", "text_encoder.layers.7.mlp")
HELD = ("transformer.blocks.0.mlp", "text_encoder.layers.0.mlp")


@pytest.mark.parametrize("fmt", ["fp8", "fp4", "fp6", "int8"])
def test_one_format_quantizes_everything_declared(fmt):
    """`--gemm_quantization <fmt>`: every target takes it, keep_high included."""
    plan = resolve(RULES_TARGETS, GemmQuantizationSpec.parse(fmt), enable=BOTH)
    for path in PLAIN + HELD:
        assert plan.format_for(path) == fmt


@pytest.mark.parametrize("low, high", [("fp4", "fp8"), ("fp4", "fp6")])
def test_two_formats_split_along_keep_high(low, high):
    """Everything takes low, minus keep_high, which takes high."""
    plan = resolve(
        RULES_TARGETS,
        GemmQuantizationSpec.parse(f"low={low},high={high}"),
        enable=BOTH,
    )
    for path in PLAIN:
        assert plan.format_for(path) == low
    for path in HELD:
        assert plan.format_for(path) == high


def test_the_text_encoder_obeys_the_same_two_rules():
    """Stated separately because it is the component that used to differ."""
    pure = resolve(RULES_TARGETS, GemmQuantizationSpec.parse("fp4"), enable=BOTH)
    tier = resolve(
        RULES_TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp8"), enable=BOTH
    )
    assert pure.format_for("text_encoder.layers.0.mlp") == "fp4"
    assert pure.format_for("text_encoder.layers.7.mlp") == "fp4"
    assert tier.format_for("text_encoder.layers.0.mlp") == "fp8"
    assert tier.format_for("text_encoder.layers.7.mlp") == "fp4"


@pytest.mark.parametrize("low, high", [("fp4", "fp6"), ("fp6", "fp8"), ("fp4", "int8")])
def test_any_sensible_pair_tiers(low, high):
    """Hybrid is generic: the resolver never reads a format name."""
    plan = resolve(
        RULES_TARGETS,
        GemmQuantizationSpec.parse(f"low={low},high={high}"),
        enable=BOTH,
    )
    assert plan.format_for(PLAIN[0]) == low
    assert plan.format_for(HELD[0]) == high


@pytest.mark.parametrize(
    "raw, message",
    [
        # `none` is a known format, just not one a tier can name, and the
        # refusal says that rather than calling it a typo.
        ("low=none,high=fp8", "a GEMM tier names a format to quantize to"),
        ("low=fp8,high=none", "a GEMM tier names a format to quantize to"),
        ("low=fp9,high=fp8", "unknown GEMM quantization format"),
        ("low=fp4,high=fp4", "must differ"),
    ],
)
def test_a_nonsense_pair_is_still_refused(raw, message):
    with pytest.raises(ValueError, match=message):
        GemmQuantizationSpec.parse(raw)


# ---------------------------------------------------------------------------
# `only`: quantizing part of a block
# ---------------------------------------------------------------------------

PARTIAL = GemmTargets(
    transformer=Select(
        modules=("transformer.blocks",),
        only=("attn.to_qkv", "ff.net.0.proj"),
    ),
)


def test_only_narrows_the_selection_to_named_leaves():
    plan = resolve(PARTIAL, GemmQuantizationSpec.parse("fp8"))
    assert plan.format_for("transformer.blocks.3.attn.to_qkv") == "fp8"
    assert plan.format_for("transformer.blocks.3.ff.net.0.proj") == "fp8"
    assert plan.format_for("transformer.blocks.3.attn.to_out.0") is None
    assert plan.format_for("transformer.blocks.3.ff.net.2") is None


def test_only_still_requires_the_subtree():
    """It narrows what is already selected; it does not select on its own."""
    plan = resolve(PARTIAL, GemmQuantizationSpec.parse("fp8"))
    assert plan.format_for("text_encoder.layers.0.attn.to_qkv") is None


def test_only_leaves_the_declared_roots_intact():
    """Consumers still walk the subtree; the narrowing applies inside it."""
    plan = resolve(PARTIAL, GemmQuantizationSpec.parse("fp8"))
    assert plan.roots("fp8") == ("transformer.blocks",)


def test_only_composes_with_keep_high():
    targets = GemmTargets(
        transformer=Select(
            modules=("transformer.blocks",),
            only=("attn.to_qkv", "ff.net.0.proj"),
        ),
        keep_high=Select(modules=("transformer.blocks.0",)),
    )
    plan = resolve(targets, GemmQuantizationSpec.parse("low=fp4,high=fp8"))
    assert plan.format_for("transformer.blocks.0.attn.to_qkv") == "fp8"
    assert plan.format_for("transformer.blocks.7.attn.to_qkv") == "fp4"
    # narrowed out of the target set entirely, in either tier
    assert plan.format_for("transformer.blocks.0.attn.to_out.0") is None
    assert plan.format_for("transformer.blocks.7.attn.to_out.0") is None


# ---------------------------------------------------------------------------
# short_sequence: a kernel's shape limit, declared rather than branched on
# ---------------------------------------------------------------------------

REFINERS = GemmTargets(
    transformer=Select(
        modules=(
            "transformer.layers",
            "transformer.noise_refiner",
            "transformer.context_refiner",
        )
    ),
    short_sequence=Select(modules=("transformer.context_refiner",)),
)


def _resolved(raw, *, sp_world_size, targets=REFINERS):
    return resolve(
        targets, GemmQuantizationSpec.parse(raw), sp_world_size=sp_world_size
    )


def test_a_short_module_keeps_its_format_without_sequence_parallelism():
    """One rank sees the whole sequence, so M is whatever the caption is."""
    plan = _resolved("int8", sp_world_size=1)
    assert plan.format_for("transformer.context_refiner.0.attn.to_q") == "int8"


def test_sequence_parallelism_drops_a_short_module_from_a_floored_format():
    plan = _resolved("int8", sp_world_size=8)
    assert plan.format_for("transformer.context_refiner.0.attn.to_q") is None
    # its neighbours are untouched: the limit is about this module's M
    assert plan.format_for("transformer.noise_refiner.0.attn.to_q") == "int8"
    assert plan.format_for("transformer.layers.0.attn.to_q") == "int8"


def test_a_format_without_a_floor_keeps_the_short_module_under_sp():
    """FP8 has no minimum M, so the same module quantizes fine."""
    plan = _resolved("fp8", sp_world_size=8)
    assert plan.format_for("transformer.context_refiner.0.attn.to_q") == "fp8"


def test_the_dropped_module_is_gone_from_every_consumer():
    """Narrowed at resolve time, so roots and walks agree with format_for."""
    plan = _resolved("int8", sp_world_size=8)
    assert "transformer.context_refiner" not in plan.declared_roots()
    assert "transformer.context_refiner" not in plan.roots("int8")


def test_a_tiered_run_asks_the_format_the_module_would_have_taken():
    """Held high at fp8, the floor never applies however low the run goes."""
    targets = GemmTargets(
        transformer=Select(
            modules=("transformer.layers", "transformer.context_refiner")
        ),
        keep_high=Select(modules=("transformer.context_refiner",)),
        short_sequence=Select(modules=("transformer.context_refiner",)),
    )
    plan = _resolved("low=int8,high=fp8", sp_world_size=8, targets=targets)
    assert plan.format_for("transformer.context_refiner.0.attn.to_q") == "fp8"
    assert plan.format_for("transformer.layers.0.attn.to_q") == "int8"


def test_short_sequence_must_name_a_declared_target():
    with pytest.raises(ValueError, match="not declared targets"):
        GemmTargets(
            transformer=Select(modules=("transformer.layers",)),
            short_sequence=Select(modules=("transformer.context_refiner",)),
        )


def test_short_sequence_refuses_to_carve_inside_a_target():
    with pytest.raises(ValueError, match="names whole targets"):
        GemmTargets(
            transformer=Select(modules=("transformer.layers",)),
            short_sequence=Select(suffixes=("attn.to_q",)),
        )


@pytest.mark.parametrize("raw", ["fp8", "fp4", "fp6", "low=fp4,high=fp8"])
def test_only_narrows_every_format_the_same_way(raw):
    """A model names the leaves it wants quantized, not the leaves one format
    quantizes. The legacy primary-format walk ignored the narrowing while every
    other walk applied it; the plan has one answer for all of them."""
    targets = GemmTargets(
        transformer=Select(
            modules=("transformer.transformer_blocks",),
            only=("attn.to_qkv", "ff.net.0.proj"),
        ),
    )
    plan = resolve(targets, GemmQuantizationSpec.parse(raw))
    block = "transformer.transformer_blocks.7"
    assert plan.format_for(f"{block}.attn.to_qkv") is not None
    assert plan.format_for(f"{block}.ff.net.0.proj") is not None
    for excluded in ("attn.to_out.0", "ff.net.2", "adaln_proj.linear"):
        assert plan.format_for(f"{block}.{excluded}") is None


def test_the_log_names_a_suffix_carve_out():
    """Wan2.2-TI2V holds two feed-forward leaves in every block at the better
    format. They have no module path, so a roots-only log leaves the one place
    a run says what it did silently wrong about them."""
    from types import SimpleNamespace
    from xfuser.model_executor.models.runner_models.loading import quantization_plan

    targets = GemmTargets(
        transformer=Select(modules=("transformer.blocks",)),
        keep_high=Select(
            prefixes=("transformer.blocks.0",), suffixes=("net.0.proj", "net.2")
        ),
    )
    plan = resolve(targets, GemmQuantizationSpec.parse("low=fp4,high=fp8"))

    lines = []
    original = quantization_plan.log
    quantization_plan.log = lines.append
    try:
        model = SimpleNamespace(
            settings=SimpleNamespace(gemm_targets=targets),
            config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        )
        quantization_plan.QuantizationPlan(model)._log_resolved_plan(plan)
    finally:
        quantization_plan.log = original

    header = [l for l in lines if l.startswith("GEMM high-precision tier:")]
    assert len(header) == 1
    assert "transformer.blocks.0" in header[0]
    assert "net.0.proj" in header[0] and "net.2" in header[0]
    assert "GEMM quantization: transformer.blocks -> FP4; selected layers -> FP8" in lines
