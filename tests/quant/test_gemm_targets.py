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
        ("low=none,high=fp8", "unknown GEMM quantization format"),
        ("low=fp4,high=fp4", "must differ"),
    ],
)
def test_a_nonsense_pair_is_still_refused(raw, message):
    with pytest.raises(ValueError, match=message):
        GemmQuantizationSpec.parse(raw)
