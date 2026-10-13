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
    """ "3." vs "30." is the trap absolute segment matching removes."""
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
        TARGETS,
        GemmQuantizationSpec.parse("fp8"),
        enable=("transformer", "text_encoder"),
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
    tier = resolve(RULES_TARGETS, GemmQuantizationSpec.parse("low=fp4,high=fp8"), enable=BOTH)
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
    return resolve(targets, GemmQuantizationSpec.parse(raw), sp_world_size=sp_world_size)


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
        transformer=Select(modules=("transformer.layers", "transformer.context_refiner")),
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
        keep_high=Select(prefixes=("transformer.blocks.0",), suffixes=("net.0.proj", "net.2")),
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

    header = [line for line in lines if line.startswith("GEMM high-precision tier:")]
    assert len(header) == 1
    assert "transformer.blocks.0" in header[0]
    assert "net.0.proj" in header[0] and "net.2" in header[0]
    assert "GEMM quantization: transformer.blocks -> FP4; selected layers -> FP8" in lines


# ---------------------------------------------------------------------------
# pinned components
# ---------------------------------------------------------------------------

#: What the text encoder's loaders can place, as `text_encoder_formats()`
#: reports it. Spelled out here so the resolver's behaviour is pinned without
#: importing the registry.
ENCODER = ("fp8",)
ENCODER_LEAF = "text_encoder.encoder.block.0.layer.0.SelfAttention.q"
BOTH = ("transformer", "text_encoder")


def _plan(profile, **kwargs):
    return resolve(
        TARGETS,
        GemmQuantizationSpec.parse(profile),
        enable=BOTH,
        component_formats={"text_encoder": ENCODER},
        **kwargs,
    )


@pytest.mark.parametrize("profile", ["fp8", "fp4", "fp6", "int8", "low=fp4,high=fp6"])
def test_a_pinned_component_takes_its_own_format_whatever_the_run_named(profile):
    """The encoder's loaders carry one format. The run names another and the
    encoder still gets that one -- in the plan, so every walk agrees rather
    than each route deciding for itself."""
    assert _plan(profile).format_for(ENCODER_LEAF) == "fp8"


@pytest.mark.parametrize("profile", ["fp8", "fp4", "low=fp4,high=fp6"])
def test_a_pin_leaves_the_rest_of_the_plan_alone(profile):
    """Only the component that needed it moves."""
    pinned = _plan(profile)
    plain = resolve(TARGETS, GemmQuantizationSpec.parse(profile), enable=BOTH)

    for path in (f"{BLOCKS}.20.attn.to_q", f"{BLOCKS}.0.attn.to_q"):
        assert pinned.format_for(path) == plain.format_for(path)


#: An encoder carved into `keep_high`, the way several runners declare theirs
#: so a tiered run leaves it at the better format.
CARVED = GemmTargets(
    transformer=Select(modules=(BLOCKS,)),
    text_encoder=Select(modules=("text_encoder.encoder.block",)),
    keep_high=Select(modules=("text_encoder.encoder.block",)),
)


def test_nothing_is_pinned_when_the_run_already_suits_the_component():
    """A pin that changes nothing would still put a format in play and a line
    in the log, so it is not recorded."""
    assert dict(_plan("fp8").pinned) == {}


def test_a_carve_out_the_loaders_can_place_needs_no_pin():
    """`keep_high` already puts this encoder on the high tier, and the high
    tier is a format its loaders have, so the pin has nothing to do."""
    plan = resolve(
        CARVED,
        GemmQuantizationSpec.parse("low=fp4,high=fp8"),
        enable=BOTH,
        component_formats={"text_encoder": ENCODER},
    )

    assert dict(plan.pinned) == {}
    assert plan.format_for(ENCODER_LEAF) == "fp8"


def test_a_pin_outranks_a_carve_out_the_loaders_cannot_place():
    """The same declaration under a profile whose high tier the encoder's
    loaders do not have. `keep_high` asks for FP6 and the pin refuses it: a
    model may say which half of a run a component belongs to, but not that it
    takes a format nothing can load it in."""
    plan = resolve(
        CARVED,
        GemmQuantizationSpec.parse("low=fp4,high=fp6"),
        enable=BOTH,
        component_formats={"text_encoder": ENCODER},
    )

    assert dict(plan.pinned) == {"text_encoder": "fp8"}
    assert plan.format_for(ENCODER_LEAF) == "fp8"
    assert plan.format_for(f"{BLOCKS}.0.attn.to_q") == "fp4", "the transformer keeps the tier it was given"


def test_a_pin_puts_its_format_in_play():
    """Whoever probes backends has to know the encoder's format is coming, or
    it refuses to store a format the run never named."""
    plan = _plan("fp4")

    assert plan.pinned == {"text_encoder": "fp8"}
    assert set(plan.formats_in_play) == {"fp4", "fp8"}
    assert plan.formats_in_play[0] == "fp4", "the low format still comes first"


def test_a_pinned_component_belongs_to_exactly_one_walk():
    """Two walks reaching the same leaf would convert it twice."""
    plan = _plan("low=fp4,high=fp6")
    root = "text_encoder.encoder.block"

    owners = [fmt for fmt in plan.formats_in_play if root in plan.roots(fmt)]
    assert owners == ["fp8"]
    assert plan.relative_to("text_encoder", plan.roots("fp8")) == ("encoder.block",)


def test_a_component_with_no_declared_restriction_is_never_pinned():
    """`component_formats` names the components whose loaders are limited;
    anything unnamed keeps the tier the declaration gives it."""
    plan = _plan("low=fp4,high=fp8")

    assert "transformer" not in plan.pinned
    assert plan.format_for(f"{BLOCKS}.0.attn.to_q") == "fp8"
    assert plan.format_for(f"{BLOCKS}.20.attn.to_q") == "fp4"


def test_a_run_that_quantizes_nothing_pins_nothing():
    """`--gemm_quantization none` places no format, so there is nothing for a
    pin to override and nothing to probe a backend for."""
    plan = _plan("none")

    assert dict(plan.pinned) == {}
    assert plan.formats_in_play == ()
    assert plan.format_for(ENCODER_LEAF) is None


def test_a_pin_prefers_a_run_format_its_route_can_place():
    """When the route grows a second format, a run naming it should get it
    rather than the route's first. Nothing declares two today, so this states
    the rule the resolver follows when something does."""
    plan = resolve(
        TARGETS,
        GemmQuantizationSpec.parse("low=fp4,high=fp6"),
        enable=BOTH,
        component_formats={"text_encoder": ("fp8", "fp6")},
    )

    assert plan.format_for(ENCODER_LEAF) == "fp6"


def test_the_log_names_a_pinned_component_and_why():
    """A pinned component belongs to neither tier, so both tier loops pass
    over it. A run that named FP4 and finds FP8 in its encoder has to be able
    to read why out of the log rather than out of the source."""
    from types import SimpleNamespace
    from xfuser.model_executor.models.runner_models.loading import quantization_plan

    plan = _plan("fp4")
    lines = []
    original = quantization_plan.log
    quantization_plan.log = lines.append
    try:
        model = SimpleNamespace(
            settings=SimpleNamespace(gemm_targets=TARGETS),
            config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        )
        quantization_plan.QuantizationPlan(model)._log_resolved_plan(plan)
    finally:
        quantization_plan.log = original

    named = [line for line in lines if "text_encoder.encoder.block" in line]
    assert len(named) == 1, f"the encoder is logged {len(named)} times: {lines}"
    assert "-> FP8" in named[0]
    assert "pinned" in named[0] and "fp4" in named[0]


@pytest.mark.parametrize("profile", ["fp4", "fp6", "int8", "low=fp4,high=fp6"])
def test_an_encoder_the_run_did_not_ask_for_is_left_alone(profile):
    """`--quantize_text_encoder` is what puts the encoder in the plan, and the
    pin only acts on a component that is in it.

    Without the flag the encoder is not quantized at all, and -- just as
    important -- its format never reaches the set of backends the run probes,
    so a machine that cannot store FP8 is never asked to.
    """
    plan = resolve(
        TARGETS,
        GemmQuantizationSpec.parse(profile),
        enable=("transformer",),
        component_formats={"text_encoder": ENCODER},
    )

    assert dict(plan.pinned) == {}
    assert plan.format_for(ENCODER_LEAF) is None
    assert plan.declared_roots(component="text_encoder") == ()
    assert "fp8" not in plan.formats_in_play
