"""The single walk that replaced the format/remainder pass pair.

Drives `setup_gemm_quantization` with recording adapters, so the assertions are
about which converter saw which module, and on which side of the device move.
"""

from types import SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.loading import placement
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
    QuantizationLedger,
)
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)
from xfuser.model_executor.quant.targets import GemmTargets, Select

BLOCKS = ("transformer.transformer_blocks", "transformer.single_transformer_blocks")
TEXT_ENCODER = "text_encoder.model.language_model.layers"

TARGETS = GemmTargets(
    transformer=Select(modules=BLOCKS),
    text_encoder=Select(modules=(TEXT_ENCODER,)),
    keep_high=Select(modules=(TEXT_ENCODER,)),
)


#: The linear leaves the fake pipeline carries inside every block, chosen so a
#: declaration can name one by suffix and leave its neighbours alone.
LEAVES = ("attn.to_qkv", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
BLOCK_COUNT = 4


def _leaves_under(root):
    return tuple(
        f"{root}.{index}.{leaf}" for index in range(BLOCK_COUNT) for leaf in LEAVES
    )


ALL_LEAVES = tuple(
    leaf for root in BLOCKS + (TEXT_ENCODER,) for leaf in _leaves_under(root)
)


class _Recorder:
    """An adapter that records what it converted, and what its filter kept.

    Which subtree a converter is handed says little now that every converter is
    handed the declared roots and the plan decides each leaf, so the assertions
    are about `kept`: the leaves this converter would actually replace.
    """

    def __init__(self, name, *, before_device_move=False):
        self.name = name
        self.converts_before_device_move = before_device_move
        self.seen = []

    def convert_module(self, module, **kwargs):
        self.seen.append((module.path, kwargs))

    @property
    def kept(self):
        # A walk may start at a whole block list or at one block inside it, so
        # the leaves are matched against the pipeline rather than generated
        # from whatever path this converter happened to be handed.
        found = []
        for path, kwargs in self.seen:
            filter_fn = kwargs.get("filter_fn")
            for leaf in ALL_LEAVES:
                if not (leaf == path or leaf.startswith(f"{path}.")):
                    continue
                fqn = "" if leaf == path else leaf[len(path) + 1 :]
                if filter_fn is None or filter_fn(None, fqn):
                    found.append(leaf)
        return sorted(found)


def _pipe(paths):
    root = SimpleNamespace()
    for path in paths:
        node, walked = root, []
        for part in path.split("."):
            walked.append(part)
            if not hasattr(node, part):
                setattr(node, part, SimpleNamespace(path=".".join(walked)))
            node = getattr(node, part)
    return root


def _loader(raw, *, text_encoder=True, fp8_before_move=True, targets=TARGETS):
    spec = GemmQuantizationSpec.parse(raw)
    settings = SimpleNamespace(
        gemm_targets=targets,
        fp8_gemm_include_suffixes=None,
        fp8_precision_overrides=None,
        fp8_precision_override_suffixes=None,
    )
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        quantize_text_encoder=text_encoder,
        use_hybrid_gemm_schedule=False,
    )
    model = SimpleNamespace(
        settings=settings,
        config=config,
        # Blocks as well as block lists: a carve-out named by prefix makes the
        # walk start at one block, so it has to be reachable on the pipe.
        pipe=_pipe(
            tuple(
                f"{root}.{index}"
                for root in BLOCKS + (TEXT_ENCODER,)
                for index in range(BLOCK_COUNT)
            )
        ),
    )
    primary = _Recorder("format")
    blockwise = _Recorder("blockwise_fp8", before_device_move=fp8_before_move)
    backends = SimpleNamespace(
        format=primary,
        fp8=None,
        fp6=_Recorder("fp6"),
        blockwise_fp8=blockwise,
    )
    backends.adapter_for = lambda fmt: (
        backends.fp6
        if fmt == "fp6"
        else (
            (backends.fp8 or backends.blockwise_fp8)
            if fmt == "fp8"
            else backends.format
        )
    )
    return SimpleNamespace(
        model=model,
        backends=backends,
        quantization_plan=QuantizationPlan(model),
        quantization_ledger=QuantizationLedger(),
    )


@pytest.fixture(autouse=True)
def _no_real_descriptor(monkeypatch):
    monkeypatch.setattr(
        placement,
        "prepare_native_load",
        lambda *a, **k: SimpleNamespace(
            descriptor=SimpleNamespace(log_message=lambda: "")
        ),
    )
    monkeypatch.setattr(placement, "_is_cuda", lambda: False)
    # log() asks is_last_process(), which needs a distributed env
    monkeypatch.setattr(placement, "log", lambda *a, **k: None)


def _run(loader, **kwargs):
    placement.setup_gemm_quantization(
        loader, local_rank=0, offload_requested=False, **kwargs
    )


def _block_leaves():
    return sorted(leaf for root in BLOCKS for leaf in _leaves_under(root))


def test_each_format_goes_to_its_own_converter():
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    assert loader.backends.blockwise_fp8.kept == sorted(_leaves_under(TEXT_ENCODER))
    assert loader.backends.format.kept == _block_leaves()


def test_the_adapter_decides_which_side_of_the_device_move_it_runs_on():
    """The ordering is data now, not an if/elif around the `.to(cuda)`."""
    loader = _loader("low=fp4,high=fp8", fp8_before_move=True)
    _run(loader, before_device_move=True)
    assert loader.backends.blockwise_fp8.kept == sorted(_leaves_under(TEXT_ENCODER))
    assert loader.backends.format.kept == []

    loader = _loader("low=fp4,high=fp8", fp8_before_move=False)
    _run(loader, before_device_move=True)
    assert loader.backends.blockwise_fp8.kept == []
    assert loader.backends.format.kept == []
    _run(loader, before_device_move=False)
    assert loader.backends.blockwise_fp8.kept == sorted(_leaves_under(TEXT_ENCODER))
    assert loader.backends.format.kept == _block_leaves()


def test_one_format_sends_everything_to_one_converter():
    """Including the text encoder, and including what keep_high names: one
    format means the model's high/low preference has nothing to act on."""
    loader = _loader("fp4", text_encoder=True)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert loader.backends.format.kept == sorted(
        leaf for root in BLOCKS + (TEXT_ENCODER,) for leaf in _leaves_under(root)
    )
    assert loader.backends.blockwise_fp8.kept == []


def test_the_text_encoder_is_skipped_unless_the_run_asks():
    loader = _loader("low=fp4,high=fp8", text_encoder=False)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert loader.backends.blockwise_fp8.kept == []
    assert loader.backends.format.kept == _block_leaves()


def test_only_the_primary_converter_takes_the_hybrid_arguments():
    """The FP8 converter builds no per-step wrappers, so it takes no companion."""
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    _, fp8_kwargs = loader.backends.blockwise_fp8.seen[0]
    _, primary_kwargs = loader.backends.format.seen[0]
    assert "companion" not in fp8_kwargs
    # This run asked for no hybrid schedule, so not even the primary gets one.
    assert "companion" not in primary_kwargs
    # The carve-out patterns the primary converter used to be handed are the
    # filter's business now, so neither converter is given them.
    # The carve-out patterns the primary converter used to be handed no longer
    # exist as an argument anywhere; the filter owns that decision.
    assert "fp8_layers" not in primary_kwargs


def test_nothing_is_quantized_twice():
    """A module owned by one format is excluded from the other's walk."""
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    low = loader.backends.format.kept
    high = loader.backends.blockwise_fp8.kept
    assert low and high
    assert not set(low) & set(high)


def test_a_suffix_carve_out_reaches_the_placement_walk():
    """A carve-out named by suffix has no subtree of its own to walk from.

    Starting from what each format owns would never reach it, and it would
    quietly take the low format instead of the high one.
    """
    scattered = GemmTargets(
        transformer=Select(modules=("transformer.transformer_blocks",)),
        keep_high=Select(suffixes=("ff.net.2",)),
    )
    loader = _loader("low=fp4,high=fp8", targets=scattered, text_encoder=False)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    root = "transformer.transformer_blocks"
    assert loader.backends.blockwise_fp8.kept == sorted(
        f"{root}.{index}.ff.net.2" for index in range(BLOCK_COUNT)
    )
    assert loader.backends.format.kept == sorted(
        leaf for leaf in _leaves_under(root) if not leaf.endswith("ff.net.2")
    )


def test_a_block_prefix_carve_out_reaches_the_placement_walk():
    """Segment-aware, so naming block 3 must not take block 30 with it."""
    endpoints = GemmTargets(
        transformer=Select(modules=("transformer.transformer_blocks",)),
        keep_high=Select(prefixes=("transformer.transformer_blocks.3",)),
    )
    loader = _loader("low=fp4,high=fp8", targets=endpoints, text_encoder=False)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    root = "transformer.transformer_blocks"
    assert loader.backends.blockwise_fp8.kept == sorted(
        f"{root}.3.{leaf}" for leaf in LEAVES
    )
    assert loader.backends.format.kept == sorted(
        leaf for leaf in _leaves_under(root) if not leaf.startswith(f"{root}.3.")
    )


def test_an_unquantized_run_walks_nothing():
    loader = _loader("none")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert loader.backends.format.seen == []
    assert loader.backends.blockwise_fp8.seen == []


# ---------------------------------------------------------------------------
# the FSDP path: the same decision, per wrapped block
# ---------------------------------------------------------------------------

from xfuser.model_executor.models.runner_models.loading import shard  # noqa: E402


class _BlockRecorder(_Recorder):
    def convert_block(self, block, **kwargs):
        self.seen.append((block, kwargs))


def _shard_loader(raw, *, targets=TARGETS, text_encoder=True):
    loader = _loader(raw, text_encoder=text_encoder, targets=targets)
    loader.backends.format = _BlockRecorder("format")
    loader.backends.blockwise_fp8 = _BlockRecorder("blockwise_fp8")
    loader.backends.fp6 = _BlockRecorder("fp6")
    return loader


def _kept(recorder, block_path, leaves):
    """The leaves this converter's filter would actually touch."""
    if not recorder.seen:
        return []
    _, kwargs = recorder.seen[-1]
    return [leaf for leaf in leaves if kwargs["filter_fn"](None, leaf)]


def test_a_whole_block_goes_to_one_converter():
    loader = _shard_loader("low=fp4,high=fp8")
    fn = shard.build_block_quantize_fn(
        loader, "transformer", ["transformer_blocks"], local_rank=0
    )
    fn(object(), 3)
    path = "transformer.transformer_blocks.3"
    leaves = ["attn.to_qkv", "ff.net.0.proj"]
    assert len(loader.backends.format.seen) == 1
    assert _kept(loader.backends.format, path, leaves) == leaves
    # The other converter is still offered the block -- it simply owns no leaf
    # in it, which is the filter's answer rather than a target subtraction.
    assert _kept(loader.backends.blockwise_fp8, path, leaves) == []


def test_a_block_split_by_keep_high_stays_disjoint():
    """The case target arithmetic got wrong: block low, one submodule high."""
    split = GemmTargets(
        transformer=Select(modules=("transformer.transformer_blocks",)),
        keep_high=Select(modules=("transformer.transformer_blocks.3.attn",)),
    )
    loader = _shard_loader("low=fp4,high=fp8", targets=split, text_encoder=False)
    fn = shard.build_block_quantize_fn(
        loader, "transformer", ["transformer_blocks"], local_rank=0
    )
    fn(object(), 3)

    leaves = ["attn.to_qkv", "ff.net.0.proj"]
    path = "transformer.transformer_blocks.3"
    low = _kept(loader.backends.format, path, leaves)
    high = _kept(loader.backends.blockwise_fp8, path, leaves)

    assert high == ["attn.to_qkv"]
    assert low == ["ff.net.0.proj"]
    assert not set(low) & set(high)


def test_a_suffix_carve_out_survives_sharding():
    """The same declaration the placement walk honours, per wrapped block."""
    scattered = GemmTargets(
        transformer=Select(modules=("transformer.transformer_blocks",)),
        keep_high=Select(suffixes=("ff.net.2",)),
    )
    loader = _shard_loader("low=fp4,high=fp8", targets=scattered, text_encoder=False)
    fn = shard.build_block_quantize_fn(
        loader, "transformer", ["transformer_blocks"], local_rank=0
    )
    fn(object(), 2)

    path = "transformer.transformer_blocks.2"
    leaves = ["attn.to_qkv", "ff.net.2"]
    assert _kept(loader.backends.blockwise_fp8, path, leaves) == ["ff.net.2"]
    assert _kept(loader.backends.format, path, leaves) == ["attn.to_qkv"]


def test_an_untargeted_component_gets_no_callable():
    loader = _shard_loader("fp4", text_encoder=False)
    assert (
        shard.build_block_quantize_fn(loader, "vae", ["decoder"], local_rank=0) is None
    )


def test_an_unquantized_run_gets_no_callable():
    loader = _shard_loader("none")
    assert (
        shard.build_block_quantize_fn(
            loader, "transformer", ["transformer_blocks"], local_rank=0
        )
        is None
    )


# ---------------------------------------------------------------------------
# phase 4, step 3: the FSDP predicates ask the plan
# ---------------------------------------------------------------------------


def _backends_for(raw, *, text_encoder=True, targets=TARGETS):
    from xfuser.model_executor.models.runner_models.loading.backend_selection import (
        QuantizationBackends,
    )

    loader = _loader(raw, text_encoder=text_encoder, targets=targets)
    backends = QuantizationBackends(loader)
    loader.backends = backends
    return backends


def test_the_high_tier_is_what_the_primary_format_leaves():
    plan = _backends_for("low=fp4,high=fp8").loader.quantization_plan.gemm_plan
    assert set(plan.roots("fp8")) == {TEXT_ENCODER}
    assert set(plan.roots("fp4")) == set(BLOCKS)


@pytest.mark.parametrize("raw", ["fp8", "fp4"])
def test_a_pure_profile_puts_one_format_in_play(raw):
    """Nothing is held back, so only one converter is ever resolved."""
    assert _backends_for(raw)._formats_in_play() == (raw,)


def test_a_tier_puts_both_formats_in_play():
    assert _backends_for("low=fp4,high=fp8")._formats_in_play() == ("fp4", "fp8")


def test_no_high_tier_without_the_component_enabled():
    """FLUX.2 holds only its text encoder high; leave it out and nothing is."""
    backends = _backends_for("low=fp4,high=fp8", text_encoder=False)
    assert set(backends.loader.quantization_plan.gemm_plan.roots("fp8")) == set()


def test_a_narrowed_declaration_filters_inside_the_subtree():
    """`only` reaches the converter, not just format_for."""
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    narrowed = GemmTargets(
        transformer=Select(modules=BLOCKS, only=("attn.to_qkv",)),
    )
    # fp4 so the primary converter owns it; fp8 would route to blockwise.
    loader = _loader("fp4", text_encoder=False, targets=narrowed)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    assert sorted(p for p, _ in loader.backends.format.seen) == sorted(BLOCKS)
    _, kwargs = loader.backends.format.seen[0]
    keep = kwargs["filter_fn"]
    assert keep(None, "3.attn.to_qkv") is True
    assert keep(None, "3.attn.to_out.0") is False
