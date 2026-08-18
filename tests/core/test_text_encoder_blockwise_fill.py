"""Routing text encoders through the blockwise disk fill.

The fill is worth being on: one block of the encoder is real at a time, on the rank that reads it,
instead of rank 0 holding the whole encoder on host before scattering it. Taking it needs the
encoder's live names mapped onto checkpoint keys, so these cover the decision and its consequences:
that the meta layout follows the mapping rather than the reverse, that fp8 is what benefits rather
than what falls back, that a refusal restores the layout the fallback needs, and that ranks never
split across the two paths.
"""

import weakref
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def meta_load():
    pytest.importorskip("torch", reason="PyTorch is required for meta-load tests")
    from xfuser.model_executor.models.runner_models.loading import meta_load

    return meta_load


class SingleRankWorld:
    rank_in_group = 0
    world_size = 1
    local_rank = 0

    def all_reduce(self, tensor):
        return tensor


class SplitWorld:
    """Two ranks whose all_reduce reports how many of them refused."""

    world_size = 2
    local_rank = 0

    def __init__(self, rank, total_refusals):
        self.rank_in_group = rank
        self.total_refusals = total_refusals

    def all_reduce(self, tensor):
        tensor.fill_(self.total_refusals)
        return tensor


class Encoder:
    """Stands in for a meta-built text encoder; only identity and swap-ability matter here."""

    def __init__(self, name="text_encoder"):
        self.name = name
        self.swapped = []


def make_loader(
    meta_load,
    *,
    wrap_attrs=("model.layers",),
    meta_text_encoders=("text_encoder",),
):
    from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
        QuantizationLedger,
    )

    model = SimpleNamespace(
        settings=SimpleNamespace(
            model_name="acme/encoder",
            fsdp_strategy={
                "transformer": {"wrap_attrs": ["blocks"]},
                "text_encoder": {"wrap_attrs": list(wrap_attrs)},
            },
        ),
        load_declaration=SimpleNamespace(
            meta_text_encoders=tuple(meta_text_encoders)
        ),
        quantization_ledger=QuantizationLedger(
            fp8_streaming_targets={"text_encoder.model.layers"}
        ),
    )
    loader = object.__new__(meta_load.ModelLoader)
    loader.model = model
    loader.load_declaration = model.load_declaration
    loader.quantization_ledger = model.quantization_ledger
    loader._blockwise_sources = weakref.WeakKeyDictionary()
    loader._te_routes = None
    return loader


def test_fsdp_strategy_does_not_auto_enroll_an_undeclared_encoder(meta_load):
    loader = make_loader(meta_load, meta_text_encoders=())

    assert loader._te_component_names() == []


def install(
    monkeypatch,
    meta_load,
    *,
    manifest,
    refusal,
    world=None,
    component=None,
    resolve_calls=None,
):
    monkeypatch.setattr(
        meta_load, "get_world_group", lambda: world or SingleRankWorld()
    )
    built = component if component is not None else Encoder()
    monkeypatch.setattr(
        meta_load.ModelLoader,
        "build_meta_component",
        lambda self, name, fp8=True: built,
    )

    def resolve(component, request):
        if resolve_calls is not None:
            resolve_calls.append(request)
        return manifest, refusal

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading."
        "text_encoder_adapter.resolve_transformers_manifest",
        resolve,
    )
    monkeypatch.setattr(
        meta_load.ModelLoader,
        "apply_meta_te_fp8",
        lambda self, component, name: component.swapped.append(name) or True,
    )
    monkeypatch.setattr(meta_load, "log", lambda *a, **k: None)
    return built


def test_a_mapped_encoder_is_routed_to_the_blockwise_fill(monkeypatch, meta_load):
    loader = make_loader(meta_load)
    manifest = object()
    encoder = install(monkeypatch, meta_load, manifest=manifest, refusal=None)

    kwargs, te_quant = loader.meta_te_kwargs()

    assert kwargs == {"text_encoder": encoder}
    assert te_quant is None
    assert loader.self_fills_from_disk(encoder)
    assert loader._blockwise_sources[encoder] is manifest


def test_a_mapped_encoder_is_left_bf16_so_the_fill_matches_the_checkpoint(
    monkeypatch, meta_load
):
    """The fp8 swap names tensors no bf16 checkpoint has, so it must not happen on this path."""
    loader = make_loader(meta_load)
    encoder = install(monkeypatch, meta_load, manifest=object(), refusal=None)

    loader.meta_te_kwargs()

    assert encoder.swapped == []


def test_a_refused_encoder_is_swapped_to_fp8_for_the_broadcast_fallback(
    monkeypatch, meta_load
):
    """The fallback broadcasts rank0's fp8 tensors, so the peer layout has to be fp8 to match."""
    loader = make_loader(meta_load)
    encoder = install(
        monkeypatch, meta_load, manifest=None, refusal="keys need a fused split"
    )

    kwargs, _ = loader.meta_te_kwargs()

    assert kwargs == {"text_encoder": encoder}
    assert encoder.swapped == ["text_encoder"]
    assert not loader.self_fills_from_disk(encoder)


def test_an_encoder_with_no_declared_blocks_is_refused_without_reading_the_checkpoint(
    monkeypatch, meta_load
):
    loader = make_loader(meta_load, wrap_attrs=())
    calls = []
    install(
        monkeypatch,
        meta_load,
        manifest=object(),
        refusal=None,
        resolve_calls=calls,
    )

    loader.meta_te_kwargs()

    assert calls == []
    assert not loader.will_fill_blockwise("text_encoder")


def test_one_rank_refusing_makes_every_rank_refuse(monkeypatch, meta_load):
    """The two paths run different collectives, so a split would hang rather than fail."""
    loader = make_loader(meta_load)
    encoder = install(
        monkeypatch,
        meta_load,
        manifest=object(),
        refusal=None,
        world=SplitWorld(rank=0, total_refusals=1),
    )

    loader.meta_te_kwargs()

    assert not loader.will_fill_blockwise("text_encoder")
    assert encoder.swapped == ["text_encoder"]


def test_unanimous_agreement_still_takes_the_blockwise_fill(monkeypatch, meta_load):
    loader = make_loader(meta_load)
    manifest = object()
    encoder = install(
        monkeypatch,
        meta_load,
        manifest=manifest,
        refusal=None,
        world=SplitWorld(rank=1, total_refusals=0),
    )

    loader.meta_te_kwargs()

    assert loader.will_fill_blockwise("text_encoder")
    assert loader._blockwise_sources[encoder] is manifest


def test_a_refusing_rank_reports_its_own_reason(monkeypatch, meta_load):
    """Its reason is the actionable one; a rank that only inherited the refusal says so instead."""
    loader = make_loader(meta_load)
    install(
        monkeypatch,
        meta_load,
        manifest=None,
        refusal="no transformers checkpoint to map",
        world=SplitWorld(rank=1, total_refusals=1),
    )

    loader.te_blockwise_routes()

    assert loader._te_routes["text_encoder"][2] == "no transformers checkpoint to map"


def test_a_rank_that_could_map_it_says_why_it_is_falling_back_anyway(
    monkeypatch, meta_load
):
    loader = make_loader(meta_load)
    install(
        monkeypatch,
        meta_load,
        manifest=object(),
        refusal=None,
        world=SplitWorld(rank=0, total_refusals=1),
    )

    refusal = loader.te_blockwise_routes()["text_encoder"][2]

    assert "1 of 2 ranks" in refusal
    assert "this rank could" in refusal


def test_the_route_is_decided_once_so_the_plan_and_the_load_cannot_disagree(
    monkeypatch, meta_load
):
    """The fp8 plan asks before the load does; a second resolve could answer differently."""
    loader = make_loader(meta_load)
    calls = []
    install(
        monkeypatch,
        meta_load,
        manifest=object(),
        refusal=None,
        resolve_calls=calls,
    )

    assert loader.will_fill_blockwise("text_encoder")
    loader.meta_te_kwargs()

    assert len(calls) == 1


def test_an_encoder_that_cannot_be_rebuilt_leaves_the_caller_its_normal_load(
    monkeypatch, meta_load
):
    loader = make_loader(meta_load)
    install(monkeypatch, meta_load, manifest=object(), refusal=None)
    monkeypatch.setattr(
        meta_load.ModelLoader,
        "build_meta_component",
        lambda self, name, fp8=True: None,
    )

    assert loader.meta_te_kwargs() is None
    assert not loader.will_fill_blockwise("text_encoder")


def test_the_fp8_swap_is_applied_only_where_the_component_has_targets(meta_load):
    """apply_meta_te_fp8 reports whether it changed anything, which is what defers it safely."""
    loader = make_loader(meta_load)

    assert loader._meta_te_fp8_targets("text_encoder") == ("model.layers",)
    assert loader._meta_te_fp8_targets("text_encoder_2") == ()
    assert loader.apply_meta_te_fp8(object(), "text_encoder_2") is False
