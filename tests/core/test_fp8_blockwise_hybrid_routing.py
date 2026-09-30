"""Routing tests for FP8-only blocks inside FP4/hybrid materialization."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import xFuserModel
from xfuser.model_executor.models.runner_models.loading import (
    placement,
    shard,
    transformer_load,
)
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
    QuantizationLedger,
)
from xfuser.model_executor.models.runner_models.loading import fp8_backends
from xfuser.model_executor.models.runner_models.loading.contracts import (
    MaterializationMode,
    QuantizationBackend,
    QuantizationFormat,
)
from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
    Fp8BackendCapabilities,
    select_blockwise_fp8_backend,
)


def backends(model):
    """The quantization backend selector under test, bound to a fake model."""
    from xfuser.model_executor.models.runner_models.loading.backend_selection import (
        QuantizationBackends,
    )

    loader = runtime(model)
    selected = QuantizationBackends(loader)
    for runtime_name, fixture_name in (
        ("fp8", "fp8_backend"),
        ("format", "format_backend"),
        ("blockwise_fp8", "blockwise_fp8_backend"),
    ):
        if hasattr(model, fixture_name):
            selected.__dict__[runtime_name] = getattr(model, fixture_name)
    return selected


class _StubPlan:
    """The plan a legacy-shaped fixture describes.

    These fixtures declare a per-format list each, which is how models used to
    declare. The equivalent `gemm_targets` targets their union and holds at the
    better format whatever the low list leaves out -- the same translation the
    models themselves went through -- so the fixtures keep describing one model
    while the code under test reads a resolved plan.
    """

    def __init__(self, model):
        self.model = model

    def _lists(self):
        listed = getattr(getattr(self.model, "fp8", None), "module_list", None)
        fp8 = list(listed()) if listed is not None else []
        for name in ("fp4", "int8", "fp6"):
            listed = getattr(self.model.settings, f"{name}_gemm_module_list", None)
            if listed:
                return name, list(listed), fp8
        return "fp8", fp8, fp8

    @property
    def gemm_plan(self):
        from xfuser.config.gemm import GemmQuantizationSpec
        from xfuser.model_executor.quant.targets import (
            GemmTargets,
            Select,
            resolve,
        )

        settings = self.model.settings
        low_name, low, fp8 = self._lists()
        if not (low or fp8):
            return None

        held = [t for t in fp8 if not any(_covers(m, t) for m in low)]
        prefixes = tuple(
            f"{target}.{entry.rstrip('.')}"
            for entry in (getattr(settings, "fp8_precision_overrides", None) or ())
            for target in low
        )
        suffixes = tuple(
            entry.lstrip(".")
            for entry in (
                getattr(settings, "fp8_precision_override_suffixes", None) or ()
            )
        )
        keep_high = Select(modules=tuple(held), prefixes=prefixes, suffixes=suffixes)

        high_name = None
        if low_name != "fp8" and (held or prefixes or suffixes):
            high_name = "fp6" if getattr(
                self.model.config, "use_fp6_gemms", False
            ) else "fp8"
        spec = (
            GemmQuantizationSpec(low_name, high_name)
            if high_name
            else GemmQuantizationSpec(low_name)
        )

        targets = GemmTargets(
            transformer=Select(
                modules=tuple(dict.fromkeys(low + fp8)),
                only=tuple(
                    getattr(settings, "fp8_gemm_include_suffixes", None) or ()
                ),
            ),
            keep_high=keep_high if high_name else Select(),
        )
        return resolve(targets, spec)

    def module_list(self, format_name="fp8"):
        if format_name == "fp8":
            return list(self.model.fp8.module_list())
        return list(
            getattr(
                self.model.settings,
                f"{format_name}_gemm_module_list",
                (),
            )
            or ()
        )

    def targets_for(self, component_name, format_name="fp8"):
        if format_name == "fp8" and hasattr(self.model.fp8, "targets_for"):
            return self.model.fp8.targets_for(component_name)
        prefix = f"{component_name}."
        return [
            "" if target == component_name else target[len(prefix) :]
            for target in self.module_list(format_name)
            if target == component_name or target.startswith(prefix)
        ]


def _covers(owner, path):
    return not owner or path == owner or path.startswith(f"{owner}.")


def runtime(model):
    """Wrap a legacy-shaped test fixture in the loader-owned runtime surface."""
    plan = _StubPlan(model)
    backends = SimpleNamespace(
        fp8=getattr(model, "fp8_backend", None),
        format=getattr(model, "format_backend", None),
        fp6=getattr(model, "fp6_backend", None),
        blockwise_fp8=getattr(model, "blockwise_fp8_backend", None),
        format_targets_for=lambda name: plan.targets_for(name, "fp4"),
    )
    backends.adapter_for = lambda format_name: (
        backends.fp6
        if format_name == "fp6"
        else (backends.fp8 or backends.blockwise_fp8)
        if format_name == "fp8"
        else backends.format
    )
    return SimpleNamespace(
        model=model,
        load_contract=getattr(model, "load_contract", None),
        quantization_plan=plan,
        quantization_ledger=getattr(model, "quantization_ledger", QuantizationLedger()),
        backends=backends,
        fill_eager_transformers=lambda: None,
        replicated_broadcast_load=getattr(
            model, "_replicated_broadcast_load", lambda: False
        ),
        broadcast_fill_replicated=lambda offload: None,
    )

class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def convert_block(self, block, *, device, filter_fn=None):
        self.calls.append((block, device))


class RecordingFormatAdapter:
    def __init__(self):
        self.calls = []

    def convert_block(self, block, **kwargs):
        self.calls.append((block, kwargs))


class FilterRecordingFp8Adapter:
    backend = QuantizationBackend.TORCHAO

    def __init__(self):
        self.calls = []

    def convert_block(self, block, *, device, filter_fn):
        self.calls.append((block, device, filter_fn))


def _hybrid_model(
    *,
    adapter,
    format_adapter=None,
    overrides=(),
    suffixes=None,
    fp8_include_suffixes=None,
):
    return SimpleNamespace(
        config=SimpleNamespace(
            gemm_formats=frozenset({"fp4"}),
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=["transformer.blocks"],
            fp8_precision_overrides=overrides,
            fp8_precision_override_suffixes=suffixes,
            fp8_gemm_include_suffixes=fp8_include_suffixes,
            int8_gemm_module_list=None,
        ),
        fp8=SimpleNamespace(
            module_list=lambda: [
                "transformer.blocks",
                "transformer_2.blocks",
            ]
        ),
        blockwise_fp8_backend=adapter,
        format_backend=format_adapter or RecordingFormatAdapter(),
    )


def test_wan_fp8_only_second_transformer_uses_blockwise_backend(monkeypatch):
    adapter = RecordingAdapter()
    model = _hybrid_model(adapter=adapter)

    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer_2", ["blocks"], local_rank=3
    )
    block = object()
    quantize(block, 0)

    assert adapter.calls == [(block, "cuda:3")]
    assert model.format_backend.calls == []


def test_blockwise_fp8_only_target_honors_include_suffixes():
    adapter = FilterRecordingFp8Adapter()
    model = _hybrid_model(
        adapter=adapter,
        fp8_include_suffixes=("attn.to_qkv", "ff.net.0.proj"),
    )

    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer_2", ["blocks"], local_rank=0
    )
    quantize(object(), 0)
    filter_fn = adapter.calls[0][2]

    assert filter_fn(object(), "attn.to_qkv")
    assert filter_fn(object(), "ff.net.0.proj")
    assert not filter_fn(object(), "attn.to_out.0")


def test_native_fp8_streaming_is_disabled_for_suffix_restricted_targets(
    monkeypatch,
):
    observed = {}
    monkeypatch.setattr(
        fp8_backends,
        "prepare_native_transformer_fp8_load",
        lambda _adapter, **kwargs: observed.update(kwargs),
    )
    from xfuser.config.gemm import GemmQuantizationSpec
    from xfuser.model_executor.quant.targets import GemmTargets, Select, resolve

    narrowed = resolve(
        GemmTargets(
            transformer=Select(
                modules=("transformer.blocks",), only=("attn.to_qkv",)
            )
        ),
        GemmQuantizationSpec.parse("fp8"),
    )

    transformer_load._prepare_native_load(
        SimpleNamespace(settings=SimpleNamespace(gemm_targets=None)),
        SimpleNamespace(format=QuantizationFormat.FP8),
        "transformer",
        ("blocks",),
        True,
        lambda: object(),
        plan=narrowed,
    )

    assert observed["stream_quant"] is False


def test_native_int8_does_not_receive_fp4_precision_overrides(monkeypatch):
    observed = {}

    def prepare(_adapter, **kwargs):
        observed.update(kwargs)

    from xfuser.model_executor.models.runner_models.loading import format_backends

    monkeypatch.setattr(
        format_backends,
        "prepare_native_transformer_format_load",
        prepare,
    )
    model = SimpleNamespace(
        settings=SimpleNamespace(gemm_targets=None),
        config=SimpleNamespace(use_hybrid_gemm_schedule=True),
    )

    transformer_load._prepare_native_load(
        model,
        SimpleNamespace(format=QuantizationFormat.INT8),
        "transformer",
        ("blocks",),
        True,
        lambda: object(),
    )

    # Carve-out patterns are not a native-load argument at all any more, and
    # the hybrid schedule belongs to the FP4 family alone.
    assert "precision_prefixes" not in observed
    assert "precision_suffixes" not in observed
    assert observed["hybrid"] is False


def test_precision_overrides_are_owned_by_the_high_format(monkeypatch):
    """They used to be an argument to the FP4 converter, which carved them out
    itself. They are a `keep_high` carve-out now, so the FP8 converter takes
    those leaves and the FP4 filter declines them."""
    fp8_adapter = FilterRecordingFp8Adapter()
    format_adapter = RecordingFormatAdapter()
    model = _hybrid_model(
        adapter=fp8_adapter,
        format_adapter=format_adapter,
        overrides=("3.attn.proj", "8.mlp"),
        suffixes=(".net.0.proj",),
    )

    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=1
    )
    block = object()
    quantize(block, 3)

    high = fp8_adapter.calls[0][2]
    low = format_adapter.calls[0][1]["filter_fn"]
    for held in ("attn.proj", "ff.net.0.proj"):
        assert high(object(), held), held
        assert not low(object(), held), held
    assert low(object(), "attn.to_q")
    assert not high(object(), "attn.to_q")


@pytest.mark.parametrize(
    ("format_name", "config_flags", "target_setting"),
    [
        (
            "fp4",
            {"use_fp4_gemms": True, "use_int8_gemms": False},
            "fp4_gemm_module_list",
        ),
        (
            "int8",
            {"use_fp4_gemms": False, "use_int8_gemms": True},
            "int8_gemm_module_list",
        ),
    ],
)
def test_blockwise_fp4_and_int8_route_through_format_adapter(
    format_name,
    config_flags,
    target_setting,
):
    adapter = RecordingFormatAdapter()
    settings = SimpleNamespace(
        fp4_gemm_module_list=[],
        int8_gemm_module_list=[],
        fp8_precision_overrides=("2.attn",),
        fp8_precision_override_suffixes=(".proj",),
    )
    setattr(settings, target_setting, ["transformer.blocks"])
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_hybrid_gemm_schedule=False,
            **config_flags,
        ),
        settings=settings,
        fp8=SimpleNamespace(module_list=lambda: []),
        blockwise_fp8_backend=FilterRecordingFp8Adapter(),
        format_backend=adapter,
    )

    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=2
    )
    block = object()
    quantize(block, 2)

    # The primary converter takes the hybrid schedule and nothing else; the
    # carve-outs it used to be handed are the filter's now.
    call_block, call_kwargs = adapter.calls[0]
    call_kwargs.pop("filter_fn")
    assert [(call_block, call_kwargs)] == [
        (block, {"device": "cuda:2", "hybrid": False})
    ]


def test_blockwise_exact_component_target_routes_wrapped_blocks():
    adapter = RecordingFormatAdapter()
    model = SimpleNamespace(
        config=SimpleNamespace(
            gemm_formats=frozenset({"int8"}),
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=[],
            int8_gemm_module_list=["transformer"],
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        fp8=SimpleNamespace(module_list=lambda: []),
        blockwise_fp8_backend=FilterRecordingFp8Adapter(),
        format_backend=adapter,
    )

    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=1
    )
    block = object()
    quantize(block, 0)

    call_block, call_kwargs = adapter.calls[0]
    filter_fn = call_kwargs.pop("filter_fn")
    assert filter_fn(object(), "anything")
    assert [(call_block, call_kwargs)] == [
        (block, {"device": "cuda:1", "hybrid": False})
    ]


def _targeted_block_model(*, format_adapter, fp4=(), int8=(), fp8=()):
    return SimpleNamespace(
        config=SimpleNamespace(
            use_fp4_gemms=bool(fp4),
            use_fp8_gemms=bool(fp8),
            use_int8_gemms=bool(int8),
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=list(fp4),
            int8_gemm_module_list=list(int8),
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        fp8=SimpleNamespace(module_list=lambda: list(fp8)),
        blockwise_fp8_backend=(format_adapter if fp8 else None),
        format_backend=(None if fp8 else format_adapter),
    )


@pytest.mark.parametrize("format_name", ["fp4", "int8", "fp8"])
def test_descendant_target_quantizes_only_block_zero_subpath(format_name):
    adapter = (
        FilterRecordingFp8Adapter()
        if format_name == "fp8"
        else RecordingFormatAdapter()
    )
    targets = {format_name: ("transformer.blocks.0.attn",)}
    model = _targeted_block_model(
        format_adapter=adapter,
        fp4=targets.get("fp4", ()),
        int8=targets.get("int8", ()),
        fp8=targets.get("fp8", ()),
    )
    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=0
    )
    blocks = [object(), object()]

    quantize(blocks[0], 0)
    quantize(blocks[1], 1)

    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    filter_fn = call[2] if format_name == "fp8" else call[1]["filter_fn"]
    assert filter_fn(object(), "attn.proj")
    assert not filter_fn(object(), "mlp.proj")


def test_descendant_target_filter_is_suffix_collision_safe():
    adapter = RecordingFormatAdapter()
    model = _targeted_block_model(
        format_adapter=adapter,
        int8=("transformer.blocks.0.attn",),
    )
    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=0
    )

    quantize(object(), 0)

    filter_fn = adapter.calls[0][1]["filter_fn"]
    assert filter_fn(object(), "attn.proj")
    assert not filter_fn(object(), "attention.proj")


def test_multiple_wrap_attrs_resolve_flattened_index_to_actual_fqn():
    adapter = RecordingFormatAdapter()
    model = _targeted_block_model(
        format_adapter=adapter,
        int8=("transformer.refiner.0.attn",),
    )
    component = SimpleNamespace(
        blocks=[object(), object()],
        refiner=[object(), object()],
    )
    quantize = shard.build_block_quantize_fn(
        runtime(model),
        "transformer",
        ["blocks", "refiner"],
        local_rank=0,
        component=component,
    )
    flattened = component.blocks + component.refiner

    for index, block in enumerate(flattened):
        quantize(block, index)

    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] is component.refiner[0]
    filter_fn = adapter.calls[0][1]["filter_fn"]
    assert filter_fn(object(), "attn.proj")
    assert not filter_fn(object(), "attention.proj")


@pytest.mark.parametrize(
    "target",
    ["transformer", "transformer.blocks"],
)
def test_whole_component_or_list_target_quantizes_every_wrapped_block(target):
    adapter = RecordingFormatAdapter()
    model = _targeted_block_model(
        format_adapter=adapter,
        int8=(target,),
    )
    quantize = shard.build_block_quantize_fn(
        runtime(model), "transformer", ["blocks"], local_rank=0
    )
    blocks = [object(), object()]

    for index, block in enumerate(blocks):
        quantize(block, index)

    assert [call[0] for call in adapter.calls] == blocks
    assert all(call[1]["filter_fn"](object(), "any.linear") for call in adapter.calls)


def test_exact_component_target_maps_to_transformer_root():
    adapter = object()
    model = SimpleNamespace(
        load_contract=SimpleNamespace(requested_format=QuantizationFormat.INT8),
        settings=SimpleNamespace(
            fp4_gemm_module_list=[],
            int8_gemm_module_list=["transformer", "transformer_2.blocks"],
        ),
        format_backend=adapter,
        fp8=SimpleNamespace(targets_for=lambda component: ()),
    )

    assert backends(model).format_targets_for("transformer") == ("",)
    assert backends(model).format_targets_for("transformer_2") == ("blocks",)
    assert backends(model).transformer_adapter("transformer") == (
        adapter,
        ("",),
    )
    assert backends(model).transformer_adapter("transformer_2") == (
        adapter,
        ("blocks",),
    )


@pytest.mark.parametrize(
    ("target", "wrapped", "expected"),
    [
        ("transformer", "blocks", True),
        ("transformer.blocks.attn", "blocks", True),
        ("transformer.blocks", "blocks", True),
        ("transformer.block", "blocks", False),
        ("transformer.blocks_extra", "blocks", False),
    ],
)
def test_format_fsdp_preflight_uses_boundary_safe_path_containment(
    target,
    wrapped,
    expected,
):
    model = SimpleNamespace(
        config=SimpleNamespace(fully_shard_degree=2),
        load_contract=SimpleNamespace(requested_format=QuantizationFormat.INT8),
        settings=SimpleNamespace(
            fsdp_strategy={"transformer": {"wrap_attrs": [wrapped]}},
            fp4_gemm_module_list=[],
            int8_gemm_module_list=[target],
        ),
    )

    assert backends(model).places_format_backend_under_fsdp2() is expected


def test_pure_fp4_wan_targets_require_backend_preflight():
    model = _hybrid_model(adapter=RecordingAdapter())

    assert backends(model).requires_blockwise_fp8()


# `test_narrow_fp4_target_preserves_broad_fp8_remainder` lived here. It
# described a narrow FP4 target inside a broader FP8 one, which a declaration
# cannot express and no model ever used: `keep_high` carves the better format
# out of the low one, never the reverse. The survey before the migration found
# every model declaring identical lists or a clean superset.


def test_narrow_fp4_target_under_broad_fp8_requires_backend_preflight():
    model = _hybrid_model(adapter=RecordingAdapter())
    model.settings.fp4_gemm_module_list = ["transformer.blocks.0.attn"]
    model.fp8 = SimpleNamespace(module_list=lambda: ["transformer.blocks"])

    assert backends(model).requires_blockwise_fp8()


def test_eager_fp4_with_fp8_only_target_preflights_component_backend():
    model = _hybrid_model(adapter=RecordingAdapter())
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.EAGER,
    )

    assert backends(model).uses_blockwise_fp8()


@pytest.mark.parametrize(
    ("uses_blockwise", "expected"),
    [
        (False, ["fp8", "format"]),
        (True, ["fp8", "format", "blockwise_fp8"]),
    ],
)
def test_quantization_backends_preflight_resolves_required_adapters(
    monkeypatch, uses_blockwise, expected
):
    from xfuser.model_executor.models.runner_models.loading.backend_selection import (
        QuantizationBackends,
    )

    observed = []
    for name in ("fp8", "format", "blockwise_fp8"):
        monkeypatch.setattr(
            QuantizationBackends,
            name,
            property(lambda _self, name=name: observed.append(name)),
        )
    selected = object.__new__(QuantizationBackends)
    selected.uses_blockwise_fp8 = lambda: uses_blockwise

    selected.preflight()

    assert observed == expected


def _fsdp_patch_model(
    *,
    strategy,
    fp4_targets=("transformer.blocks",),
    fp8_targets=("transformer.blocks", "transformer_2.blocks"),
    prefixes=(),
    suffixes=(),
    hybrid=False,
    fully_shard_degree=2,
):
    model = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=fully_shard_degree,
            gemm_formats=frozenset({"fp4"}),
            use_hybrid_gemm_schedule=hybrid,
        ),
        settings=SimpleNamespace(
            fsdp_strategy=strategy,
            fp4_gemm_module_list=list(fp4_targets),
            fp8_precision_overrides=prefixes,
            fp8_precision_override_suffixes=suffixes,
        ),
        fp8=SimpleNamespace(module_list=lambda: list(fp8_targets)),
    )
    return model


def test_fp8_only_target_outside_fsdp_strategy_needs_no_torchao_patches():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)

    assert not backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)


def test_fsdp_sharded_fp8_only_torchao_target_needs_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)

    assert backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("transformer", True),
        ("transformer.blocks.attn", True),
        ("transformer.blocks", True),
        ("transformer.block", False),
        ("transformer.blocks_extra", False),
    ],
)
def test_fp8_fsdp_preflight_uses_boundary_safe_path_containment(
    target,
    expected,
):
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        fp4_targets=(),
        fp8_targets=(target,),
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)

    assert (
        backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)
        is expected
    )


def test_fsdp_sharded_fp8_only_aiter_target_needs_no_torchao_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert not backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)


@pytest.mark.parametrize(
    ("prefixes", "suffixes", "hybrid"),
    [
        (("0.",), (), False),
        ((), (".net.0.proj",), False),
        ((), (), True),
    ],
)
def test_fsdp_sharded_fp4_torchao_fp8_paths_need_patches(
    prefixes,
    suffixes,
    hybrid,
):
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        prefixes=prefixes,
        suffixes=suffixes,
        hybrid=hybrid,
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)


def test_fp4_torchao_fp8_paths_outside_fsdp_strategy_need_no_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
        prefixes=("0.",),
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert not backends(model).places_torchao_tensor_subclass_under_fsdp2(adapter)


def test_fsdp_fp4_override_triggers_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        fp8_targets=("transformer.blocks",),
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.FSDP_META,
    )

    assert backends(model).uses_blockwise_fp8()


def test_eager_fsdp_fp8_triggers_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        fp4_targets=(),
        fp8_targets=("transformer.blocks",),
    )
    model.config.use_fp4_gemms = False
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP8,
        materialization_mode=MaterializationMode.EAGER,
    )

    assert backends(model).uses_blockwise_fp8()


def test_fp4_override_outside_strategy_skips_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
        fp8_targets=("transformer.blocks",),
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.FSDP_META,
    )

    # A prefix carve-out is placed by the FP8 converter now rather than by the
    # FP4 owner, so the backend is required wherever the carve-out is.
    assert backends(model).uses_blockwise_fp8()


def test_backend_preflight_uses_component_target_requirement(monkeypatch):
    observed = []
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(),
    )

    def select_backend(contract, *, capabilities):
        return adapter

    monkeypatch.setattr(
        fp8_backends,
        "select_blockwise_fp8_backend",
        select_backend,
    )
    monkeypatch.setattr(
        fp8_backends,
        "validate_torchao_fsdp2_patches",
        lambda contract, *, capabilities, required: observed.append(required),
    )
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.TORCHAO,
        materialization_mode=MaterializationMode.FSDP_META,
    )

    selected = backends(model).blockwise_fp8

    assert selected is adapter
    assert observed == [True]


def test_component_outside_fsdp_strategy_does_not_block_startup(monkeypatch):
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="patches unavailable",
        ),
    )
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.TORCHAO,
    )

    adapter = backends(model).blockwise_fp8

    assert adapter.backend is QuantizationBackend.TORCHAO


def test_fsdp_boundary_has_no_global_fp8_patch_assertion(monkeypatch):
    from xfuser.core.utils import runner_utils

    monkeypatch.setattr(runner_utils, "_TORCHAO_FLOAT8_FSDP2_PATCHES", [])
    monkeypatch.setattr(
        shard,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=0),
    )
    monkeypatch.setattr(
        shard,
        "get_fs_group",
        lambda: SimpleNamespace(local_rank=0, device_group=object()),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(use_fp8_gemms=True),
        settings=SimpleNamespace(fsdp_strategy={}),
        pipe=SimpleNamespace(components={}),
        _loader=object(),
    )

    shard.shard_pipeline_components(runtime(model))


def test_fp4_override_under_fsdp_requires_patches_with_aiter_fp8_backend(
    monkeypatch,
):
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(
            aiter_block_scale=True,
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="missing fsdp_post_all_gather",
        ),
    )
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.AITER,
    )

    with pytest.raises(ValueError, match=r"FSDP.*fsdp_post_all_gather"):
        backends(model).blockwise_fp8
def test_streamed_fp8_target_does_not_skip_disjoint_target_in_component(
    monkeypatch,
):
    streamed_module = object()
    post_load_module = object()
    fp8_calls = []
    adapter = SimpleNamespace(
        converts_before_device_move=False,
        backend=QuantizationBackend.TORCHAO,
        storage_semantics="tensorwise_dynamic",
        convert_module=lambda module, **kwargs: fp8_calls.append((module, kwargs)),
    )
    pipe = SimpleNamespace(
        transformer=SimpleNamespace(
            blocks=streamed_module,
            encoder=post_load_module,
        )
    )
    pipe.to = lambda _device: pipe
    model = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=1,
            enable_model_cpu_offload=False,
            enable_sequential_cpu_offload=False,
            enable_group_cpu_offload=False,
            gemm_formats=frozenset({"fp8"}),
            use_hybrid_attn_schedule=False,
            use_hybrid_gemm_schedule=False,
            use_vae_channels_last_format=False,
        ),
        settings=SimpleNamespace(
            int8_gemm_module_list=None,
            fp8_gemm_include_suffixes=None,
        ),
        fp8=SimpleNamespace(
            module_list=lambda: [
                "transformer.blocks",
                "transformer.encoder",
            ]
        ),
        fp8_backend=adapter,
        pipe=pipe,
        quantization_ledger=QuantizationLedger(
            fp8_streaming_targets={"transformer.blocks"},
            fp8_descriptor_components={"transformer"},
        ),
        _replicated_broadcast_load=lambda: False,
    )
    monkeypatch.setattr(
        placement,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=0),
    )

    placement.place_pipeline_components(runtime(model))

    module, kwargs = fp8_calls[0]
    kwargs.pop("filter_fn")
    assert len(fp8_calls) == 1
    assert (module, kwargs) == (post_load_module, {"device": "cuda:0"})
