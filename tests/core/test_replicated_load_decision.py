"""Unit tests for the memory-efficient load's branch selection.

Both loads are collective, so choosing the wrong branch hangs the run instead of raising: hence
--memory_efficient_replicated_load being opt-in rather than inferred, hence its exclusions
(weight-splitting parallelism, single rank, runners that do not build through the meta seams), and
hence a fill strategy chosen by identity rather than by guessing from a component's name. Those
decisions are pure and worth pinning on CPU; the fills themselves need multiple GPUs and are covered
by the GPU suite.

Run with:
    pytest tests/core/test_replicated_load_decision.py -v
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.models.runner_models.loading import meta_load
from xfuser.model_executor.models.runner_models.loading.meta_load import (
    MemoryEfficientLoader,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadCapability,
    LoaderAdapter,
    UnsupportedLoadContract,
)
from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
    CheckpointRequest,
)


@pytest.fixture(autouse=True)
def single_process_env(monkeypatch):
    """The decision logs, and runner_utils.log reads RANK/WORLD_SIZE straight from the env."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def make_loader(
    monkeypatch,
    *,
    requested=True,
    world_size=2,
    fully_shard_degree=1,
    pipefusion_parallel_degree=1,
    tensor_parallel_degree=1,
    supported=True,
    memory_efficient_sharding=False,
):
    monkeypatch.setattr(
        meta_load, "get_world_group", lambda: SimpleNamespace(world_size=world_size)
    )
    model = SimpleNamespace(
        settings=SimpleNamespace(model_name="stand-in/checkpoint"),
        config=SimpleNamespace(
            memory_efficient_replicated_load=requested,
            memory_efficient_sharding=memory_efficient_sharding,
            fully_shard_degree=fully_shard_degree,
            pipefusion_parallel_degree=pipefusion_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
        ),
        load_capability=(
            LoadCapability.meta("transformer", replicated=True)
            if supported
            else LoadCapability.unsupported("test runner bypasses the seam")
        ),
    )
    model.settings.fsdp_strategy = {"transformer": {"wrap_attrs": ["blocks"]}}
    return MemoryEfficientLoader(model)


# ============================================================================
# The flag: opt-in only, never inferred
# ============================================================================


def test_off_unless_asked_for(monkeypatch):
    """Default off like --memory_efficient_sharding: a mismatched collective here hangs rather than
    raises, so it is the user's call and never inferred from the hardware."""
    assert not make_loader(monkeypatch, requested=False).replicated_broadcast_load()


def test_broadcasts_when_asked_for_on_a_replicated_multi_gpu_run(monkeypatch):
    assert make_loader(monkeypatch, requested=True).replicated_broadcast_load()


# ============================================================================
# Exclusions that hold even when the flag is passed
# ============================================================================


@pytest.mark.parametrize(
    "degrees",
    [
        {"fully_shard_degree": 2},
        {"pipefusion_parallel_degree": 2},
        {"tensor_parallel_degree": 2},
    ],
)
def test_never_broadcasts_when_ranks_hold_different_weights(monkeypatch, degrees):
    """Nothing is replicated under FSDP/PipeFusion/TP, so broadcasting rank0 would be wrong."""
    loader = make_loader(monkeypatch, **degrees)
    assert not loader.replicated_broadcast_load()


def test_never_broadcasts_on_a_single_rank(monkeypatch):
    loader = make_loader(monkeypatch, world_size=1)
    assert not loader.replicated_broadcast_load()


def test_single_rank_unsupported_runner_resolves_to_eager(monkeypatch):
    loader = make_loader(monkeypatch, world_size=1, supported=False)

    assert not loader.replicated_broadcast_load()


def test_never_broadcasts_for_a_runner_that_is_not_wired_for_it(monkeypatch):
    """A runner loading its components directly leaves peers with no meta tensors to fill."""
    loader = make_loader(monkeypatch, supported=False)
    with pytest.raises(UnsupportedLoadContract, match="bypasses the seam"):
        loader.replicated_broadcast_load()


# ============================================================================
# Caching: several load-time seams consult the decision and must all agree
# ============================================================================


def test_decision_is_resolved_once(monkeypatch):
    loader = make_loader(monkeypatch)
    calls = []
    original = loader._resolve_replicated_broadcast_load

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(loader, "_resolve_replicated_broadcast_load", counted)
    assert loader.replicated_broadcast_load()
    assert loader.replicated_broadcast_load()
    assert len(calls) == 1


# ============================================================================
# The FSDP meta path is a separate decision
# ============================================================================


def test_fsdp_meta_load_needs_both_the_flag_and_sharding(monkeypatch):
    assert not make_loader(monkeypatch, memory_efficient_sharding=True).fsdp_meta_load()
    assert not make_loader(monkeypatch, fully_shard_degree=2).fsdp_meta_load()
    assert make_loader(
        monkeypatch, memory_efficient_sharding=True, fully_shard_degree=2
    ).fsdp_meta_load()


# ============================================================================
# Which meta components can self-fill from disk
# ============================================================================


class FakeWrapper(torch.nn.Module):
    """Stands in for a diffusers transformer wrapper: enough of the from_config API to be built."""

    @classmethod
    def load_config(cls, model_name, subfolder=None, **kwargs):
        return {"hidden": 4}

    @classmethod
    def from_config(cls, config, **kwargs):
        module = cls()
        module.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(config["hidden"], config["hidden"])]
        )
        return module


def test_only_the_transformer_we_built_self_fills_from_disk(monkeypatch):
    """The two fill strategies need different collectives, and only a transformer this loader
    meta-built has the checkpoint mapping the per-block disk fill reads. Regression: this used to be
    decided by whether the component's name started with "transformer"."""
    loader = make_loader(monkeypatch)
    built = loader.build_meta_transformer(FakeWrapper, subfolder="transformer")
    assert loader.self_fills_from_disk(built)
    # Anything this loader did not build (a text encoder, or a transformer some runner meta-built
    # itself) is filled by the generic rank0 broadcast instead.
    assert not loader.self_fills_from_disk(FakeWrapper.from_config({"hidden": 4}))


def test_the_meta_transformer_is_built_on_meta_in_bf16(monkeypatch):
    """The per-block disk fill and the AITER quantize both expect the checkpoint's bf16 on meta;
    from_config would otherwise leave fp32 on cpu."""
    loader = make_loader(monkeypatch)
    built = loader.build_meta_transformer(FakeWrapper, subfolder="transformer")
    assert all(p.is_meta for p in built.parameters())
    assert all(p.dtype is torch.bfloat16 for p in built.parameters())


def test_tracking_a_built_transformer_does_not_keep_it_alive(monkeypatch):
    """The bookkeeping must not pin a component the pipeline has replaced or dropped."""
    import gc

    loader = make_loader(monkeypatch)
    built = loader.build_meta_transformer(FakeWrapper, subfolder="transformer")
    assert len(loader._meta_transformers) == 1
    del built
    gc.collect()
    assert len(loader._meta_transformers) == 0


def test_custom_mapped_source_can_build_meta_only_for_local_fill(monkeypatch):
    loader = make_loader(monkeypatch, world_size=1)
    loader.model.load_capability = LoadCapability(
        local_meta_transformers=("transformer",),
        loader_adapter=LoaderAdapter.DISTILLED_WAN,
    )
    source = CheckpointManifest(
        weight_map={"blocks.0.weight": "/weights/distilled.safetensors"}
    )

    built = loader.build_meta_transformer(
        FakeWrapper,
        subfolder="transformer",
        weight_source=source,
    )

    assert loader._meta_transformers[built] is source


def test_disk_filler_receives_the_exact_request_used_for_meta_build(monkeypatch):
    loader = make_loader(monkeypatch)
    request = CheckpointRequest(
        "org/repo",
        subfolder="transformer",
        revision="refs/pr/7",
        variant="fp16",
        token="secret",
        cache_dir="/cache",
        local_files_only=True,
    )
    built = loader.build_meta_transformer(FakeWrapper, request)
    captured = []

    class Filler:
        def __init__(self, model, component, wrap_attrs, request, device, group):
            captured.append(request)

        fill_block = None
        finalize = None

    monkeypatch.setattr(meta_load, "_TransformerDiskFiller", Filler)
    loader.build_transformer_disk_loaders(built, ["blocks"], "transformer", "cpu")

    assert captured == [request]
    assert captured[0] is request


def test_dual_meta_transformers_keep_distinct_checkpoint_requests(monkeypatch):
    loader = make_loader(monkeypatch)
    loader.model.load_capability = LoadCapability.meta(
        "transformer", "transformer_2", replicated=True
    )
    loader.model.settings.fsdp_strategy["transformer_2"] = {"wrap_attrs": ["blocks"]}
    first_request = CheckpointRequest(
        "org/repo", subfolder="transformer", revision="main"
    )
    second_request = CheckpointRequest(
        "org/repo", subfolder="transformer_2", revision="refs/pr/9"
    )
    first = loader.build_meta_transformer(FakeWrapper, first_request)
    second = loader.build_meta_transformer(FakeWrapper, second_request)
    captured = []

    class Filler:
        def __init__(self, model, component, wrap_attrs, request, device, group):
            captured.append((component, request))

        fill_block = None
        finalize = None

    monkeypatch.setattr(meta_load, "_TransformerDiskFiller", Filler)
    loader.build_transformer_disk_loaders(first, ["blocks"], "transformer", "cpu")
    loader.build_transformer_disk_loaders(second, ["blocks"], "transformer_2", "cpu")

    assert captured == [(first, first_request), (second, second_request)]


# ============================================================================
# Every runner's declared capability matches how it actually loads
# ============================================================================


def test_runners_that_bypass_the_meta_seam_declare_it():
    """A runner claiming meta support must construct through _build_transformer."""
    import inspect

    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        # _load_model may be inherited; read the source of whichever class actually defines it.
        goes_through_seam = "_build_transformer" in inspect.getsource(cls._load_model)
        declares_support = bool(cls.load_capability.meta_transformers)
        if declares_support and not goes_through_seam:
            mismatched.append(f"{cls.__name__} (from {cls._load_model.__qualname__})")

    assert not mismatched, (
        "these runners load their transformer outside _build_transformer but still declare "
        "meta-load support: " + ", ".join(sorted(mismatched))
    )


def test_runner_load_capabilities_match_model_quantization_capabilities():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        expected = LoadCapability.for_runner(cls.capabilities).quantization_contracts
        if cls.load_capability.quantization_contracts != expected:
            mismatched.append(cls.__name__)

    assert not mismatched, (
        "load_capability quantization contracts disagree with ModelCapabilities: "
        + ", ".join(sorted(mismatched))
    )


def test_runner_fsdp_meta_support_matches_capabilities_and_strategy():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        declaration = cls.load_capability
        candidates = declaration.replicated_meta_transformers
        expected = (
            tuple(
                name
                for name in candidates
                if cls.settings.fsdp_strategy.get(name, {}).get("wrap_attrs")
            )
            if cls.capabilities.fully_shard_degree
            else ()
        )
        if declaration.fsdp_meta_transformers != expected:
            mismatched.append(
                f"{cls.__name__}: expected {expected}, "
                f"declared {declaration.fsdp_meta_transformers}"
            )

    assert not mismatched, "\n".join(mismatched)


def test_base_runner_selects_the_production_contract_before_loading(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(base_model, "_use_aiter_fp8_rdna4", lambda: True)
    monkeypatch.setattr(base_model, "_is_cuda", lambda: False)
    model_capabilities = base_model.ModelCapabilities(
        fully_shard_degree=True,
        use_fp8_gemms=True,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=2,
            pipefusion_parallel_degree=1,
            tensor_parallel_degree=1,
            memory_efficient_sharding=True,
            memory_efficient_replicated_load=False,
            use_fp8_gemms=True,
            use_fp4_gemms=False,
            use_int8_gemms=False,
        ),
        settings=SimpleNamespace(
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}}
        ),
        load_capability=LoadCapability.for_runner(
            model_capabilities,
            meta_transformers=("transformer",),
            replicated=True,
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        ),
    )

    selected = base_model.xFuserModel._select_preload_contract(runner, world_size=2)

    assert selected.requested_format.name == "FP8"
    assert selected.selected_backend.name == "AITER"
    assert selected.materialization_mode.name == "FSDP_META"


def test_base_runner_rejects_unsupported_meta_mode_before_loading(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(base_model, "_use_aiter_fp8_rdna4", lambda: False)
    monkeypatch.setattr(base_model, "_is_cuda", lambda: True)
    runner = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=1,
            pipefusion_parallel_degree=1,
            tensor_parallel_degree=1,
            memory_efficient_sharding=False,
            memory_efficient_replicated_load=True,
            use_fp8_gemms=False,
            use_fp4_gemms=False,
            use_int8_gemms=False,
        ),
        settings=SimpleNamespace(fsdp_strategy={}),
        load_capability=LoadCapability.for_runner(
            base_model.ModelCapabilities(),
            unsupported_reason="custom loader bypasses the seam",
        ),
    )

    with pytest.raises(
        UnsupportedLoadContract, match="custom loader bypasses the seam"
    ):
        base_model.xFuserModel._select_preload_contract(runner, world_size=2)


def test_base_runner_uses_effective_single_rank_mode(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(base_model, "_use_aiter_fp8_rdna4", lambda: False)
    monkeypatch.setattr(base_model, "_is_cuda", lambda: True)
    runner = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=1,
            pipefusion_parallel_degree=1,
            tensor_parallel_degree=1,
            memory_efficient_sharding=False,
            memory_efficient_replicated_load=True,
            use_fp8_gemms=False,
            use_fp4_gemms=False,
            use_int8_gemms=False,
        ),
        settings=SimpleNamespace(fsdp_strategy={}),
        load_capability=LoadCapability.for_runner(
            base_model.ModelCapabilities(),
            unsupported_reason="custom loader bypasses the seam",
        ),
    )

    selected = base_model.xFuserModel._select_preload_contract(runner, world_size=1)

    assert selected.materialization_mode.name == "EAGER"


def test_wan22_instance_settings_refresh_both_transformers(monkeypatch):
    import copy

    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.wan import (
        xFuserWan22I2VModel,
    )

    runner = object.__new__(xFuserWan22I2VModel)
    runner.settings = copy.deepcopy(xFuserWan22I2VModel.settings)
    runner.config = SimpleNamespace(
        fully_shard_degree=2,
        pipefusion_parallel_degree=1,
        tensor_parallel_degree=1,
        memory_efficient_sharding=True,
        memory_efficient_replicated_load=False,
        use_fp8_gemms=True,
        use_fp4_gemms=False,
        use_int8_gemms=False,
    )
    runner._customize_settings(SimpleNamespace())
    monkeypatch.setattr(base_model, "_use_aiter_fp8_rdna4", lambda: True)
    monkeypatch.setattr(base_model, "_is_cuda", lambda: False)

    runner._refresh_load_capability()
    fsdp_selected = runner._select_preload_contract(world_size=2)

    assert runner.load_capability.fsdp_meta_transformers == (
        "transformer",
        "transformer_2",
    )
    assert runner.load_capability.replicated_meta_transformers == (
        "transformer",
        "transformer_2",
    )
    assert fsdp_selected.materialization_mode.name == "FSDP_META"

    runner.config.fully_shard_degree = 1
    runner.config.memory_efficient_sharding = False
    runner.config.memory_efficient_replicated_load = True
    replicated_selected = runner._select_preload_contract(world_size=2)

    assert replicated_selected.materialization_mode.name == "REPLICATED_META"


def test_build_transformer_preserves_request_subfolder_without_override():
    from xfuser.model_executor.models.runner_models import base_model

    calls = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return "loaded"

    request = CheckpointRequest(
        "org/repo",
        subfolder="transformer_2",
        revision="refs/pr/9",
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: []),
        fp8_backend=None,
        settings=SimpleNamespace(fsdp_strategy={}),
    )

    result = base_model.xFuserModel._build_transformer(
        runner, Wrapper, checkpoint_request=request
    )

    assert result == "loaded"
    assert calls == [
        (
            "org/repo",
            {
                "torch_dtype": torch.bfloat16,
                "quantization_config": None,
                "subfolder": "transformer_2",
                "revision": "refs/pr/9",
                "local_files_only": False,
            },
        )
    ]


def test_transformer_structure_inspection_keeps_parameters_and_buffers_meta(
    monkeypatch,
):
    from contextlib import contextmanager
    import accelerate

    from xfuser.model_executor.models.runner_models import base_model

    calls = []

    @contextmanager
    def empty_weights(**kwargs):
        calls.append(kwargs)
        yield

    class Wrapper:
        @classmethod
        def load_config(cls, model_name, **kwargs):
            return "config"

        @classmethod
        def from_config(cls, config, **kwargs):
            return "structure"

    monkeypatch.setattr(accelerate, "init_empty_weights", empty_weights)

    result = base_model.xFuserModel._build_transformer_structure(
        SimpleNamespace(),
        Wrapper,
        CheckpointRequest("org/repo", subfolder="transformer"),
        None,
    )

    assert result == "structure"
    assert calls == [{"include_buffers": True}]


def test_structure_inspection_reports_old_accelerate_explicitly(monkeypatch):
    import accelerate

    from xfuser.model_executor.models.runner_models import base_model

    def legacy_empty_weights():
        raise AssertionError("must fail before entering context")

    class Wrapper:
        @classmethod
        def load_config(cls, model_name, **kwargs):
            return "config"

    monkeypatch.setattr(accelerate, "init_empty_weights", legacy_empty_weights)

    with pytest.raises(
        RuntimeError,
        match=r"accelerate\.init_empty_weights.*include_buffers",
    ):
        base_model.xFuserModel._build_transformer_structure(
            SimpleNamespace(),
            Wrapper,
            CheckpointRequest("org/repo", subfolder="transformer"),
            None,
        )


def test_structure_inspection_catches_include_buffers_error_on_context_enter(
    monkeypatch,
):
    from contextlib import contextmanager
    import accelerate

    from xfuser.model_executor.models.runner_models import base_model

    @contextmanager
    def broken_empty_weights(**kwargs):
        raise TypeError("include_buffers is unsupported")
        yield

    class Wrapper:
        @classmethod
        def load_config(cls, model_name, **kwargs):
            return "config"

        @classmethod
        def from_config(cls, config, **kwargs):
            raise AssertionError("structure allocation must not start")

    monkeypatch.setattr(accelerate, "init_empty_weights", broken_empty_weights)

    with pytest.raises(
        RuntimeError,
        match=r"accelerate\.init_empty_weights.*include_buffers",
    ):
        base_model.xFuserModel._build_transformer_structure(
            SimpleNamespace(),
            Wrapper,
            CheckpointRequest("org/repo", subfolder="transformer"),
            None,
        )


def test_build_transformer_routes_torchao_fp8_to_native_diffusers_config(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
        Fp8BackendCapabilities,
        select_fp8_backend,
    )

    contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP8,
        selected_backend=QuantizationBackend.TORCHAO,
    )
    adapter = select_fp8_backend(
        contract,
        capabilities=Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_diffusers_streaming=True,
        ),
    )
    calls = []
    sentinel = object()

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(kwargs)
            return "streamed"

    monkeypatch.setattr(
        adapter,
        "_stream_config_factory",
        lambda exclusions: sentinel,
    )

    def structure_factory(*args, **kwargs):
        return SimpleNamespace(
            named_modules=lambda: [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.proj", torch.nn.Linear(2, 2)),
            ],
            get_submodule=lambda name: object(),
        )

    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        _fp8_streaming_targets=set(),
        _quantization_streaming_targets=set(),
        _build_transformer_structure=structure_factory,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "streamed"
    assert calls[0]["quantization_config"] is sentinel
    assert runner._fp8_streaming_targets == {"transformer.blocks"}


def test_blockwise_transformer_marks_only_wrapped_target_as_streamed(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import fp8_backends

    adapter = SimpleNamespace(format=SimpleNamespace(value="fp8"))
    descriptor = SimpleNamespace(
        materialization_mode="blockwise",
        log_message=lambda: "blockwise fp8",
    )
    monkeypatch.setattr(
        fp8_backends,
        "plan_blockwise_transformer_fp8_load",
        lambda *args, **kwargs: descriptor,
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: True,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks", "encoder"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}}
        ),
        config=SimpleNamespace(use_fp4_gemms=False),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
        _quantization_descriptor_components=set(),
        _quantization_streaming_targets=set(),
        _loader=SimpleNamespace(build_meta_transformer=lambda *args, **kwargs: "meta"),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, SimpleNamespace())

    assert result == "meta"
    assert runner._fp8_streaming_targets == {"transformer.blocks"}
    assert runner._quantization_streaming_targets == {"transformer.blocks"}


def test_blockwise_fp4_marks_only_wrapped_fp8_remainder_as_streamed(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import format_backends

    adapter = SimpleNamespace(format=SimpleNamespace(value="fp4"))
    descriptor = SimpleNamespace(
        materialization_mode="blockwise",
        log_message=lambda: "blockwise fp4",
    )
    monkeypatch.setattr(
        format_backends,
        "describe_blockwise_format_load",
        lambda *args, **kwargs: descriptor,
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: True,
        _transformer_quantization_adapter=lambda name: (
            adapter,
            ("blocks",),
        ),
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks", "encoder"]),
        settings=SimpleNamespace(
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}}
        ),
        config=SimpleNamespace(use_fp4_gemms=True),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
        _quantization_descriptor_components=set(),
        _quantization_streaming_targets=set(),
        _loader=SimpleNamespace(build_meta_transformer=lambda *args, **kwargs: "meta"),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, SimpleNamespace())

    assert result == "meta"
    assert runner._fp8_streaming_targets == {"transformer.blocks"}
    assert runner._quantization_streaming_targets == {"transformer.blocks"}


def test_build_transformer_logs_explicit_torchao_post_load_fallback(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
        Fp8BackendCapabilities,
        select_fp8_backend,
    )

    calls = []
    logs = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(kwargs)
            return "loaded"

    adapter = select_fp8_backend(
        SimpleNamespace(
            requested_format=QuantizationFormat.FP8,
            selected_backend=QuantizationBackend.TORCHAO,
        ),
        capabilities=Fp8BackendCapabilities(torchao_fp8=True),
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )
    monkeypatch.setattr(base_model, "log", logs.append)

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "loaded"
    assert calls[0]["quantization_config"] is None
    assert any(
        "backend=torchao" in message
        and "materialization=post_load" in message
        and "fallback=" in message
        for message in logs
    )


def test_build_transformer_mapping_failure_falls_back_without_streaming_claim(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
        Fp8BackendCapabilities,
        select_fp8_backend,
    )

    calls = []
    logs = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(kwargs)
            return "loaded"

    adapter = select_fp8_backend(
        SimpleNamespace(
            requested_format=QuantizationFormat.FP8,
            selected_backend=QuantizationBackend.TORCHAO,
        ),
        capabilities=Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_diffusers_streaming=True,
        ),
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    def structure_factory(*args, **kwargs):
        raise RuntimeError("config shape unavailable")

    runner._build_transformer_structure = structure_factory
    monkeypatch.setattr(base_model, "log", logs.append)

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "loaded"
    assert calls[0]["quantization_config"] is None
    assert any(
        "backend=torchao" in message
        and "materialization=post_load" in message
        and "target mapping unavailable" in message
        for message in logs
    )


def test_eager_post_load_fallback_builds_meta_for_local_block_fill(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import format_backends
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )

    adapter = format_backends.AiterMxfp4BackendAdapter(
        backend=QuantizationBackend.AITER,
        format_=QuantizationFormat.FP4,
        native_unavailable_reason="requires full-precision block",
    )
    meta_component = object()
    marked = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("local blockwise plan must bypass eager from_pretrained")

    loader = SimpleNamespace(
        plan_eager_blockwise_fallback=lambda prepared, targets, wrap_attrs: (
            SimpleNamespace(enabled=True)
        ),
        build_meta_transformer=lambda *args, **kwargs: meta_component,
        mark_local_blockwise=marked.append,
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        _transformer_quantization_adapter=lambda component: (
            adapter,
            ("blocks",),
        ),
        _loader=loader,
        _fp8_descriptor_components=set(),
        _quantization_descriptor_components=set(),
        _fp8_streaming_targets=set(),
        _quantization_streaming_targets=set(),
        fp8=SimpleNamespace(targets_for=lambda component: ()),
        settings=SimpleNamespace(
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(
            use_fp4_gemms=True,
            use_hybrid_gemm_schedule=False,
        ),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )
    monkeypatch.setattr(base_model, "log", lambda message: None)

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result is meta_component
    assert marked == [meta_component]
    assert runner._quantization_streaming_targets == {"transformer.blocks"}


def test_build_transformer_preserves_aiter_native_streaming(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
        Fp8BackendCapabilities,
        select_fp8_backend,
    )

    calls = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(kwargs)
            return "loaded"

    adapter = select_fp8_backend(
        SimpleNamespace(
            requested_format=QuantizationFormat.FP8,
            selected_backend=QuantizationBackend.AITER,
        ),
        capabilities=Fp8BackendCapabilities(aiter_block_scale=True),
    )
    sentinel = object()
    monkeypatch.setattr(adapter, "_stream_config_factory", lambda targets: sentinel)
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "loaded"
    assert calls[0]["quantization_config"] is sentinel


@pytest.mark.parametrize(
    ("format_name", "adapter_name"),
    [
        ("FP4", "TorchaoNvfp4BackendAdapter"),
        ("INT8", "TorchaoInt8BackendAdapter"),
    ],
)
def test_build_transformer_streams_native_fp4_and_int8_configs(
    monkeypatch,
    format_name,
    adapter_name,
):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.loading import (
        format_backends,
    )

    adapter = getattr(format_backends, adapter_name)(
        backend=QuantizationBackend.TORCHAO,
        format_=getattr(QuantizationFormat, format_name),
        native_transformer_streaming=True,
    )
    sentinel = object()
    monkeypatch.setattr(adapter, "_stream_config_factory", lambda exclusions: sentinel)
    calls = []

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(kwargs)
            return "streamed"

    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        _transformer_quantization_adapter=lambda component: (
            adapter,
            ("blocks",),
        ),
        _native_quantization_device_map=lambda: {"": 0},
        _fp8_streaming_targets=set(),
        _quantization_streaming_targets=set(),
        _build_transformer_structure=lambda *args, **kwargs: SimpleNamespace(
            named_modules=lambda: [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.proj", torch.nn.Linear(1024, 1024)),
            ],
            get_submodule=lambda name: object(),
        ),
        settings=SimpleNamespace(
            fsdp_strategy={},
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "streamed"
    assert calls[0]["quantization_config"] is sentinel
    assert calls[0]["device_map"] == {"": 0}
    assert runner._quantization_streaming_targets == {"transformer.blocks"}


def test_build_transformer_records_only_streamed_nvfp4_leaves(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import format_backends
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
    )

    adapter = format_backends.TorchaoNvfp4BackendAdapter(
        backend=QuantizationBackend.TORCHAO,
        format_=QuantizationFormat.FP4,
        native_transformer_streaming=True,
    )
    sentinel = object()
    monkeypatch.setattr(adapter, "_stream_config_factory", lambda exclusions: sentinel)

    class Wrapper:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return "streamed"

    structure = SimpleNamespace(
        named_modules=lambda: [
            ("", object()),
            ("blocks", object()),
            ("blocks.0.keep", torch.nn.Linear(16, 16)),
            ("blocks.0.override", torch.nn.Linear(16, 16)),
            ("input_proj", torch.nn.Linear(16, 16)),
        ],
        get_submodule=lambda name: object(),
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        _transformer_quantization_adapter=lambda component: (
            adapter,
            ("blocks",),
        ),
        _native_quantization_device_map=lambda: {"": 0},
        _fp8_streaming_targets=set(),
        _quantization_streaming_targets=set(),
        _build_transformer_structure=lambda *args, **kwargs: structure,
        settings=SimpleNamespace(
            fsdp_strategy={},
            fp8_precision_overrides=("0.override",),
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = base_model.xFuserModel._build_transformer(runner, Wrapper)

    assert result == "streamed"
    assert runner._quantization_streaming_targets == {"transformer.blocks.0.keep"}


def test_eager_te_adapter_maps_multiple_components_and_logs_each(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    logs = []
    sentinel = SimpleNamespace(quant_mapping={})
    runner = SimpleNamespace(
        load_contract=SimpleNamespace(requested_format=SimpleNamespace(value="fp8")),
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: False,
        fp8_backend=SimpleNamespace(),
        fp8=SimpleNamespace(
            targets_for=lambda name: {
                "text_encoder": ["encoder.block"],
                "text_encoder_2": ["model.layers"],
            }[name]
        ),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=[
                "text_encoder.encoder.block",
                "text_encoder_2.model.layers",
            ]
        ),
        _loader=SimpleNamespace(
            build_meta_component=lambda name, fp8=False: (name, fp8)
        ),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
    )
    prepared = {
        name: SimpleNamespace(
            descriptor=SimpleNamespace(
                materialization_mode="streaming",
                log_message=lambda name=name: f"{name} descriptor",
            ),
            quantization_config=f"{name}-config",
        )
        for name in ("text_encoder", "text_encoder_2")
    }
    calls = []

    def prepare(
        adapter,
        *,
        component_name,
        targets,
        model_factory,
        stream_quant,
        supports_post_load,
    ):
        calls.append(
            (
                component_name,
                tuple(targets),
                model_factory(),
                stream_quant,
                supports_post_load,
            )
        )
        return prepared[component_name]

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends.prepare_text_encoder_fp8_load",
        prepare,
    )
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.text_encoder_adapter.TextEncoderFrameworkAdapter.pipeline_quantization_config",
        lambda self, mapping, existing=None: (
            setattr(sentinel, "quant_mapping", dict(mapping)) or sentinel
        ),
    )
    monkeypatch.setattr(base_model, "log", logs.append)

    kwargs, config = base_model.xFuserModel._meta_te_kwargs(runner)

    assert kwargs == {}
    assert calls == [
        (
            "text_encoder",
            ("encoder.block",),
            ("text_encoder", False),
            True,
            True,
        ),
        (
            "text_encoder_2",
            ("model.layers",),
            ("text_encoder_2", False),
            True,
            True,
        ),
    ]
    assert config is sentinel
    assert config.quant_mapping == {
        "text_encoder": "text_encoder-config",
        "text_encoder_2": "text_encoder_2-config",
    }
    assert logs == [
        "text_encoder descriptor",
        "text_encoder_2 descriptor",
    ]
    assert runner._fp8_streaming_targets == {
        "text_encoder.encoder.block",
        "text_encoder_2.model.layers",
    }


def test_hybrid_meta_te_uses_blockwise_fp8_backend(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    sentinel = SimpleNamespace(backend=SimpleNamespace(value="torchao"))
    runner = SimpleNamespace(
        load_contract=SimpleNamespace(
            requested_format=SimpleNamespace(value="fp8_fp4")
        ),
        fp8_backend=None,
        blockwise_fp8_backend=sentinel,
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: True,
        fp8=SimpleNamespace(targets_for=lambda name: ["encoder.block"]),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=["text_encoder.encoder.block"]
        ),
        _loader=SimpleNamespace(
            meta_te_kwargs=lambda: ({"text_encoder": "meta"}, None),
            build_meta_component=lambda name, fp8=False: object(),
        ),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
    )
    observed = []

    def prepare(adapter, **kwargs):
        observed.append(adapter)
        return SimpleNamespace(
            descriptor=SimpleNamespace(
                materialization_mode="post_load",
                log_message=lambda: "post-load",
            ),
            quantization_config=None,
        )

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends."
        "prepare_text_encoder_fp8_load",
        prepare,
    )
    monkeypatch.setattr(base_model, "log", lambda message: None)

    kwargs, config = base_model.xFuserModel._meta_te_kwargs(runner)

    assert (kwargs, config) == ({"text_encoder": "meta"}, None)
    assert observed == [sentinel]


def test_meta_te_placement_disables_torchao_native_pipeline_streaming(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models import base_model

    runner = SimpleNamespace(
        load_contract=SimpleNamespace(requested_format=SimpleNamespace(value="fp8")),
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: True,
        fp8_backend=SimpleNamespace(backend=SimpleNamespace(value="torchao")),
        fp8=SimpleNamespace(targets_for=lambda name: ["encoder.block"]),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=["text_encoder.encoder.block"]
        ),
        _loader=SimpleNamespace(
            meta_te_kwargs=lambda: ({"text_encoder": "meta"}, None),
            build_meta_component=lambda name, fp8=False: object(),
        ),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
    )
    observed = []

    def prepare(adapter, **kwargs):
        observed.append((kwargs["stream_quant"], kwargs["supports_post_load"]))
        return SimpleNamespace(
            descriptor=SimpleNamespace(
                materialization_mode="post_load",
                log_message=lambda: "post-load",
            ),
            quantization_config=None,
        )

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends.prepare_text_encoder_fp8_load",
        prepare,
    )
    monkeypatch.setattr(base_model, "log", lambda message: None)

    kwargs, config = base_model.xFuserModel._meta_te_kwargs(runner)

    assert (kwargs, config) == ({"text_encoder": "meta"}, None)
    assert observed == [(False, False)]
    assert runner._fp8_streaming_targets == set()


def test_meta_fsdp_rejects_text_encoder_post_load_fallback(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    runner = SimpleNamespace(
        load_contract=SimpleNamespace(requested_format=SimpleNamespace(value="fp8")),
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: True,
        fp8_backend=SimpleNamespace(backend=SimpleNamespace(value="aiter")),
        fp8=SimpleNamespace(targets_for=lambda name: ["encoder.block"]),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=["text_encoder.encoder.block"]
        ),
        _loader=SimpleNamespace(
            build_meta_component=lambda name, fp8=False: object(),
        ),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
    )

    def prepare(adapter, **kwargs):
        if not kwargs["supports_post_load"]:
            raise RuntimeError("text_encoder FP8 cannot fall back before allocation")
        pytest.fail("memory-efficient FSDP incorrectly allowed post-load")

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends.prepare_text_encoder_fp8_load",
        prepare,
    )

    with pytest.raises(RuntimeError, match="before allocation"):
        base_model.xFuserModel._meta_te_kwargs(runner)
