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

from xfuser.model_executor.models.runner_models.loading import (
    meta_load,
    text_encoder_plan,
    transformer_load,
)
from xfuser.model_executor.models.runner_models.loading.meta_load import (
    ModelLoader,
)
from xfuser.model_executor.models.runner_models.loading.backend_selection import (
    QuantizationBackends,
)
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadDeclaration,
    LoadRoute,
    LoadSupport,
    STANDARD_LOAD_ROUTES,
    UnsupportedLoadContract,
)
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
    QuantizationLedger,
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


def loader_for(runner):
    """The loader surface the load routes need, backed by a stub runner.

    A test describes the runner it wants and gets the loader that owns it, so the stubs stay a
    description of one model rather than of two collaborating objects. Anything the route can
    call is present: a test that leaves out the meta fill wants the case where it is never
    reached, not an AttributeError from the branch it forgot.
    """
    inner = getattr(
        runner, "loader", getattr(runner, "_loader", SimpleNamespace())
    )

    def missing_route(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"this case should not reach {name}")

        return fail

    ledger = getattr(runner, "quantization_ledger", QuantizationLedger())
    runner.quantization_ledger = ledger
    plan = getattr(runner, "fp8", SimpleNamespace(targets_for=lambda name: ()))
    backends = getattr(runner, "backends", SimpleNamespace())
    transformer_adapter = getattr(
        runner,
        "_transformer_quantization_adapter",
        lambda component_name: (
            getattr(runner, "fp8_backend", None),
            plan.targets_for(component_name),
        ),
    )
    loader = SimpleNamespace(
        model=runner,
        load_contract=getattr(runner, "load_contract", None),
        checkpoint_request=lambda subfolder=None, **kwargs: runner._checkpoint_request(
            subfolder, **kwargs
        ),
        quantization_ledger=ledger,
        quantization_plan=plan,
        backends=backends,
        transformer_quantization_adapter=transformer_adapter,
        fsdp_meta_load=runner._memory_efficient_fsdp_load,
        replicated_broadcast_load=runner._replicated_broadcast_load,
        build_meta_transformer=getattr(
            inner, "build_meta_transformer", missing_route("build_meta_transformer")
        ),
        build_meta_component=getattr(
            inner, "build_meta_component", missing_route("build_meta_component")
        ),
        plan_eager_blockwise_fallback=getattr(
            inner, "plan_eager_blockwise_fallback", lambda *args, **kwargs: None
        ),
        mark_local_blockwise=getattr(
            inner, "mark_local_blockwise", lambda component: None
        ),
        will_fill_blockwise=getattr(inner, "will_fill_blockwise", lambda name: False),
        meta_te_kwargs=getattr(inner, "meta_te_kwargs", lambda: None),
        meta_te_kwargs_replicated=getattr(
            inner,
            "meta_te_kwargs_replicated",
            missing_route("meta_te_kwargs_replicated"),
        ),
    )
    runner.loader = loader
    return loader


def preflight_loader(runner):
    """Build only the loader-owned runtime state needed by preflight tests."""
    for name in (
        "fp8_gemm_module_list",
        "fp8_text_encoder_module_list",
        "fp4_gemm_module_list",
        "int8_gemm_module_list",
        "fp8_precision_overrides",
        "fp8_precision_override_suffixes",
    ):
        if not hasattr(runner.settings, name):
            setattr(runner.settings, name, None)
    if not hasattr(runner.config, "use_fp8_text_encoder"):
        runner.config.use_fp8_text_encoder = False
    if not hasattr(runner.config, "use_hybrid_gemm_schedule"):
        runner.config.use_hybrid_gemm_schedule = False
    loader = object.__new__(ModelLoader)
    loader.model = runner
    loader.load_declaration = runner.load_declaration
    loader.load_contract = None
    loader.quantization_plan = QuantizationPlan(runner)
    loader.backends = SimpleNamespace(
        fp8=None,
        format=None,
        uses_blockwise_fp8=lambda: False,
        preflight=lambda: None,
    )
    return loader


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
        capabilities=SimpleNamespace(fully_shard_degree=True),
        load_support=(
            LoadSupport(
                meta_transformers=("transformer",),
                replicated_meta=True,
                routes=STANDARD_LOAD_ROUTES,
            )
            if supported
            else LoadSupport()
        ),
        config=SimpleNamespace(
            memory_efficient_replicated_load=requested,
            memory_efficient_sharding=memory_efficient_sharding,
            fully_shard_degree=fully_shard_degree,
            pipefusion_parallel_degree=pipefusion_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
        ),
    )
    model.settings.fsdp_strategy = {"transformer": {"wrap_attrs": ["blocks"]}}
    return ModelLoader(model)


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
    with pytest.raises(
        UnsupportedLoadContract,
        match="does not support replicated_meta materialization",
    ):
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


class FakeFp32Wrapper(FakeWrapper):
    """A wrapper that pins some modules to fp32, the way diffusers models do."""

    _keep_in_fp32_modules = ["norm2", "scale_shift_table"]

    @classmethod
    def from_config(cls, config, **kwargs):
        module = cls()
        hidden = config["hidden"]
        block = torch.nn.Module()
        block.attn = torch.nn.Linear(hidden, hidden)
        block.norm2 = torch.nn.LayerNorm(hidden)
        block.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 6, hidden))
        module.blocks = torch.nn.ModuleList([block])
        return module


def test_the_meta_transformer_keeps_the_wrappers_fp32_modules(monkeypatch):
    """The disk fill adopts each meta parameter's dtype, so demoting the modules
    diffusers pins to fp32 would silently round their checkpoint weights."""
    loader = make_loader(monkeypatch)

    built = loader.build_meta_transformer(FakeFp32Wrapper, subfolder="transformer")

    dtypes = dict(
        (name, param.dtype) for name, param in built.named_parameters()
    )
    assert dtypes["blocks.0.norm2.weight"] is torch.float32
    assert dtypes["blocks.0.scale_shift_table"] is torch.float32
    assert dtypes["blocks.0.attn.weight"] is torch.bfloat16


def test_the_meta_transformer_is_built_in_eval_mode(monkeypatch):
    """from_pretrained ends with eval(); from_config leaves nn.Module's training
    default, which would run inference with dropout active."""
    loader = make_loader(monkeypatch)

    built = loader.build_meta_transformer(FakeWrapper, subfolder="transformer")

    assert not built.training
    assert all(not module.training for module in built.modules())


def test_tracking_a_built_transformer_does_not_keep_it_alive(monkeypatch):
    """The bookkeeping must not pin a component the pipeline has replaced or dropped."""
    import gc

    loader = make_loader(monkeypatch)
    built = loader.build_meta_transformer(FakeWrapper, subfolder="transformer")
    assert len(loader._blockwise_sources) == 1
    del built
    gc.collect()
    assert len(loader._blockwise_sources) == 0


def test_custom_mapped_source_can_build_meta_only_for_local_fill(monkeypatch):
    loader = make_loader(monkeypatch, world_size=1)
    loader.load_declaration = LoadDeclaration(
        local_meta_transformers=("transformer",),
        routes=LoadRoute.LOCAL_BLOCKWISE,
    )
    source = CheckpointManifest(
        weight_map={"blocks.0.weight": "/weights/distilled.safetensors"}
    )

    built = loader.build_meta_transformer(
        FakeWrapper,
        subfolder="transformer",
        weight_source=source,
    )

    assert loader._blockwise_sources[built] is source


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

    monkeypatch.setattr(meta_load, "_BlockwiseDiskFiller", Filler)
    loader.build_blockwise_disk_loaders(built, ["blocks"], "transformer", "cpu")

    assert captured == [request]
    assert captured[0] is request


def test_dual_meta_transformers_keep_distinct_checkpoint_requests(monkeypatch):
    loader = make_loader(monkeypatch)
    loader.load_declaration = LoadDeclaration.meta(
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

    monkeypatch.setattr(meta_load, "_BlockwiseDiskFiller", Filler)
    loader.build_blockwise_disk_loaders(first, ["blocks"], "transformer", "cpu")
    loader.build_blockwise_disk_loaders(second, ["blocks"], "transformer_2", "cpu")

    assert captured == [(first, first_request), (second, second_request)]


def resolved_class_load_declaration(cls):
    return LoadDeclaration.for_runner(
        cls.capabilities,
        load_support=cls.load_support,
        fsdp_strategy=cls.settings.fsdp_strategy,
    )
def test_runner_load_declarations_match_model_quantization_capabilities():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        expected = LoadDeclaration.for_runner(cls.capabilities).quantization_contracts
        if resolved_class_load_declaration(cls).quantization_contracts != expected:
            mismatched.append(cls.__name__)

    assert not mismatched, (
        "load_declaration quantization contracts disagree with ModelCapabilities: "
        + ", ".join(sorted(mismatched))
    )


def test_runner_fsdp_meta_support_matches_capabilities_and_strategy():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        declaration = resolved_class_load_declaration(cls)
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

    monkeypatch.setattr(meta_load, "_use_aiter_fp8_rdna4", lambda: True)
    monkeypatch.setattr(meta_load, "_is_cuda", lambda: False)
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
        load_declaration=LoadDeclaration.for_runner(
            model_capabilities,
            load_support=LoadSupport(
                meta_transformers=("transformer",),
                replicated_meta=True,
                routes=STANDARD_LOAD_ROUTES,
            ),
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        ),
    )

    loader = preflight_loader(runner)
    loader.preflight(world_size=2)
    selected = loader.load_contract

    assert selected.requested_format.name == "FP8"
    assert selected.selected_backend.name == "AITER"
    assert selected.materialization_mode.name == "FSDP_META"


def test_base_runner_rejects_unsupported_meta_mode_before_loading(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(meta_load, "_use_aiter_fp8_rdna4", lambda: False)
    monkeypatch.setattr(meta_load, "_is_cuda", lambda: True)
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
        load_declaration=LoadDeclaration.for_runner(
            base_model.ModelCapabilities(),
            load_support=LoadSupport(),
        ),
    )

    with pytest.raises(
        UnsupportedLoadContract,
        match="does not support replicated_meta materialization",
    ):
        preflight_loader(runner).preflight(world_size=2)


def test_base_runner_uses_effective_single_rank_mode(monkeypatch):
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(meta_load, "_use_aiter_fp8_rdna4", lambda: False)
    monkeypatch.setattr(meta_load, "_is_cuda", lambda: True)
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
        load_declaration=LoadDeclaration.for_runner(
            base_model.ModelCapabilities(),
            load_support=LoadSupport(),
        ),
    )

    loader = preflight_loader(runner)
    loader.preflight(world_size=1)
    selected = loader.load_contract

    assert selected.materialization_mode.name == "EAGER"


def test_wan22_spec_resolves_after_dynamic_instance_settings(monkeypatch):
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
    monkeypatch.setattr(meta_load, "_use_aiter_fp8_rdna4", lambda: True)
    monkeypatch.setattr(meta_load, "_is_cuda", lambda: False)

    loader = ModelLoader(runner)
    loader.backends = SimpleNamespace(
        fp8=None,
        format=None,
        uses_blockwise_fp8=lambda: False,
        preflight=lambda: None,
    )
    loader.preflight(world_size=2)
    fsdp_selected = loader.load_contract

    assert loader.load_declaration.fsdp_meta_transformers == (
        "transformer",
        "transformer_2",
    )
    assert loader.load_declaration.replicated_meta_transformers == (
        "transformer",
        "transformer_2",
    )
    assert fsdp_selected.materialization_mode.name == "FSDP_META"

    runner.config.fully_shard_degree = 1
    runner.config.memory_efficient_sharding = False
    runner.config.memory_efficient_replicated_load = True
    # One loader owns one immutable preflight result; a changed run gets a new loader.
    loader = ModelLoader(runner)
    loader.backends = SimpleNamespace(
        fp8=None,
        format=None,
        uses_blockwise_fp8=lambda: False,
        preflight=lambda: None,
    )
    loader.preflight(world_size=2)
    replicated_selected = loader.load_contract

    assert replicated_selected.materialization_mode.name == "REPLICATED_META"


def test_build_transformer_preserves_request_subfolder_without_override():

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

    result = transformer_load.load_transformer(loader_for(runner), Wrapper, checkpoint_request=request
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

    result = transformer_load.build_transformer_structure(
        Wrapper,
        CheckpointRequest("org/repo", subfolder="transformer"),
        None,
    )

    assert result == "structure"
    assert calls == [{"include_buffers": True}]


def test_structure_inspection_reports_old_accelerate_explicitly(monkeypatch):
    import accelerate


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
        transformer_load.build_transformer_structure(
            Wrapper,
            CheckpointRequest("org/repo", subfolder="transformer"),
            None,
        )


def test_structure_inspection_catches_include_buffers_error_on_context_enter(
    monkeypatch,
):
    from contextlib import contextmanager
    import accelerate


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
        transformer_load.build_transformer_structure(
            Wrapper,
            CheckpointRequest("org/repo", subfolder="transformer"),
            None,
        )


def test_build_transformer_routes_torchao_fp8_to_native_diffusers_config(
    monkeypatch,
):
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

    monkeypatch.setattr(
        transformer_load, "build_transformer_structure", structure_factory
    )
    monkeypatch.setattr(
        transformer_load, "native_quantization_device_map", lambda *args: {"": 0}
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

    assert result == "streamed"
    assert calls[0]["quantization_config"] is sentinel
    assert runner.quantization_ledger.fp8_streaming_targets == {"transformer.blocks"}


def test_blockwise_transformer_marks_only_wrapped_target_as_streamed(
    monkeypatch,
):
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
        _loader=SimpleNamespace(build_meta_transformer=lambda *args, **kwargs: "meta"),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), SimpleNamespace())

    assert result == "meta"
    assert runner.quantization_ledger.fp8_streaming_targets == {"transformer.blocks"}
    assert runner.quantization_ledger.streaming_targets == {"transformer.blocks"}


def test_blockwise_fp4_marks_only_wrapped_fp8_remainder_as_streamed(
    monkeypatch,
):
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
        _loader=SimpleNamespace(build_meta_transformer=lambda *args, **kwargs: "meta"),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), SimpleNamespace())

    assert result == "meta"
    assert runner.quantization_ledger.fp8_streaming_targets == {"transformer.blocks"}
    assert runner.quantization_ledger.streaming_targets == {"transformer.blocks"}


def test_build_transformer_logs_explicit_torchao_post_load_fallback(monkeypatch):
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
    monkeypatch.setattr(transformer_load, "log", logs.append)

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

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

    monkeypatch.setattr(
        transformer_load, "build_transformer_structure", structure_factory
    )
    monkeypatch.setattr(transformer_load, "log", logs.append)

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

    assert result == "loaded"
    assert calls[0]["quantization_config"] is None
    assert any(
        "backend=torchao" in message
        and "materialization=post_load" in message
        and "target mapping unavailable" in message
        for message in logs
    )


def test_eager_post_load_fallback_builds_meta_for_local_block_fill(monkeypatch):
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
    monkeypatch.setattr(transformer_load, "log", lambda message: None)

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

    assert result is meta_component
    assert marked == [meta_component]
    assert runner.quantization_ledger.streaming_targets == {"transformer.blocks"}


def test_build_transformer_preserves_aiter_native_streaming(monkeypatch):
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
    monkeypatch.setattr(
        transformer_load, "native_quantization_device_map", lambda *args: {"": 0}
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        fp8=SimpleNamespace(targets_for=lambda name: ["blocks"]),
        fp8_backend=adapter,
        settings=SimpleNamespace(fsdp_strategy={}),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

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

    monkeypatch.setattr(
        transformer_load,
        "build_transformer_structure",
        lambda *args, **kwargs: SimpleNamespace(
            named_modules=lambda: [
                ("", object()),
                ("blocks", object()),
                ("blocks.0.proj", torch.nn.Linear(1024, 1024)),
            ],
            get_submodule=lambda name: object(),
        ),
    )
    monkeypatch.setattr(
        transformer_load, "native_quantization_device_map", lambda *args: {"": 0}
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        _transformer_quantization_adapter=lambda component: (
            adapter,
            ("blocks",),
        ),
        settings=SimpleNamespace(
            fsdp_strategy={},
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

    assert result == "streamed"
    assert calls[0]["quantization_config"] is sentinel
    assert calls[0]["device_map"] == {"": 0}
    assert runner.quantization_ledger.streaming_targets == {"transformer.blocks"}


def test_build_transformer_records_only_streamed_nvfp4_leaves(monkeypatch):
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
    monkeypatch.setattr(
        transformer_load, "build_transformer_structure", lambda *args, **kwargs: structure
    )
    monkeypatch.setattr(
        transformer_load, "native_quantization_device_map", lambda *args: {"": 0}
    )
    runner = SimpleNamespace(
        _memory_efficient_fsdp_load=lambda: False,
        _replicated_broadcast_load=lambda: False,
        _transformer_quantization_adapter=lambda component: (
            adapter,
            ("blocks",),
        ),
        settings=SimpleNamespace(
            fsdp_strategy={},
            fp8_precision_overrides=("0.override",),
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        _checkpoint_request=lambda name: CheckpointRequest("org/repo", subfolder=name),
    )

    result = transformer_load.load_transformer(loader_for(runner), Wrapper)

    assert result == "streamed"
    assert runner.quantization_ledger.streaming_targets == {"transformer.blocks.0.keep"}



def _attach_backends(runner):
    """Give a fake runner the quantization selector the loader reads adapters through."""
    from xfuser.model_executor.models.runner_models.loading.backend_selection import (
        QuantizationBackends,
    )

    loader = loader_for(runner)
    backends = QuantizationBackends(loader)
    backends.__dict__.update(
        fp8=getattr(runner, "fp8_backend", None),
        blockwise_fp8=getattr(runner, "blockwise_fp8_backend", None),
        format=getattr(runner, "format_backend", None),
    )
    loader.backends = backends
    runner.backends = backends
    return runner

def test_eager_te_adapter_maps_multiple_components_and_logs_each(monkeypatch):

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
    monkeypatch.setattr(text_encoder_plan, "log", logs.append)

    _attach_backends(runner)
    kwargs, config = text_encoder_plan.plan_text_encoders(loader_for(runner))

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
    assert runner.quantization_ledger.fp8_streaming_targets == {
        "text_encoder.encoder.block",
        "text_encoder_2.model.layers",
    }


def test_hybrid_meta_te_uses_blockwise_fp8_backend(monkeypatch):

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
            will_fill_blockwise=lambda name: False,
        ),
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
    monkeypatch.setattr(text_encoder_plan, "log", lambda message: None)

    _attach_backends(runner)
    kwargs, config = text_encoder_plan.plan_text_encoders(loader_for(runner))

    assert (kwargs, config) == ({"text_encoder": "meta"}, None)
    assert observed == [sentinel]


def test_meta_te_placement_disables_torchao_native_pipeline_streaming(
    monkeypatch,
):

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
            will_fill_blockwise=lambda name: False,
        ),
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
    monkeypatch.setattr(text_encoder_plan, "log", lambda message: None)

    _attach_backends(runner)
    kwargs, config = text_encoder_plan.plan_text_encoders(loader_for(runner))

    assert (kwargs, config) == ({"text_encoder": "meta"}, None)
    assert observed == [(False, False)]
    assert runner.quantization_ledger.fp8_streaming_targets == set()


def test_a_blockwise_filled_text_encoder_needs_no_post_load_fallback(monkeypatch):
    """The refusal below objects to converting a layout after FSDP wraps it.

    A blockwise-filled encoder is quantized per block on the way in from disk, before wrapping, so
    the objection does not apply and TorchAO can quantize a text encoder on the FSDP meta path. This
    is the case the blockwise fill exists for, so reaching prepare_text_encoder_fp8_load here would
    mean the encoder had been routed back to the whole-encoder rank0 load.
    """

    adapter = SimpleNamespace(
        backend=SimpleNamespace(value="torchao"),
        format=SimpleNamespace(value="fp8"),
        storage_semantics="",
    )
    runner = SimpleNamespace(
        load_contract=SimpleNamespace(requested_format=SimpleNamespace(value="fp8")),
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: True,
        fp8_backend=adapter,
        config=SimpleNamespace(use_fp4_gemms=False),
        fp8=SimpleNamespace(targets_for=lambda name: ["encoder.block"]),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=["text_encoder.encoder.block"],
            fsdp_strategy={"text_encoder": {"wrap_attrs": ["encoder.block"]}},
        ),
        _loader=SimpleNamespace(
            meta_te_kwargs=lambda: ({"text_encoder": "meta"}, None),
            build_meta_component=lambda name, fp8=False: object(),
            will_fill_blockwise=lambda name: True,
        ),
    )

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends."
        "prepare_text_encoder_fp8_load",
        lambda *a, **k: pytest.fail(
            "a blockwise-filled encoder was routed to the whole-encoder rank0 load"
        ),
    )
    monkeypatch.setattr(text_encoder_plan, "log", lambda message: None)

    _attach_backends(runner)
    kwargs, config = text_encoder_plan.plan_text_encoders(loader_for(runner))

    assert (kwargs, config) == ({"text_encoder": "meta"}, None)
    # Recorded as already quantized, so the post-load walk leaves the filled blocks alone.
    assert "text_encoder" in runner.quantization_ledger.fp8_descriptor_components
    assert runner.quantization_ledger.fp8_streaming_targets


def test_meta_fsdp_rejects_text_encoder_post_load_fallback(monkeypatch):

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
            will_fill_blockwise=lambda name: False,
        ),
    )

    def prepare(adapter, **kwargs):
        if not kwargs["supports_post_load"]:
            raise RuntimeError("text_encoder FP8 cannot fall back before allocation")
        pytest.fail("memory-efficient FSDP incorrectly allowed post-load")

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends.prepare_text_encoder_fp8_load",
        prepare,
    )

    _attach_backends(runner)
    with pytest.raises(RuntimeError, match="before allocation"):
        text_encoder_plan.plan_text_encoders(loader_for(runner))
