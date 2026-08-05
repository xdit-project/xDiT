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
from xfuser.model_executor.models.runner_models.loading.meta_load import MemoryEfficientLoader


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
    monkeypatch.setattr(meta_load, "get_world_group", lambda: SimpleNamespace(world_size=world_size))
    model = SimpleNamespace(
        settings=SimpleNamespace(model_name="stand-in/checkpoint"),
        config=SimpleNamespace(
            memory_efficient_replicated_load=requested,
            memory_efficient_sharding=memory_efficient_sharding,
            fully_shard_degree=fully_shard_degree,
            pipefusion_parallel_degree=pipefusion_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
        ),
        _supports_replicated_meta_load=lambda: supported,
    )
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


def test_never_broadcasts_for_a_runner_that_is_not_wired_for_it(monkeypatch):
    """A runner loading its components directly leaves peers with no meta tensors to fill."""
    loader = make_loader(monkeypatch, supported=False)
    assert not loader.replicated_broadcast_load()


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
    def load_config(cls, model_name, subfolder=None):
        return {"hidden": 4}

    @classmethod
    def from_config(cls, config, **kwargs):
        module = cls()
        module.blocks = torch.nn.ModuleList([torch.nn.Linear(config["hidden"], config["hidden"])])
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


# ============================================================================
# Every runner's declared capability matches how it actually loads
# ============================================================================


def test_runners_that_bypass_the_meta_seam_declare_it():
    """_supports_replicated_meta_load is hand-maintained, so it can drift from the load path it
    describes. A runner whose _load_model does its own from_pretrained gets no meta components for
    the rank0 broadcast to fill; it must opt out, or it claims a win it cannot take (and logs that
    it is taking it). Checked over the whole registry so a new runner cannot quietly skip it."""
    import inspect

    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    mismatched = []
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        # _load_model may be inherited; read the source of whichever class actually defines it.
        goes_through_seam = "_build_transformer" in inspect.getsource(cls._load_model)
        declares_support = cls._supports_replicated_meta_load(object.__new__(cls))
        if declares_support and not goes_through_seam:
            mismatched.append(f"{cls.__name__} (from {cls._load_model.__qualname__})")

    assert not mismatched, (
        "these runners load their transformer outside _build_transformer but still declare "
        "_supports_replicated_meta_load: " + ", ".join(sorted(mismatched))
    )
