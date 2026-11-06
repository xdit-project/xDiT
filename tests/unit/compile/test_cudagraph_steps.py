"""Compiling a component blockwise must tell CUDA Graphs where each step ends.

Without a step boundary, inference fails with "accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run". It needs both reduce-overhead and sharding to appear: sharding is
what turns one compiled transformer into one compiled graph per block, so the two together produce
graph segments that read each other's buffers across steps. One rank, or no sharding, is immune.
"""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.runner_models.base_model import xFuserModel


class _Runner(xFuserModel):
    """The smallest concrete runner, since compilation is defined on the base class."""

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def _model(mode: str, *, fully_shard_degree: int):
    blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    transformer = torch.nn.Module()
    transformer.blocks = blocks
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        fully_shard_degree=fully_shard_degree,
        profile_capture_phase=False,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=transformer)
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}})
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: mode
    model._get_compile_dynamic = lambda: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda input_args: None
    model._run_timed_pipe = lambda input_args: None
    return model, transformer


def test_blockwise_compilation_under_cuda_graphs_marks_step_boundaries():
    model, transformer = _model("reduce-overhead", fully_shard_degree=8)

    model._compile_model({"num_inference_steps": 4})

    assert transformer._xfuser_marks_cudagraph_steps
    assert len(transformer._forward_pre_hooks) == 1


def test_the_marker_actually_announces_the_step_to_torch(monkeypatch):
    model, transformer = _model("reduce-overhead", fully_shard_degree=8)
    announced = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: announced.append(True))

    model._compile_model({"num_inference_steps": 4})
    hook = next(iter(transformer._forward_pre_hooks.values()))
    hook(transformer, (), {})

    assert announced == [True]


@pytest.mark.parametrize(
    "mode,fully_shard_degree",
    [
        # No CUDA graphs, so there are no reused buffers to announce a boundary for.
        ("default", 8),
        # One graph for the whole component: its own output is the only thing anyone reads.
        ("reduce-overhead", 1),
    ],
)
def test_no_marker_where_there_is_nothing_to_mark(mode, fully_shard_degree):
    model, transformer = _model(mode, fully_shard_degree=fully_shard_degree)

    model._compile_model({"num_inference_steps": 4})

    assert not getattr(transformer, "_xfuser_marks_cudagraph_steps", False)
    assert not transformer._forward_pre_hooks


def test_marking_a_component_twice_leaves_one_marker():
    """Two announcements per step would be harmless but would hide who registered them."""
    model, transformer = _model("reduce-overhead", fully_shard_degree=8)

    model._mark_cudagraph_steps(transformer)
    model._mark_cudagraph_steps(transformer)

    assert len(transformer._forward_pre_hooks) == 1


@pytest.mark.parametrize(
    "requested,ulysses_degree,expected",
    [
        ("reduce-overhead", 8, "default"),
        ("max-autotune", 8, "max-autotune-no-cudagraphs"),
        ("reduce-overhead", 1, "reduce-overhead"),
    ],
)
def test_sdma_ulysses_compiles_without_cuda_graphs(monkeypatch, requested, ulysses_degree, expected):
    """SDMA copies are issued from the host, so a captured graph would replay RCCL instead."""
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(base_model, "get_ulysses_parallel_world_size", lambda: ulysses_degree)
    previous = usp_sdma.configure("sdma", 2)
    try:
        model, transformer = _model(requested, fully_shard_degree=8)
        assert model._compile_mode() == expected
        if requested == "reduce-overhead":
            model._compile_model({"num_inference_steps": 4})
            assert getattr(transformer, "_xfuser_marks_cudagraph_steps", False) == (expected == "reduce-overhead")
    finally:
        usp_sdma.configure(*previous)


@pytest.mark.parametrize(
    "offload,prepared",
    [(None, True), ("enable_model_cpu_offload", False), ("enable_group_cpu_offload", False)],
)
def test_grouped_qkv_weights_are_not_made_under_cpu_offload(monkeypatch, offload, prepared):
    """Offload hooks set up before the copies would never move them to the device."""
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.models.runner_models import base_model

    monkeypatch.setattr(base_model, "get_ulysses_parallel_world_size", lambda: 8)
    calls, logged = [], []
    monkeypatch.setattr(usp_sdma, "prepare_grouped_qkv", lambda component, world: calls.append(world) or 1)
    monkeypatch.setattr(base_model, "log", lambda message, **kwargs: logged.append(message))
    previous = usp_sdma.configure("sdma", 2, overlap_qkv=True)
    try:
        model, _ = _model("default", fully_shard_degree=1)
        for flag in ("enable_model_cpu_offload", "enable_sequential_cpu_offload", "enable_group_cpu_offload"):
            setattr(model.config, flag, flag == offload)
        model._prepare_grouped_qkv()
        assert calls == ([8] if prepared else [])
        assert any("ignored with CPU offload" in m for m in logged) == (not prepared)
    finally:
        usp_sdma.configure(*previous)
