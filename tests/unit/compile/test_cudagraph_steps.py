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
        # One graph for the whole component: its outputs are copied out instead.
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


class _GraphPool(torch.nn.Module):
    """Returns the same buffer from every call, as a replayed CUDA graph returns its pool."""

    def __init__(self):
        super().__init__()
        self.out = torch.zeros(2)

    def forward(self, value):
        self.out.fill_(value)
        return (self.out,)


@pytest.mark.parametrize(("mode", "kept"), [("reduce-overhead", [1.0, 1.0]), ("default", [2.0, 2.0])])
def test_a_whole_component_under_cuda_graphs_returns_outputs_the_next_call_cannot_overwrite(monkeypatch, mode, kept):
    """Sequential true CFG reads the conditional output after the unconditional call."""
    model, _ = _model(mode, fully_shard_degree=1)
    model.pipe.transformer = _GraphPool()
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)

    model._compile_model({"num_inference_steps": 4})
    conditional = model.pipe.transformer(1.0)[0]
    model.pipe.transformer(2.0)

    assert conditional.tolist() == kept


def test_sdma_ulysses_drops_cuda_graphs_for_the_transformer_only(monkeypatch):
    """The SDMA copies are queued from the host: the transformer compiles without CUDA graphs,
    the other components keep them."""
    from xfuser.config.ulysses_a2a import UlyssesA2AConfig
    from xfuser.core.distributed import runtime_state
    from xfuser.model_executor.layers import usp_sdma

    config = UlyssesA2AConfig("sdma", 2)
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(ulysses_a2a=config),
        has_attention_schedule=lambda: False,
        has_gemm_schedule=lambda: False,
    )
    monkeypatch.setattr(runtime_state, "_RUNTIME", state)
    monkeypatch.setattr(usp_sdma, "get_ulysses_parallel_world_size", lambda: 8)
    model, transformer = _model("reduce-overhead", fully_shard_degree=1)
    model.pipe.vae = vae = torch.nn.Linear(4, 4)
    model.settings.transformer_attr_names = ["transformer"]
    model._get_compiled_pipe_components = lambda: ["transformer", "vae"]
    modes = {}

    def compile(module, **kwargs):
        modes[id(module)] = kwargs["mode"]
        return module

    monkeypatch.setattr(torch, "compile", compile)

    model._compile_model({"num_inference_steps": 4})

    assert modes[id(transformer)] == "default"
    assert modes[id(vae)] == "reduce-overhead"
