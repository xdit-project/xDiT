"""With a step cache, torch.compile must leave the transformer itself uncompiled.

The cache is applied after compilation and patches transformer.forward, so the runner
compiles the transformer's blocks one by one instead.
"""

from types import SimpleNamespace

import pytest
import torch
import torch._inductor.config as inductor_config

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models import base_model, hunyuan

HUNYUANVIDEO_15 = pytest.param(hunyuan.xFuserHunyuanvideo15Model, "t2v", ("transformer_blocks",), id="hunyuanvideo-1.5")
HUNYUANVIDEO = pytest.param(
    hunyuan.xFuserHunyuanvideoModel,
    None,
    ("transformer_blocks", "single_transformer_blocks"),
    id="hunyuanvideo",
)


class _Compiled(torch.nn.Module):
    """What the stand-in torch.compile returns: a marker around what it was given."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _runtime_state(has_attention_schedule):
    return lambda: SimpleNamespace(has_attention_schedule=lambda: has_attention_schedule)


@pytest.fixture
def boundaries(monkeypatch):
    """Stand in for torch.compile, the pipeline run and the distributed state.

    Records what torch.compile was handed and the arguments of every warmup pipeline call.
    """
    compiled, warmups = [], []

    def fake_compile(module, *args, **kwargs):
        compiled.append(module)
        return _Compiled(module)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(base_model.xFuserModel, "_run_timed_pipe", lambda self, args: warmups.append(args))
    monkeypatch.setattr(base_model, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_runtime_state", _runtime_state(False))
    # The runner sets global Inductor options; restore them after each test.
    for name in (
        "post_grad_custom_post_pass",
        "reorder_for_compute_comm_overlap",
        "reorder_for_compute_comm_overlap_passes",
    ):
        monkeypatch.setattr(inductor_config, name, getattr(inductor_config, name))
    return SimpleNamespace(compiled=compiled, warmups=warmups)


def _transformer(*block_attrs):
    transformer = torch.nn.Module()
    for attr in block_attrs:
        setattr(transformer, attr, torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)]))
    return transformer


def _runner(cls, task, transformer, cache_method):
    model = cls(xFuserArgs(model="test-model", task=task, use_torch_compile=True, cache_method=cache_method))
    model.pipe = SimpleNamespace(transformer=transformer)
    return model


def _warmup_steps(boundaries):
    return [args["num_inference_steps"] for args in boundaries.warmups]


@pytest.mark.parametrize("cls, task, block_attrs", [HUNYUANVIDEO_15, HUNYUANVIDEO])
def test_step_cache_compiles_blocks_not_the_transformer(boundaries, cls, task, block_attrs):
    transformer = _transformer(*block_attrs)
    blocks = [block for attr in block_attrs for block in getattr(transformer, attr)]
    model = _runner(cls, task, transformer, cache_method="dbcache")

    model._compile_model({"num_inference_steps": 50})

    assert model.pipe.transformer is transformer
    assert boundaries.compiled == blocks
    for attr in block_attrs:
        assert all(isinstance(block, _Compiled) for block in getattr(transformer, attr))
    assert _warmup_steps(boundaries) == [2]


def test_hunyuanvideo_15_without_a_step_cache_still_compiles_the_transformer(boundaries):
    transformer = _transformer("transformer_blocks")
    model = _runner(hunyuan.xFuserHunyuanvideo15Model, "t2v", transformer, cache_method=None)

    model._compile_model({"num_inference_steps": 50})

    assert boundaries.compiled == [transformer]
    assert model.pipe.transformer.module is transformer
    assert _warmup_steps(boundaries) == [2]


@pytest.mark.parametrize("cache_method", [None, "dbcache"])
def test_hunyuanvideo_attention_schedule_warms_up_every_step(boundaries, monkeypatch, cache_method):
    monkeypatch.setattr(hunyuan, "get_runtime_state", _runtime_state(True))
    transformer = _transformer("transformer_blocks", "single_transformer_blocks")
    model = _runner(hunyuan.xFuserHunyuanvideoModel, None, transformer, cache_method=cache_method)

    model._compile_model({"num_inference_steps": 50})

    assert _warmup_steps(boundaries) == [50]


def test_step_cache_refuses_to_compile_a_transformer_whose_blocks_it_cannot_find(boundaries):
    transformer = _transformer("layers")
    model = _runner(hunyuan.xFuserHunyuanvideo15Model, "t2v", transformer, cache_method="dbcache")

    with pytest.raises(ValueError, match="wrap_attrs"):
        model._compile_model({"num_inference_steps": 50})

    assert boundaries.compiled == []
    assert model.pipe.transformer is transformer
    assert boundaries.warmups == []
