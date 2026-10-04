"""With a step cache, torch.compile must leave the transformer itself uncompiled.

The cache is applied after compilation and patches transformer.forward. A compiled
transformer would either trace that patch into its graph or hide it behind the
compiled wrapper, so the runner compiles the transformer's blocks one by one.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.runner_models import hunyuan
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


class _Compiled(torch.nn.Module):
    """What the stand-in torch.compile returns: a marker around the module it was given."""

    def __init__(self, module):
        super().__init__()
        self.module = module


@pytest.fixture
def compiled(monkeypatch):
    """Replace torch.compile and record every module handed to it."""
    seen = []

    def fake_compile(module, *args, **kwargs):
        seen.append(module)
        return _Compiled(module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(hunyuan, "install_inductor_passes", lambda: None)
    return seen


def _transformer(*block_attrs):
    transformer = torch.nn.Module()
    for attr in block_attrs:
        setattr(transformer, attr, torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)]))
    return transformer


def _runner(cls, transformer, cache_method):
    model = object.__new__(cls)
    model.config = SimpleNamespace(cache_method=cache_method, fully_shard_degree=1, profile_capture_phase=False)
    model.settings = copy.deepcopy(cls.settings)
    model.pipe = SimpleNamespace(transformer=transformer)
    model._enable_compute_comm_overlap = lambda: None
    model._run_timed_pipe = lambda input_args: None
    model._run_compile_warmup = lambda input_args: None
    model._get_compile_warmup_steps = lambda input_args: 2
    return model


@pytest.mark.parametrize(
    "cls, block_attrs",
    [
        pytest.param(hunyuan.xFuserHunyuanvideo15Model, ("transformer_blocks",), id="hunyuanvideo-1.5"),
        pytest.param(
            hunyuan.xFuserHunyuanvideoModel,
            ("transformer_blocks", "single_transformer_blocks"),
            id="hunyuanvideo",
        ),
    ],
)
def test_step_cache_compiles_blocks_not_the_transformer(compiled, cls, block_attrs):
    transformer = _transformer(*block_attrs)
    blocks = [block for attr in block_attrs for block in getattr(transformer, attr)]
    model = _runner(cls, transformer, cache_method="dbcache")

    model._compile_model({"num_inference_steps": 50})

    assert model.pipe.transformer is transformer
    assert compiled == blocks
    for attr in block_attrs:
        assert all(isinstance(block, _Compiled) for block in getattr(transformer, attr))


def test_hunyuanvideo_15_without_a_step_cache_still_compiles_the_transformer(compiled, monkeypatch):
    monkeypatch.setattr(xFuserModel, "_enable_compute_comm_overlap", lambda self: None)
    transformer = _transformer("transformer_blocks")
    model = _runner(hunyuan.xFuserHunyuanvideo15Model, transformer, cache_method=None)

    model._compile_model({"num_inference_steps": 50})

    assert compiled == [transformer]
    assert model.pipe.transformer.module is transformer


class _Runner(xFuserModel):
    """The smallest concrete runner, since compilation is defined on the base class."""

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def test_step_cache_refuses_to_compile_a_transformer_whose_blocks_it_cannot_find(compiled):
    transformer = _transformer("layers")
    model = _runner(_Runner, transformer, cache_method="dbcache")
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}})
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda: None
    model._get_compiled_pipe_components = lambda: ["transformer"]

    with pytest.raises(ValueError, match="wrap_attrs"):
        model._compile_model({"num_inference_steps": 50})

    assert compiled == []
    assert model.pipe.transformer is transformer
