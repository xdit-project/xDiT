from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.layers.transposed_linear import TransposedLinear
from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import ModelFeatures, xFuserModel
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFlux2Klein9BModel,
    xFuserFlux2Model,
    xFuserFluxKontextModel,
    xFuserFluxModel,
)
from xfuser.model_executor.models.runner_models.qwen import (
    xFuserQwenImage21Model,
    xFuserQwenImage21TurboModel,
    xFuserQwenImageModel,
)
from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel, xFuserZImageTurboModel


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


class _Runner(xFuserModel):
    features = ModelFeatures(transposes_linears=True)

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def _runner(**config):
    runner = object.__new__(_Runner)
    runner.config = SimpleNamespace(
        fully_shard_degree=1,
        memory_efficient_sharding=False,
        enable_group_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
        lora_path=None,
    )
    vars(runner.config).update(config)
    runner.pipe = SimpleNamespace(transformer=torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 4)))
    return runner


def _transposed(runner):
    return [type(linear) is TransposedLinear for linear in runner.pipe.transformer]


@pytest.mark.parametrize(("arch", "expected"), [("gfx942", True), ("gfx950", False), ("", False)])
def test_only_gfx942_stores_the_weights_transposed(monkeypatch, arch, expected):
    monkeypatch.setattr(base_model, "device_arch", lambda: arch)
    runner = _runner()

    runner._transpose_linears()

    assert _transposed(runner) == [expected] * 2


@pytest.mark.parametrize(
    "config",
    [
        {"fully_shard_degree": 2},
        {"memory_efficient_sharding": True},
        {"enable_group_cpu_offload": True},
        {"enable_sequential_cpu_offload": True},
        {"enable_model_cpu_offload": True},
        {"lora_path": "lora.safetensors"},
    ],
)
def test_sharded_or_offloaded_weights_and_loras_keep_their_layout(monkeypatch, config):
    monkeypatch.setattr(base_model, "device_arch", lambda: "gfx942")
    runner = _runner(**config)

    runner._transpose_linears()

    assert _transposed(runner) == [False] * 2


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (xFuserZImageModel, True),
        (xFuserFlux2Model, True),
        (xFuserFluxModel, True),
        (xFuserQwenImage21Model, True),
        (xFuserQwenImage21TurboModel, True),
        # Measured within noise or slower.
        (xFuserFlux2Klein9BModel, False),
        (xFuserFluxKontextModel, False),
        (xFuserQwenImageModel, False),
        # Not measured.
        (xFuserZImageTurboModel, False),
    ],
)
def test_which_runners_transpose(model_cls, expected):
    assert model_cls.features.transposes_linears is expected


def test_the_post_load_step_transposes_once_the_weights_are_placed(monkeypatch):
    monkeypatch.setattr(base_model, "device_arch", lambda: "gfx942")
    runner = _runner(
        use_hybrid_attn_schedule=False,
        use_hybrid_gemm_schedule=False,
        solattn_beta_schedule=None,
        use_vae_channels_last_format=False,
    )
    placed = []
    runner.loader = SimpleNamespace(materialize_pipeline=lambda: placed.append(_transposed(runner)))

    runner._post_load_and_state_initialization({})

    assert placed == [[False] * 2]
    assert _transposed(runner) == [True] * 2
