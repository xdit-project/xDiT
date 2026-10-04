"""Runner loads that bypass the shared loader still have to follow HF_HUB_OFFLINE.

Diffusers asks the hub which shards a sharded checkpoint has unless ``local_files_only`` is
set, so a fully cached sharded checkpoint still fails under HF_HUB_OFFLINE=1 when the runner
does not pass it.
"""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import diffusers
import pytest

from xfuser.model_executor.models.runner_models import ideogram4
from xfuser.model_executor.models.runner_models.hunyuan import (
    xFuserHunyuanvideo15Model,
)
from xfuser.model_executor.models.runner_models.ideogram4 import xFuserIdeogram4Model
from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader


def _runner(model_cls, model_name, **config):
    model = object.__new__(model_cls)
    model.settings = copy.deepcopy(model_cls.settings)
    model.settings.model_name = model_name
    model.config = SimpleNamespace(**config)
    model.loader = object.__new__(ModelLoader)
    model.loader.model = model
    return model


@pytest.fixture(params=[True, False], ids=["offline", "online"])
def hub_offline(request, monkeypatch):
    # The runner logs from the last rank of a single-process run.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", request.param)
    return request.param


def test_hunyuanvideo15_transformer_load_follows_offline_mode(hub_offline, monkeypatch):
    from xfuser.model_executor.models.transformers import (
        transformer_hunyuan_video15,
    )

    load_transformer = MagicMock(name="transformer.from_pretrained")
    load_pipeline = MagicMock(name="pipeline.from_pretrained")
    monkeypatch.setattr(
        transformer_hunyuan_video15.xFuserHunyuanVideo15Transformer3DWrapper,
        "from_pretrained",
        load_transformer,
    )
    monkeypatch.setattr(diffusers.HunyuanVideo15Pipeline, "from_pretrained", load_pipeline)
    repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    model = _runner(xFuserHunyuanvideo15Model, repo, task="t2v")

    assert model._load_model() is load_pipeline.return_value

    load_transformer.assert_called_once()
    args, kwargs = load_transformer.call_args
    assert args == (repo,)
    assert kwargs["subfolder"] == "transformer"
    assert kwargs.get("local_files_only", False) is hub_offline


def test_ideogram4_bf16_transformer_loads_follow_offline_mode(hub_offline, monkeypatch):
    from xfuser.model_executor.models.transformers import transformer_ideogram4

    transformer_class = MagicMock(name="transformer_class")
    pipeline_class = MagicMock(name="pipeline_class")
    monkeypatch.setattr(ideogram4, "_is_fp8_checkpoint", lambda model_id: False)
    monkeypatch.setattr(
        transformer_ideogram4,
        "get_ideogram4_transformer_wrapper_class",
        lambda: transformer_class,
    )
    monkeypatch.setattr(ideogram4, "get_ideogram4_pipeline_class", lambda: pipeline_class)
    # Keep the optional prompt enhancer head off the hub.
    monkeypatch.setattr(diffusers, "Ideogram4PromptEnhancerHead", MagicMock(), raising=False)
    repo = "CalamitousFelicitousness/Ideogram-4-bf16-Diffusers"
    model = _runner(xFuserIdeogram4Model, repo)

    assert model._load_model() is pipeline_class.from_pretrained.return_value

    calls = transformer_class.from_pretrained.call_args_list
    assert [call.kwargs["subfolder"] for call in calls] == [
        "transformer",
        "unconditional_transformer",
    ]
    for call in calls:
        assert call.args == (repo,)
        assert call.kwargs.get("local_files_only", False) is hub_offline
