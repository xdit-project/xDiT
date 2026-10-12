"""Load real sharded Diffusers weights from a Hub cache without network access."""

import json
import socket
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")

from diffusers.configuration_utils import register_to_config
from diffusers.utils import hub_utils
from huggingface_hub.errors import OfflineModeIsEnabled

from xfuser.model_executor.models.runner_models.base_model import (
    ModelCapabilities,
    ModelSettings,
)
from xfuser.model_executor.models.runner_models.loading.contracts import LoadSupport
from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader


class TinyShardedModel(diffusers.ModelMixin, diffusers.ConfigMixin):
    @register_to_config
    def __init__(self, width=4):
        super().__init__()
        self.first = torch.nn.Linear(width, width)
        self.second = torch.nn.Linear(width, width)


@pytest.fixture
def cached_sharded_checkpoint(tmp_path):
    repo_id = "offline-regression/tiny-sharded"
    commit = "a" * 40
    cache_dir = tmp_path / "hub"
    repo_cache = cache_dir / "models--offline-regression--tiny-sharded"
    component = repo_cache / "snapshots" / commit / "transformer"
    refs = repo_cache / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)

    reference = TinyShardedModel()
    with torch.no_grad():
        for index, parameter in enumerate(reference.parameters(), start=1):
            parameter.fill_(index)
    reference.save_pretrained(component, safe_serialization=True, max_shard_size=64)
    index = json.loads((component / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) > 1
    return repo_id, cache_dir, reference.state_dict()


@pytest.fixture
def offline_hub(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def offline_metadata(*args, **kwargs):
        raise OfflineModeIsEnabled("Hub metadata is unavailable in offline mode")

    # Only the remote metadata boundary is replaced. Config, index, snapshot,
    # safetensors deserialization and model construction all use the real APIs.
    monkeypatch.setattr(hub_utils, "model_info", offline_metadata)

    def unexpected_network(*args, **kwargs):
        pytest.fail("Offline checkpoint loading attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected_network)
    monkeypatch.setattr(socket, "create_connection", unexpected_network)


@pytest.mark.parametrize("explicit_online", [False, True], ids=["bare", "explicit-online"])
def test_cached_shards_load_with_offline_checkpoint_request(cached_sharded_checkpoint, offline_hub, explicit_online):
    repo_id, cache_dir, expected_weights = cached_sharded_checkpoint
    loader = ModelLoader(
        SimpleNamespace(
            settings=ModelSettings(model_name=repo_id),
            capabilities=ModelCapabilities(),
            load_support=LoadSupport(),
        )
    )

    # The old bare call still asks for remote shard metadata even with every
    # checkpoint file cached. Explicit caller choices also retain that behavior.
    if explicit_online:
        online_request = loader.checkpoint_request("transformer", cache_dir=cache_dir, local_files_only=False)
        online_kwargs = online_request.from_pretrained_kwargs()
    else:
        online_kwargs = {"subfolder": "transformer", "cache_dir": cache_dir}
    with pytest.raises(OfflineModeIsEnabled, match="Hub metadata"):
        TinyShardedModel.from_pretrained(repo_id, **online_kwargs)

    request = loader.checkpoint_request("transformer", cache_dir=cache_dir)
    loaded = TinyShardedModel.from_pretrained(request.model_name_or_path, **request.from_pretrained_kwargs())
    actual_weights = loaded.state_dict()
    assert actual_weights.keys() == expected_weights.keys()
    for name, expected in expected_weights.items():
        assert actual_weights[name].device.type == "cpu"
        torch.testing.assert_close(actual_weights[name], expected, rtol=0, atol=0)
    assert not loaded.training
