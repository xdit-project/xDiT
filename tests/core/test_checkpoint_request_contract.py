"""Dependency-light tests for checkpoint requests and discovery."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/checkpoint.py"
)


def _load_checkpoint_module():
    spec = importlib.util.spec_from_file_location(
        "roadmap3_checkpoint", CHECKPOINT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checkpoint():
    return _load_checkpoint_module()


def test_request_builds_from_pretrained_kwargs_without_dropping_false(checkpoint):
    request = checkpoint.CheckpointRequest(
        "org/repo",
        subfolder="transformer",
        revision="refs/pr/7",
        variant="fp16",
        token="secret",
        cache_dir="/cache",
        local_files_only=False,
    )

    assert request.from_pretrained_kwargs() == {
        "subfolder": "transformer",
        "revision": "refs/pr/7",
        "variant": "fp16",
        "token": "secret",
        "cache_dir": "/cache",
        "local_files_only": False,
    }


def test_local_request_resolves_subfolder_without_hub(checkpoint, tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    weight = component / "weights.safetensors"
    weight.write_bytes(b"header")

    request = checkpoint.CheckpointRequest(
        str(tmp_path),
        subfolder="transformer",
        revision="ignored-for-local-layout",
        local_files_only=True,
    )

    assert checkpoint.resolve_checkpoint_file(
        request, "weights.safetensors"
    ) == str(weight)
    assert checkpoint.resolve_checkpoint_file(request, "missing") is None


def test_hub_resolution_propagates_request_kwargs(checkpoint, monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")

    class EntryNotFoundError(Exception):
        pass

    class LocalEntryNotFoundError(EntryNotFoundError):
        pass

    def download(**kwargs):
        calls.append(kwargs)
        return "/cache/file"

    hub.hf_hub_download = download
    errors.EntryNotFoundError = EntryNotFoundError
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

    request = checkpoint.CheckpointRequest(
        "org/repo",
        subfolder="transformer",
        revision="main",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
    )

    assert checkpoint.resolve_checkpoint_file(request, "index.json") == "/cache/file"
    assert calls == [
        {
            "repo_id": "org/repo",
            "filename": "index.json",
            "subfolder": "transformer",
            "revision": "main",
            "token": "token",
            "cache_dir": "/cache",
            "local_files_only": True,
        }
    ]


def test_sharded_discovery_maps_keys_without_reading_tensors(checkpoint, tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    shard = component / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"not opened by discovery")
    (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blocks.0.weight": shard.name}})
    )

    manifest = checkpoint.discover_checkpoint(
        checkpoint.CheckpointRequest(str(tmp_path), subfolder="transformer")
    )

    assert manifest.weight_map == {"blocks.0.weight": str(shard)}
    assert manifest.shard_paths == frozenset({str(shard)})


def test_variant_uses_diffusers_checkpoint_filenames(checkpoint, tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    shard = component / "diffusion_pytorch_model.fp16.safetensors"
    shard.write_bytes(b"header only")

    manifest = checkpoint.discover_checkpoint(
        checkpoint.CheckpointRequest(
            str(tmp_path), subfolder="transformer", variant="fp16"
        ),
        key_reader=lambda path: ("weight",),
    )

    assert manifest.weight_map == {"weight": str(shard)}


def test_variant_index_places_variant_before_json_extension(checkpoint, tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    shard = component / "diffusion_pytorch_model.fp16-00001-of-00001.safetensors"
    shard.write_bytes(b"header only")
    (
        component
        / "diffusion_pytorch_model.safetensors.index.fp16.json"
    ).write_text(json.dumps({"weight_map": {"weight": shard.name}}))

    manifest = checkpoint.discover_checkpoint(
        checkpoint.CheckpointRequest(
            str(tmp_path), subfolder="transformer", variant="fp16"
        ),
        key_reader=lambda path: pytest.fail("variant index was not discovered"),
    )

    assert manifest.weight_map == {"weight": str(shard)}
