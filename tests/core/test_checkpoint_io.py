"""Unit tests for xfuser.core.utils.checkpoint_io.

Checkpoint file resolution has to work for both shapes a user can pass as a model name: a Hub repo
id and a path to a local checkpoint directory. The local shape has no coverage in the runners
(they need GPUs and real weights), so it is pinned here — these tests are CPU-only and need no
network, since a local directory never reaches the Hub.

Run with:
    pytest tests/core/test_checkpoint_io.py -v
"""

import json
import os

import pytest

from xfuser.core.utils.checkpoint_io import (
    component_shard_paths,
    resolve_checkpoint_weight_map,
    resolve_repo_file,
)

safetensors = pytest.importorskip("safetensors")


# ============================================================================
# Test Fixtures
# ============================================================================


def _write_safetensors(path, tensors):
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)


@pytest.fixture
def single_file_checkpoint(tmp_path):
    """Local checkpoint directory holding one unsharded transformer safetensors file."""
    import torch

    _write_safetensors(
        str(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"),
        {"blocks.0.weight": torch.zeros(4, 4), "blocks.1.weight": torch.zeros(4, 4)},
    )
    return tmp_path


@pytest.fixture
def sharded_checkpoint(tmp_path):
    """Local checkpoint directory holding a two-shard transformer plus its index."""
    import torch

    subfolder = tmp_path / "transformer"
    _write_safetensors(
        str(subfolder / "diffusion_pytorch_model-00001-of-00002.safetensors"),
        {"blocks.0.weight": torch.zeros(4, 4)},
    )
    _write_safetensors(
        str(subfolder / "diffusion_pytorch_model-00002-of-00002.safetensors"),
        {"blocks.1.weight": torch.zeros(4, 4)},
    )
    index = {
        "weight_map": {
            "blocks.0.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
            "blocks.1.weight": "diffusion_pytorch_model-00002-of-00002.safetensors",
        }
    }
    (subfolder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    return tmp_path


# ============================================================================
# resolve_repo_file
# ============================================================================


def test_resolve_repo_file_finds_local_file(single_file_checkpoint):
    resolved = resolve_repo_file(
        str(single_file_checkpoint), "transformer/diffusion_pytorch_model.safetensors"
    )
    assert resolved == str(
        single_file_checkpoint / "transformer" / "diffusion_pytorch_model.safetensors"
    )


def test_resolve_repo_file_returns_none_for_absent_local_file(single_file_checkpoint):
    """Absence must be None, not an exception: callers probe for the optional shard index."""
    assert (
        resolve_repo_file(str(single_file_checkpoint), "transformer/nope.json") is None
    )


def test_resolve_repo_file_does_not_treat_a_directory_as_a_file(single_file_checkpoint):
    assert resolve_repo_file(str(single_file_checkpoint), "transformer") is None


# ============================================================================
# resolve_checkpoint_weight_map
# ============================================================================


def test_weight_map_from_local_single_file(single_file_checkpoint):
    """Regression: this raised HFValidationError for local dirs, killing the self-fill path."""
    expected_path = str(
        single_file_checkpoint / "transformer" / "diffusion_pytorch_model.safetensors"
    )
    weight_map = resolve_checkpoint_weight_map(
        str(single_file_checkpoint), "transformer"
    )
    assert weight_map == {
        "blocks.0.weight": expected_path,
        "blocks.1.weight": expected_path,
    }


def test_weight_map_from_local_shards(sharded_checkpoint):
    weight_map = resolve_checkpoint_weight_map(str(sharded_checkpoint), "transformer")
    subfolder = sharded_checkpoint / "transformer"
    assert weight_map == {
        "blocks.0.weight": str(
            subfolder / "diffusion_pytorch_model-00001-of-00002.safetensors"
        ),
        "blocks.1.weight": str(
            subfolder / "diffusion_pytorch_model-00002-of-00002.safetensors"
        ),
    }


def test_weight_map_raises_when_component_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_weight_map(str(tmp_path), "transformer")


# ============================================================================
# component_shard_paths
# ============================================================================


def test_shard_paths_from_local_single_file(single_file_checkpoint):
    """Regression: this silently returned an empty set, no-oping the page-cache drop."""
    paths = component_shard_paths(
        str(single_file_checkpoint), "transformer", "diffusion_pytorch_model"
    )
    assert paths == {
        str(
            single_file_checkpoint
            / "transformer"
            / "diffusion_pytorch_model.safetensors"
        )
    }


def test_shard_paths_from_local_shards(sharded_checkpoint):
    paths = component_shard_paths(
        str(sharded_checkpoint), "transformer", "diffusion_pytorch_model"
    )
    subfolder = sharded_checkpoint / "transformer"
    assert paths == {
        str(subfolder / "diffusion_pytorch_model-00001-of-00002.safetensors"),
        str(subfolder / "diffusion_pytorch_model-00002-of-00002.safetensors"),
    }


def test_shard_paths_empty_for_component_without_safetensors(single_file_checkpoint):
    """A component with no safetensors of that basename is not an error; it just isn't dropped."""
    assert (
        component_shard_paths(str(single_file_checkpoint), "text_encoder", "model")
        == set()
    )


# ============================================================================
# Hub probing: absence vs unreachability
# ============================================================================


def test_absent_hub_file_probes_as_none(monkeypatch):
    """Callers probe for the shard index to decide the layout, so a repo that simply has no such
    file must read as None rather than raise."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def absent(*args, **kwargs):
        raise EntryNotFoundError("no such file in repo")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", absent)
    assert resolve_repo_file("org/repo", "transformer/index.json") is None


def test_unreachable_hub_file_raises_instead_of_probing_as_none(monkeypatch):
    """LocalEntryNotFoundError subclasses EntryNotFoundError but means "could not reach it", not
    "it does not exist". Swallowing it reports a sharded checkpoint as unsharded, and the run then
    fails against the single-file name, blaming a file that was never the problem."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(*args, **kwargs):
        raise LocalEntryNotFoundError("offline and not in cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(LocalEntryNotFoundError):
        resolve_repo_file("org/repo", "transformer/index.json")
