from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.pipelines.pipeline_ideogram4 import (
    _broadcast_object_in_group,
)


def test_ideogram4_broadcast_object_uses_cpu_subgroup(monkeypatch):
    cpu_group = object()
    coordinator = SimpleNamespace(first_rank=3, cpu_group=cpu_group)

    def fake_broadcast(payload, src, group):
        assert src == 3
        assert group is cpu_group
        payload[0] = ["broadcast caption"]

    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        fake_broadcast,
    )

    assert _broadcast_object_in_group(None, coordinator) == ["broadcast caption"]


@pytest.fixture
def build_ideogram4(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import ideogram4  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name):
        return MODEL_REGISTRY[name](xFuserArgs(model=name))

    return _build


@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("Ideogram-4", "ideogram-ai/ideogram-4-fp8"),
        ("ideogram-ai/ideogram-v4", "ideogram-ai/ideogram-4-fp8"),
        ("ideogram-ai/ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8"),
        (
            "CalamitousFelicitousness/Ideogram-4-bf16-Diffusers",
            "CalamitousFelicitousness/Ideogram-4-bf16-Diffusers",
        ),
    ],
)
def test_ideogram4_name_selects_its_checkpoint(build_ideogram4, name, checkpoint):
    model = build_ideogram4(name)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint


@pytest.mark.parametrize("name", ["ideogram-ai/ideogram-4-nf4", "ideogram-ai/ideogram-4-nf4-diffusers"])
def test_ideogram4_refuses_nf4_checkpoints_instead_of_loading_fp8(build_ideogram4, name):
    with pytest.raises(ValueError, match="NF4 checkpoint"):
        build_ideogram4(name)
