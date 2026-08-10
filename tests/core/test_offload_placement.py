"""Which device each offload mode onloads to.

Diffusers defaults its offload hooks to cuda:0. A multi-rank run that accepts that
default puts every rank on one device, and nothing notices until the first
collective fails with a duplicate GPU, so the device each rank names is worth
pinning rather than leaving to a library default.
"""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch):
    monkeypatch.setattr(base_model, "log", lambda *args, **kwargs: None)


def _model(*, local_rank, monkeypatch, **offload_flags):
    monkeypatch.setattr(
        base_model,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=local_rank),
    )
    calls = {}
    pipe = SimpleNamespace(
        enable_sequential_cpu_offload=lambda **kwargs: calls.update(sequential=kwargs),
        enable_model_cpu_offload=lambda **kwargs: calls.update(model=kwargs),
        components={},
    )
    flags = {
        "use_spargeattn_head_balance": False,
        "enable_slicing": False,
        "enable_tiling": False,
        "enable_group_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "enable_model_cpu_offload": False,
        "group_offload_low_cpu_mem": False,
        **offload_flags,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(**flags),
        pipe=pipe,
        _get_compiled_pipe_components=lambda: [],
    )
    model._local_onload_device = lambda: xFuserModel._local_onload_device(model)
    return model, calls


@pytest.mark.parametrize("local_rank", [0, 3])
@pytest.mark.parametrize(
    ("flag", "recorded"),
    [
        ("enable_sequential_cpu_offload", "sequential"),
        ("enable_model_cpu_offload", "model"),
    ],
)
def test_whole_pipeline_offload_onloads_to_this_ranks_device(
    monkeypatch, local_rank, flag, recorded
):
    """Every rank taking cuda:0 collides at the first all-to-all rather than at startup."""
    model, calls = _model(local_rank=local_rank, monkeypatch=monkeypatch, **{flag: True})

    xFuserModel._enable_options(model)

    assert str(calls[recorded]["device"]) == f"cuda:{local_rank}"


@pytest.mark.parametrize("local_rank", [0, 3])
def test_group_offload_onloads_to_this_ranks_device(monkeypatch, local_rank):
    """Group offloading named the local device already; this holds it to that."""
    model, _ = _model(
        local_rank=local_rank,
        monkeypatch=monkeypatch,
        enable_group_cpu_offload=True,
    )
    offloaded = {}

    class Component(torch.nn.Module):
        """The loop filters on Module, so the recorder has to be one."""

        def enable_group_offload(self, **kwargs):
            offloaded.update(kwargs)

    model.pipe.components = {"transformer": Component()}

    xFuserModel._enable_options(model)

    assert str(offloaded["onload_device"]) == f"cuda:{local_rank}"
