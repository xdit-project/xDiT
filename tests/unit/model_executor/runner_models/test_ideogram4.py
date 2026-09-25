from types import SimpleNamespace

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
