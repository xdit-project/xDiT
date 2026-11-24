"""LTX-2.5 auxiliary networks must accept device latents under CPU offload."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.ltx import (
    xFuserLTX25DistilledVideoModel,
    xFuserLTX25FullVideoModel,
)


@pytest.mark.parametrize("cls", [xFuserLTX25DistilledVideoModel, xFuserLTX25FullVideoModel])
@pytest.mark.parametrize("offload", [False, True])
def test_auxiliary_networks_accept_device_latents(monkeypatch, tmp_path, cls, offload):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import meta_load, placement

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    local_rank = torch.cuda.device_count() - 1
    group = SimpleNamespace(local_rank=local_rank, world_size=1)
    for module in (base_model, meta_load, placement):
        monkeypatch.setattr(module, "get_world_group", lambda: group)

    model = cls(
        xFuserArgs(model="LTX-2.5", task="t2v", output_directory=str(tmp_path), enable_model_cpu_offload=offload)
    )
    # Stand in for the third-party pipeline; its device follows its main module.
    transformer = torch.nn.Linear(2, 2)
    model.pipe = mock.Mock(device=transformer.weight.device, components={})
    model.pipe.to.side_effect = lambda device: SimpleNamespace(device=transformer.to(device).weight.device)
    model.upsample_pipe = torch.nn.Linear(2, 2)
    model.decode_pipe = SimpleNamespace(diffusion_decoder=torch.nn.Linear(2, 2))
    helpers = [model.decode_pipe.diffusion_decoder]
    if model._DISTILLED:
        helpers.append(model.upsample_pipe)
    latents = torch.ones(1, 2, device=f"cuda:{local_rank}")
    expected = [helper(latents.cpu()).detach().to(latents.device) for helper in helpers]

    model._post_load_and_state_initialization({})

    assert transformer.weight.device.type == ("cpu" if offload else "cuda")
    for helper, reference in zip(helpers, expected):
        torch.testing.assert_close(helper(latents), reference)
