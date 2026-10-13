"""The automatic attention-backend choice on a host that cannot run it."""

from unittest.mock import Mock

import torch

from xfuser.core.attention import requirements
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed import runtime_state


def test_unrunnable_automatic_backend_falls_back_to_sdpa_with_a_warning(monkeypatch):
    # A CUDA build of torch reports cuDNN on a host with no visible GPU.
    monkeypatch.setattr(runtime_state.envs, "_is_hip", lambda: False)
    for package in ("has_flash_attn_4", "has_flash_attn_3", "has_flash_attn", "has_npu_flash_attn"):
        monkeypatch.setitem(runtime_state.env_info, package, False)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(requirements, "_platform", lambda: "cpu")
    logger = Mock()
    monkeypatch.setattr(runtime_state, "logger", logger)

    backend = runtime_state.RuntimeState._select_attention_backend(None)

    assert backend is AttentionBackendType.SDPA
    logger.warning.assert_called_once()
    message = logger.warning.call_args.args[0]
    assert "CUDNN" in message
    assert "requires cuda, found cpu" in message
