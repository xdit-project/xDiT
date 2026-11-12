"""Attention-backend selection: the checks it runs and the automatic choice."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xfuser.config.config import RuntimeConfig
from xfuser.core.attention import registry, requirements
from xfuser.core.attention.constraints import ANY_CALL
from xfuser.core.attention.requirements import ALWAYS
from xfuser.core.attention.spec import AttentionBackendType, Spec
from xfuser.core.distributed import runtime_state


def test_selection_checks_the_run_config_against_the_spec_after_the_kernel_resolves():
    events = []

    def validates(config):
        events.append(("validated", config.sparseattn_block_size))
        raise ValueError("no kernel at that tile")

    spec = Spec(
        AttentionBackendType.SDPA,
        impl=lambda q, k, v, c: (q, None),
        ring=ALWAYS,
        requires=ALWAYS,
        accepts=ANY_CALL,
        initializers=(lambda: events.append("resolved"),),
        validates=validates,
    )
    state = runtime_state.DiTRuntimeState.__new__(runtime_state.DiTRuntimeState)
    state.runtime_config = RuntimeConfig(sparseattn_block_size=(64, 64))
    state.parallel_config = SimpleNamespace(ring_degree=1)

    with registry.using([spec]), pytest.raises(ValueError, match="no kernel at that tile"):
        state._check_if_backend_compatible_with_current_configuration(AttentionBackendType.SDPA)

    assert events == ["resolved", ("validated", (64, 64))]


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
