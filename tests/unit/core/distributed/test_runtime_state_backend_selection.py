"""Attention-backend selection: the checks it runs and the automatic choice."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xfuser.config.config import RuntimeConfig
from xfuser.core.attention import registry, requirements
from xfuser.core.attention.constraints import ANY_CALL
from xfuser.core.attention.requirements import ALWAYS, NEVER
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


def test_a_backend_without_ring_support_is_refused_under_ring():
    spec = Spec(
        AttentionBackendType.SDPA,
        impl=lambda q, k, v, c: (q, None),
        ring=NEVER,
        requires=ALWAYS,
        accepts=ANY_CALL,
    )
    state = runtime_state.DiTRuntimeState.__new__(runtime_state.DiTRuntimeState)
    state.runtime_config = RuntimeConfig()
    state.parallel_config = SimpleNamespace(ring_degree=2)

    with registry.using([spec]), pytest.raises(RuntimeError, match="cannot be used with ring parallelism"):
        state._check_if_backend_compatible_with_current_configuration(AttentionBackendType.SDPA)


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


def _configuring_spec(backend, hook):
    return Spec(
        backend,
        impl=lambda q, k, v, c: (q, None),
        ring=ALWAYS,
        requires=ALWAYS,
        accepts=ANY_CALL,
        configures=hook,
    )


def test_every_selected_backend_gets_the_step_plan_and_a_shared_hook_runs_once():
    """The attention, cross-attention and scheduled backends are all configured: a backend picked
    only by a hybrid schedule still owns per-step state. Backends from one family share a hook,
    which must not install the same state twice."""
    calls = []

    def shared(config, steps, forwards):
        calls.append(("shared", config.solattn_beta_schedule, steps, forwards))

    def own(config, steps, forwards):
        calls.append(("own", steps, forwards))

    specs = [
        _configuring_spec(AttentionBackendType.AITER_FP8_SOL, shared),
        _configuring_spec(AttentionBackendType.AITER_BF16_SOL, shared),
        _configuring_spec(AttentionBackendType.AITER, own),
        _configuring_spec(AttentionBackendType.SDPA, None),
    ]
    state = runtime_state.DiTRuntimeState.__new__(runtime_state.DiTRuntimeState)
    state.runtime_config = RuntimeConfig(solattn_beta_schedule="1.0:0.0")
    state.attention_backend = AttentionBackendType.SDPA
    state.cross_attention_backend = AttentionBackendType.AITER
    state.attention_schedule = SimpleNamespace(
        backends=[AttentionBackendType.AITER_FP8_SOL, AttentionBackendType.AITER_BF16_SOL]
    )

    with registry.using(specs):
        state.configure_attention_backends(40, 2)

    assert sorted(calls) == [("own", 40, 2), ("shared", "1.0:0.0", 40, 2)]
