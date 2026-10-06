import builtins
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.model_executor.layers import fused_a2a_integration


@pytest.fixture(autouse=True)
def _reset_attention_a2a_state():
    yield
    fused_a2a_integration._ATTENTION_A2A_POLICY = AttentionA2AConfig()
    fused_a2a_integration._ATTENTION_A2A_TIER2_ENABLED = False
    fused_a2a_integration._ATTENTION_A2A_TIER1_ENABLED = False
    fused_a2a_integration._activate_attention_a2a_config(AttentionA2AConfig())


def test_tier2_probe_is_not_version_gated():
    with (
        mock.patch.object(torch, "__version__", "9.9.0"),
        mock.patch.object(torch.version, "git_version", "unvalidated"),
    ):
        supported, reason = fused_a2a_integration._tier2_support_status()

    assert supported
    assert reason is None


def test_tier2_probe_rejects_incompatible_signature():
    from torch._inductor import ir

    with mock.patch.object(
        ir._WaitKernel,
        "create_wait",
        lambda kernel: None,
    ):
        supported, reason = fused_a2a_integration._tier2_support_status()

    assert not supported
    assert "_WaitKernel.create_wait" in reason
    assert "incompatible signature" in reason


def test_configure_uses_tier1_when_tier2_is_unavailable():
    config = AttentionA2AConfig(profile="e4m3-e4m3")
    with (
        mock.patch.object(
            fused_a2a_integration,
            "_tier2_support_status",
            return_value=(False, "Tier-2 test failure"),
        ),
        mock.patch.object(
            fused_a2a_integration,
            "_tier1_support_status",
            return_value=(True, None),
        ),
        mock.patch.object(fused_a2a_integration, "preflight_attention_a2a") as preflight,
        mock.patch.object(fused_a2a_integration, "_register_input_collective") as register,
    ):
        fused_a2a_integration.configure_attention_a2a(config, "aiter_fp8")

    preflight.assert_called_once_with(config)
    register.assert_not_called()
    assert fused_a2a_integration.get_fused_a2a_mode() == 1
    assert fused_a2a_integration.get_attention_a2a_execution_path() == "tier1"


def test_configure_fails_when_both_compiler_tiers_are_unavailable():
    config = AttentionA2AConfig(profile="e4m3-e4m3")
    with (
        mock.patch.object(
            fused_a2a_integration,
            "_tier2_support_status",
            return_value=(False, "Tier-2 test failure"),
        ),
        mock.patch.object(
            fused_a2a_integration,
            "_tier1_support_status",
            return_value=(False, "Tier-1 test failure"),
        ),
        mock.patch.object(fused_a2a_integration, "preflight_attention_a2a") as preflight,
        pytest.raises(RuntimeError, match="--attention_a2a none"),
    ):
        fused_a2a_integration.configure_attention_a2a(config, "aiter_fp8")

    preflight.assert_not_called()


def test_disabled_feature_does_not_preflight_optional_dependencies():
    with mock.patch.object(fused_a2a_integration, "preflight_attention_a2a") as preflight:
        fused_a2a_integration.configure_attention_a2a(AttentionA2AConfig())

    preflight.assert_not_called()
    assert fused_a2a_integration.get_attention_a2a_execution_path() == "disabled"


def test_missing_mori_fails_preflight_before_allocation():
    real_import = builtins.__import__

    def import_without_mori(name, *args, **kwargs):
        if name == "mori.shmem":
            raise ModuleNotFoundError("No module named 'mori'")
        return real_import(name, *args, **kwargs)

    with (
        mock.patch.object(builtins, "__import__", import_without_mori),
        pytest.raises(RuntimeError, match="No module named 'mori'"),
    ):
        fused_a2a_integration.preflight_attention_a2a(AttentionA2AConfig(profile="e4m3-e4m3"))


def test_collects_public_per_role_results_without_copies():
    results = tuple(
        SimpleNamespace(
            payload=torch.empty(index + 1),
            scale=torch.empty(index + 2),
        )
        for index in range(3)
    )

    payloads, scales = fused_a2a_integration._collect_packed_role_results(results)

    assert all(payload is result.payload for payload, result in zip(payloads, results, strict=True))
    assert all(scale is result.scale for scale, result in zip(scales, results, strict=True))


def test_runtime_does_not_access_private_aiter_buffers():
    source = inspect.getsource(fused_a2a_integration)

    assert "._epoch" not in source
    assert ".outputs_sets" not in source
    assert ".scales_sets" not in source
    assert "lru_cache" not in source


def test_heap_size_parser():
    assert fused_a2a_integration._heap_size_bytes("12G") == 12 << 30
    assert fused_a2a_integration._heap_size_bytes("512M") == 512 << 20
    with pytest.raises(ValueError, match="unsupported size suffix"):
        fused_a2a_integration._heap_size_bytes("12T")
