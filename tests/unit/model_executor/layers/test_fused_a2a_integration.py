import builtins
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.model_executor.layers import fused_a2a_integration


@pytest.fixture(autouse=True)
def _reset_attention_a2a_state():
    fused_a2a_integration._INPUT_PENDING.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAITS.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAIT_COUNTS.clear()
    yield
    fused_a2a_integration._INPUT_PENDING.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAITS.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAIT_COUNTS.clear()
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


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("e4m3-e4m3", None),
        ("e4m3-mxfp6", None),
        ("mxfp6-e4m3", None),
        ("mxfp6-mxfp6", 95),
        ("mxfp6-mxfp4", 95),
        ("mxfp4-mxfp4", 95),
    ),
)
def test_selects_direct_valid_kv_only_for_token_local_profiles(
    profile,
    expected,
):
    config = AttentionA2AConfig(profile=profile)

    assert (
        fused_a2a_integration._select_direct_valid_kv_len(
            config,
            95,
            64,
            2,
        )
        == expected
    )


def test_direct_valid_kv_falls_back_when_padding_spans_multiple_ranks():
    config = AttentionA2AConfig(profile="mxfp6-mxfp4")

    assert (
        fused_a2a_integration._select_direct_valid_kv_len(
            config,
            400,
            64,
            8,
        )
        is None
    )
    assert (
        fused_a2a_integration._select_direct_valid_kv_len(
            config,
            449,
            64,
            8,
        )
        == 449
    )


def test_tier2_collapses_waits_only_for_mx_qk_profiles():
    pending = tuple(torch.empty(index + 1) for index in range(6))
    wait = mock.Mock(side_effect=lambda tensor: tensor)

    outputs = fused_a2a_integration._wait_tier2_outputs(pending, "mxfp6", wait)

    assert all(output is tensor for output, tensor in zip(outputs, pending, strict=True))
    wait.assert_called_once_with(pending[-1])

    wait.reset_mock()
    outputs = fused_a2a_integration._wait_tier2_outputs(pending, "e4m3", wait)

    assert all(output is tensor for output, tensor in zip(outputs, pending, strict=True))
    assert [call.args[0] for call in wait.call_args_list] == list(pending)


def _install_pending_waits(collapse_waits):
    results = tuple(SimpleNamespace(payload=torch.empty(index + 1), scale=torch.empty(index + 2)) for index in range(3))
    handle = ("test", 0, 0)
    event = object()
    fused_a2a_integration._INPUT_PENDING[handle] = {"results": results}
    tensors = [tensor for result in results for tensor in (result.payload, result.scale)]
    for tensor in tensors:
        fused_a2a_integration._INPUT_COLLECTIVE_WAITS[tensor.data_ptr()] = (
            event,
            handle,
            collapse_waits,
        )
    fused_a2a_integration._INPUT_COLLECTIVE_WAIT_COUNTS[handle] = len(tensors)
    return handle, event, tensors


def test_collapsed_tier2_wait_releases_the_complete_trio():
    handle, event, tensors = _install_pending_waits(collapse_waits=True)
    stream = mock.Mock()

    with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
        result = fused_a2a_integration._consume_input_collective_wait(tensors[-1])

    assert result is tensors[-1]
    stream.wait_event.assert_called_once_with(event)
    assert handle not in fused_a2a_integration._INPUT_PENDING
    assert handle not in fused_a2a_integration._INPUT_COLLECTIVE_WAIT_COUNTS
    assert fused_a2a_integration._INPUT_COLLECTIVE_WAITS == {}


def test_per_tensor_tier2_wait_releases_whichever_wait_runs_last():
    handle, event, tensors = _install_pending_waits(collapse_waits=False)
    stream = mock.Mock()

    with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
        fused_a2a_integration._consume_input_collective_wait(tensors[-1])
        assert handle in fused_a2a_integration._INPUT_PENDING
        for tensor in tensors[:-1]:
            fused_a2a_integration._consume_input_collective_wait(tensor)

    assert stream.wait_event.call_count == len(tensors)
    assert all(call.args == (event,) for call in stream.wait_event.call_args_list)
    assert handle not in fused_a2a_integration._INPUT_PENDING
    assert handle not in fused_a2a_integration._INPUT_COLLECTIVE_WAIT_COUNTS
    assert fused_a2a_integration._INPUT_COLLECTIVE_WAITS == {}
