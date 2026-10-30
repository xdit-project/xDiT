import builtins
import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.core.distributed import attention_a2a as fused_a2a_integration
from xfuser.model_executor.layers.quantized_attention_output import e4m3_external_scale_from_amax


@pytest.fixture(autouse=True)
def _reset_attention_a2a_state():
    fused_a2a_integration._INPUT_PENDING.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAITS.clear()
    yield
    fused_a2a_integration._INPUT_PENDING.clear()
    fused_a2a_integration._INPUT_COLLECTIVE_WAITS.clear()
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


def test_tier2_waits_on_every_output():
    pending = tuple(torch.empty(index + 1) for index in range(6))
    wait = mock.Mock(side_effect=lambda tensor: tensor)

    outputs = fused_a2a_integration._wait_tier2_outputs(pending, wait)

    assert all(output is tensor for output, tensor in zip(outputs, pending, strict=True))
    assert [call.args[0] for call in wait.call_args_list] == list(pending)


def _install_pending_waits():
    results = tuple(SimpleNamespace(payload=torch.empty(index + 1), scale=torch.empty(index + 2)) for index in range(3))
    handle = ("test", 0, 0)
    events = [object() for _ in results]
    trio = {"events": list(events), "waited": -1, "stream": None}
    fused_a2a_integration._INPUT_PENDING[handle] = {"results": results, "trio_waits": trio}
    tensors = []
    for role, result in enumerate(results):
        for tensor in (result.payload, result.scale):
            fused_a2a_integration._INPUT_COLLECTIVE_WAITS[tensor.data_ptr()] = (trio, handle, role)
            tensors.append(tensor)
    return handle, trio, events, tensors


def test_tier2_waits_once_per_trio_on_the_latest_event():
    handle, _, events, tensors = _install_pending_waits()
    stream = mock.Mock()

    with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
        for index in (2, 4, 3, 5, 0, 1):
            assert fused_a2a_integration._consume_input_collective_wait(tensors[index]) is tensors[index]

    stream.wait_event.assert_called_once_with(events[-1])
    assert handle not in fused_a2a_integration._INPUT_PENDING
    assert fused_a2a_integration._INPUT_COLLECTIVE_WAITS == {}


def test_tier2_wait_before_later_submits_waits_again_for_them():
    _, trio, events, tensors = _install_pending_waits()
    del trio["events"][1:]
    stream = mock.Mock()

    with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
        fused_a2a_integration._consume_input_collective_wait(tensors[0])
        trio["events"].extend(events[1:])
        for tensor in tensors[1:]:
            fused_a2a_integration._consume_input_collective_wait(tensor)

    assert [call.args[0] for call in stream.wait_event.call_args_list] == [events[0], events[-1]]


_OUTPUT_MODULE = "aiter.ops.flydsl.attention_output_a2a_intranode"


class _FakeOutputOp:
    instances = []

    def __init__(self, *, rank, world_size, shape, group, codec):
        self.rank, self.world_size, self.shape, self.codec = rank, world_size, shape, codec
        self.destination_scales = []
        self.payload = torch.arange(6, dtype=torch.uint8)
        self.scale = torch.ones(2, dtype=torch.uint8)
        _FakeOutputOp.instances.append(self)

    def __call__(self, output, destination_scales=None):
        self.destination_scales.append(destination_scales)
        scale = self.scale if destination_scales is None else destination_scales[self.rank : self.rank + 1].view(1, 1)
        return SimpleNamespace(payload=self.payload, scale=scale)


@pytest.fixture
def fake_output_transport(monkeypatch):
    _FakeOutputOp.instances.clear()
    group = SimpleNamespace(group_name="ulysses")
    monkeypatch.setitem(sys.modules, _OUTPUT_MODULE, SimpleNamespace(AttentionOutputA2AIntraNodeOp=_FakeOutputOp))
    monkeypatch.setattr(fused_a2a_integration, "_group_ranks", lambda group: (0, 1))
    monkeypatch.setattr(fused_a2a_integration, "_init_mori", lambda group, ranks: None)
    monkeypatch.setattr(fused_a2a_integration.dist.distributed_c10d, "_resolve_process_group", lambda name: group)
    monkeypatch.setattr(fused_a2a_integration.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(fused_a2a_integration.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    yield group
    fused_a2a_integration._OUTPUT_OP_CACHE.clear()


def test_output_hop_caches_one_op_and_returns_its_buffers(fake_output_transport):
    output = torch.randn(1, 8, 2, 128, dtype=torch.bfloat16)

    payload, scale = fused_a2a_integration.fused_a2a_output(output, fake_output_transport, 1, "mxfp4")
    fused_a2a_integration.fused_a2a_output(output, fake_output_transport, 1, "mxfp4")

    (op,) = _FakeOutputOp.instances
    assert (op.rank, op.world_size, op.shape, op.codec) == (1, 2, (1, 8, 2, 128), "mxfp4")
    assert op.destination_scales == [None, None]
    assert payload.data_ptr() == op.payload.data_ptr()
    assert scale.data_ptr() == op.scale.data_ptr()


def test_e4m3_external_scale_quantizes_each_destination_with_its_own_amax(fake_output_transport):
    output = torch.zeros(1, 8, 2, 128, dtype=torch.bfloat16)
    output[:, :4] = 0.5
    output[:, 4:, 1, 7] = -3.140625

    _, scale = fused_a2a_integration.fused_a2a_output(output, fake_output_transport, 1, "e4m3_external_scale")

    (op,) = _FakeOutputOp.instances
    expected = e4m3_external_scale_from_amax(torch.tensor([0.5, 3.140625], dtype=torch.bfloat16))
    assert torch.equal(op.destination_scales[0], expected)
    assert torch.equal(scale, expected[1:].view(1, 1))
    assert scale.storage_offset() == 0


def test_output_hop_rejects_codecs_without_an_xdit_consumer(fake_output_transport):
    output = torch.zeros(1, 8, 2, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="unsupported Attention A2A output codec 'e4m3'"):
        fused_a2a_integration.fused_a2a_output(output, fake_output_transport, 0, "e4m3")
    assert _FakeOutputOp.instances == []


@pytest.mark.parametrize(
    ("codec", "payload_shape", "scale_shape", "scale_dtype"),
    [
        ("e4m3_external_scale", (1, 1184, 40, 128), (1, 1), torch.float32),
        ("mxfp4", (1184, 2560), (1280, 160), torch.uint8),
    ],
)
def test_output_fake_matches_the_projection_layouts(codec, payload_shape, scale_shape, scale_dtype):
    output = torch.empty(1, 9472, 5, 128, dtype=torch.bfloat16, device="meta")

    payload, scale = fused_a2a_integration._fused_a2a_output_fake(output, codec, 0, 8, "ulysses")

    assert (tuple(payload.shape), payload.dtype) == (payload_shape, torch.uint8)
    assert (tuple(scale.shape), scale.dtype) == (scale_shape, scale_dtype)


def test_mxfp6_output_fake_uses_the_a6w6_pack_sizes():
    gemm_op_a6w6 = pytest.importorskip("aiter.ops.gemm_op_a6w6")
    output = torch.empty(1, 9472, 5, 128, dtype=torch.bfloat16, device="meta")

    payload, scale = fused_a2a_integration._fused_a2a_output_fake(output, "mxfp6", 0, 8, "ulysses")

    assert (payload.numel(), scale.numel()) == tuple(gemm_op_a6w6.mxfp6_gemm_pack_size(1184, 5120))


def test_output_preflight_requires_every_consumer_codec(monkeypatch):
    module = SimpleNamespace(
        ATTENTION_OUTPUT_A2A_CODECS=("e4m3", "mxfp4"),
        PackedAttentionOutputResult=namedtuple("PackedAttentionOutputResult", ("payload", "scale")),
    )
    monkeypatch.setitem(sys.modules, _OUTPUT_MODULE, module)

    with pytest.raises(RuntimeError, match=r"lacks attention-output A2A codec\(s\) e4m3_external_scale, mxfp6"):
        fused_a2a_integration._check_attention_output_a2a_contract()

    module.ATTENTION_OUTPUT_A2A_CODECS = ("e4m3", "e4m3_external_scale", "mxfp4", "mxfp6")
    fused_a2a_integration._check_attention_output_a2a_contract()


def test_output_hop_follows_the_active_recipe():
    with (
        mock.patch.object(fused_a2a_integration, "_tier2_support_status", return_value=(False, "unavailable")),
        mock.patch.object(fused_a2a_integration, "_tier1_support_status", return_value=(True, None)),
        mock.patch.object(fused_a2a_integration, "preflight_attention_a2a"),
    ):
        fused_a2a_integration.configure_attention_a2a(AttentionA2AConfig(profile="auto", output="mori"), "aiter_mxfp6")

    assert fused_a2a_integration.get_fused_a2a_profile() == "mxfp6-e4m3"
    assert fused_a2a_integration.use_fused_a2a_output()

    fused_a2a_integration._activate_attention_a2a_config(AttentionA2AConfig())
    assert not fused_a2a_integration.use_fused_a2a_output()


def test_output_ops_freeze_the_policy_until_shutdown():
    fused_a2a_integration._OUTPUT_OP_CACHE["op"] = object()

    with pytest.raises(RuntimeError, match="cannot change after its runtime state is created"):
        fused_a2a_integration.configure_attention_a2a(AttentionA2AConfig(profile="e4m3-e4m3", output="mori"))
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        fused_a2a_integration.shutdown_attention_a2a()

    assert fused_a2a_integration._OUTPUT_OP_CACHE == {}
