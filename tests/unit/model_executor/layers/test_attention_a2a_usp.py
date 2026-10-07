from types import SimpleNamespace
from unittest import mock

import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import usp


def _runtime_state():
    return SimpleNamespace(
        attention_backend=AttentionBackendType.AITER_FP8,
        fp8_comms=None,
        runtime_config=SimpleNamespace(
            use_spargeattn_head_balance=False,
        ),
    )


def test_trims_plain_packed_attention_padding():
    key = torch.empty(1, 16, 2, 128, dtype=torch.uint8)
    value = torch.empty_like(key)
    scales = (
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
    )

    key, value, scales = usp._trim_packed_attention_a2a_padding(
        key,
        value,
        scales,
        13,
        ("e4m3", "e4m3", "e4m3"),
        "e4m3-e4m3",
    )

    assert key.shape[1] == value.shape[1] == 13
    assert all(scale.numel() == 1 for scale in scales)


def test_routes_packed_a2a_input_and_rccl_output():
    query = torch.randn(1, 4, 8, 128)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    packed_query = torch.empty(1, 16, 2, 128, dtype=torch.uint8)
    packed_key = torch.empty_like(packed_query)
    packed_value = torch.empty_like(packed_query)
    packed_scales = (
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
    )
    attention_output = torch.randn(1, 2, 16, 128)
    final_output = torch.randn_like(attention_output)

    with (
        mock.patch.object(usp, "get_sequence_parallel_world_size", return_value=2),
        mock.patch.object(usp, "get_ulysses_parallel_world_size", return_value=2),
        mock.patch.object(usp, "get_ulysses_parallel_rank", return_value=0),
        mock.patch.object(usp, "get_ring_parallel_world_size", return_value=1),
        mock.patch.object(
            usp.PROCESS_GROUP,
            "ULYSSES_PG",
            SimpleNamespace(group_name="test"),
        ),
        mock.patch.object(usp, "get_runtime_state", return_value=_runtime_state()),
        mock.patch.object(usp, "_ulysses_extra_inputs", return_value=[]),
        mock.patch.object(usp, "_has_kv_cache", return_value=False),
        mock.patch.object(usp, "get_fused_a2a_mode", return_value=1),
        mock.patch.object(usp, "use_fused_a2a_packed", return_value=True),
        mock.patch.object(usp, "get_fused_a2a_profile", return_value="e4m3-e4m3"),
        mock.patch.object(
            usp,
            "fused_a2a_input",
            return_value=(
                (packed_query, packed_key, packed_value),
                packed_scales,
            ),
        ) as input_a2a,
        mock.patch.object(
            usp,
            "_attention_a2a_packed_attn_call",
            return_value=attention_output,
        ) as packed_attention,
        mock.patch.object(
            usp,
            "_ft_c_output_all_to_all",
            return_value=final_output,
        ) as output_a2a,
        mock.patch.object(usp, "fp8_observe_output"),
    ):
        result = usp.USP(
            query,
            key,
            value,
            backend=AttentionBackendType.AITER_FP8,
            attn_layer=object(),
            attention_kwargs={"valid_kv_len": 13},
            attention_a2a_enabled=True,
        )

    assert result is final_output
    input_a2a.assert_called_once()
    packed_attention.assert_called_once()
    output_a2a.assert_called_once_with(attention_output)
    packed_args = packed_attention.call_args.args
    assert packed_args[0] is packed_query
    assert packed_args[1].shape[1] == 13
    assert packed_args[2].shape[1] == 13


def test_rejects_arbitrary_varlen_packing_before_transport():
    query = torch.randn(1, 4, 8, 128)
    runtime = _runtime_state()
    kwargs = {
        "indices_k": torch.arange(8),
        "cu_seqlens_k": torch.tensor([0, 8]),
        "max_seqlen_k": 8,
    }

    with (
        mock.patch.object(usp, "get_ulysses_parallel_world_size", return_value=2),
        mock.patch.object(usp, "get_ring_parallel_world_size", return_value=1),
        mock.patch.object(usp, "get_runtime_state", return_value=runtime),
        mock.patch.object(usp, "get_fused_a2a_mode", return_value=1),
        mock.patch.object(usp, "use_fused_a2a_packed", return_value=True),
        mock.patch.object(usp, "get_fused_a2a_profile", return_value="e4m3-e4m3"),
        mock.patch.object(
            usp.attention_registry,
            "find",
            return_value=SimpleNamespace(is_sparse=False),
        ),
        mock.patch.object(usp, "_get_attention_function"),
        mock.patch.object(usp, "fp8_attention_kwargs", return_value={}),
        mock.patch.object(
            usp,
            "apply_head_balance",
            return_value=(query, query, query, False, kwargs),
        ),
        mock.patch.object(usp, "_ulysses_extra_inputs", return_value=[]),
        mock.patch.object(usp, "_has_kv_cache", return_value=False),
        mock.patch.object(usp, "fused_a2a_input") as input_a2a,
        torch.no_grad(),
    ):
        try:
            usp.USP(
                query,
                query,
                query,
                backend=AttentionBackendType.AITER_FP8,
                attn_layer=object(),
                attention_kwargs=kwargs,
                attention_a2a_enabled=True,
            )
        except NotImplementedError as exc:
            assert "arbitrary varlen key packing" in str(exc)
        else:
            raise AssertionError("expected arbitrary varlen packing rejection")

    input_a2a.assert_not_called()
