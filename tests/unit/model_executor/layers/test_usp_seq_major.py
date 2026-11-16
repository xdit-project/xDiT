from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention.constraints import ANY_CALL
from xfuser.core.attention.requirements import ALWAYS
from xfuser.core.attention.spec import AttentionBackendType, Spec
from xfuser.model_executor.layers import usp, usp_seq_major


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_the_qkv_exchange_gives_each_rank_its_heads_over_the_whole_sequence(world_size, batch):
    heads, shard_len, head_dim = 2 * world_size, 3, 4
    # Rank p holds tokens [p * shard_len, (p + 1) * shard_len) of every head.
    full = [torch.randn(batch, world_size * shard_len, heads, head_dim) for _ in range(3)]
    sends = [
        usp_seq_major._qkv_send(*(t[:, p * shard_len : (p + 1) * shard_len] for t in full), world_size)
        for p in range(world_size)
    ]

    for rank in range(world_size):
        received = torch.stack([send[rank] for send in sends])
        views = usp_seq_major._qkv_views(received)
        local_heads = slice(rank * heads // world_size, (rank + 1) * heads // world_size)
        for view, tensor in zip(views, full):
            assert view.untyped_storage().data_ptr() == received.untyped_storage().data_ptr()
            torch.testing.assert_close(view, tensor[:, :, local_heads], rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_the_output_exchange_gives_each_rank_its_tokens_with_every_head(world_size, batch):
    heads, shard_len, head_dim = 2 * world_size, 3, 4
    full = torch.randn(batch, world_size * shard_len, heads, head_dim)
    local_heads = heads // world_size
    # Rank p computed heads [p * local_heads, (p + 1) * local_heads) for the whole sequence, in
    # attention's [P*s, b, h/P, d] output send buffer.
    sends = [
        full[:, :, p * local_heads : (p + 1) * local_heads]
        .transpose(0, 1)
        .reshape(world_size, shard_len, batch, local_heads, head_dim)
        for p in range(world_size)
    ]

    for rank in range(world_size):
        received = torch.stack([send[rank] for send in sends])
        rows = usp_seq_major.attention_output(received)
        expected = full[:, rank * shard_len : (rank + 1) * shard_len].flatten(2)
        torch.testing.assert_close(rows, expected, rtol=0, atol=0)


def _plan(monkeypatch, change):
    """usp_seq_major.plan for an 8-head layer under ``change`` to a plain Ulysses 4 AITER run."""
    settings = {"ulysses": 4, "ring": 1, "heads": 8, "backend": "AITER", "fp8_comms": None} | change
    state = SimpleNamespace(
        attention_backend=AttentionBackendType[settings["backend"]],
        fp8_comms=settings["fp8_comms"],
        runtime_config=SimpleNamespace(use_sparseattn_head_balance=False),
    )
    monkeypatch.setattr(usp_seq_major, "get_runtime_state", lambda: state)
    monkeypatch.setattr(usp_seq_major, "_parallel_degrees", lambda: (settings["ulysses"], settings["ring"]))
    monkeypatch.setattr(
        usp_seq_major, "_compile_capture_bypasses_attention", lambda: settings.get("capture_bypass", False)
    )
    monkeypatch.setattr(usp_seq_major.usp_sdma, "selected", lambda: settings.get("sdma", False))
    monkeypatch.setattr(usp_seq_major, "_has_kv_cache", lambda layer: settings.get("kv_cache", False))
    return usp_seq_major.plan(
        settings.get("attn", SimpleNamespace()), settings["heads"], settings.get("attention_kwargs")
    )


@pytest.mark.parametrize(
    "change",
    [
        {"ulysses": 1},
        {"ring": 2},
        {"heads": 6},  # not divisible by the Ulysses degree
        {"backend": "SDPA"},  # its kernel copies strided views
        {"capture_bypass": True},
        {"sdma": True},  # --ulysses_a2a sdma or packed: USP's exchange
        {"kv_cache": True},
        # The exchange carries no extra Ulysses inputs and builds no VarlenPacking.
        {"attention_kwargs": {usp.ULYSSES_EXTRA_INPUTS_KEY: ("gate",), "gate": torch.ones(1)}},
        {"attention_kwargs": {"attn_mask": torch.ones(1, dtype=torch.bool)}},
        {"attention_kwargs": {"valid_kv_len": 5, "cu_seqlens_k": torch.zeros(2, dtype=torch.int32)}},
    ],
)
def test_calls_the_exchange_does_not_serve_take_usp(monkeypatch, change):
    assert _plan(monkeypatch, change) is None


@pytest.mark.parametrize("backend", ["AITER", "AITER_FP8"])
def test_ulysses_calls_on_a_strided_bshd_backend_take_the_exchange(monkeypatch, backend):
    plan = _plan(monkeypatch, {"backend": backend})

    assert plan.spec.type is AttentionBackendType[backend]
    assert plan.world_size == 4 and plan.fp8_comms is None and plan.attention_kwargs == {}


def test_the_plan_keeps_valid_kv_len_and_ignores_unset_attention_kwargs(monkeypatch):
    plan = _plan(monkeypatch, {"attention_kwargs": {"valid_kv_len": 5, "attn_mask": None}})

    assert plan.attention_kwargs == {"valid_kv_len": 5}


@pytest.mark.parametrize(
    ("change", "fp8"),
    [
        ({}, True),
        ({"synced": False}, None),  # calibration observes through USP
        ({"backend": "AITER"}, False),  # fp8 comms does not apply to bf16 attention
        ({"owner": False}, False),  # not an fp8-comms layer
    ],
)
def test_calibrated_fp8_comms_layers_exchange_fp8_on_a_prequantizing_backend(monkeypatch, change, fp8):
    owner = object()
    model_state = SimpleNamespace(synced=change.get("synced", True))
    fp8_comms = SimpleNamespace(get_model_state=lambda model: model_state if model is owner else None)
    attn = SimpleNamespace(fp8_comms_layer_idx=torch.zeros(1, dtype=torch.long))
    for name in ("fp8_q_scale", "fp8_k_scale", "fp8_v_scale", "fp8_o_scale"):
        setattr(attn, name, torch.ones(1))
    if change.get("owner", True):
        attn.fp8_comms_owner = owner

    plan = _plan(monkeypatch, {"backend": change.get("backend", "AITER_FP8"), "fp8_comms": fp8_comms, "attn": attn})

    if fp8 is None:
        assert plan is None
        return
    assert (plan.fp8_comms is not None) is fp8
    if fp8:
        assert plan.fp8_comms.q_scale is attn.fp8_q_scale and plan.fp8_comms.o_scale is attn.fp8_o_scale


def test_attention_runs_the_plans_spec_on_the_received_views(monkeypatch):
    calls = []

    def kernel(query, key, value, call):
        calls.append((query, call))
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return call.seq_major_out.copy_(out.transpose(1, 2)).transpose(1, 2), None

    spec = Spec(
        AttentionBackendType.SDPA, impl=kernel, ring=ALWAYS, requires=ALWAYS, accepts=ANY_CALL, reads_strided_bshd=True
    )
    monkeypatch.setattr(usp_seq_major, "_sdpa_all_to_all_single", lambda tensor: tensor)
    query, key, value = (torch.randn(2, 6, 4, 8) for _ in range(3))
    plan = usp_seq_major.SeqMajorPlan(spec, 1, None, {"valid_kv_len": 5})
    send = usp_seq_major.send(query, key, value, plan)

    output = usp_seq_major.attention(send, plan)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key[:, :5].transpose(1, 2), value[:, :5].transpose(1, 2)
    )
    assert isinstance(output, usp_seq_major.SeqMajorOutput) and output.descale is None
    torch.testing.assert_close(output.rows(query.dtype), expected.transpose(1, 2).flatten(2))
    received_query, call = calls[0]
    assert received_query.untyped_storage().data_ptr() == send.untyped_storage().data_ptr()
    assert call.attention_kwargs["valid_kv_len"] == 5


def test_output_rows_dequantize_fp8_with_the_descale():
    rows = torch.randn(2, 3, 2 * 4 * 8)
    received = rows.view(2, 3, 2, 4, 8).permute(2, 1, 0, 3, 4)  # [P, s, b, h/P, d]
    descale = torch.tensor([0.5])
    fp8 = (received / descale).to(torch.float8_e4m3fn)

    dequantized = usp_seq_major.SeqMajorOutput(fp8, descale).rows(torch.bfloat16)

    assert dequantized.dtype == torch.bfloat16
    expected = usp_seq_major.attention_output(fp8).float() * descale
    torch.testing.assert_close(dequantized, expected.to(torch.bfloat16), rtol=0, atol=0)
