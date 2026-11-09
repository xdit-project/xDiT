"""Which USP calls the SDMA Ulysses all-to-all takes, and how it lays out an exchange."""

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.sdma.ulysses import (
    MAX_HEAD_GROUPS,
    Exchange,
    head_split,
    pack_group,
    pieces_from_grouped,
    pieces_from_heads,
    split_heads,
)
from xfuser.model_executor.layers import usp_sdma


@pytest.mark.parametrize(
    ("heads", "groups", "sizes"),
    [
        (5, 1, [5]),
        (5, 2, [3, 2]),
        (5, 5, [1, 1, 1, 1, 1]),
        # More groups than heads: one head per group.
        (3, 8, [1, 1, 1]),
        (7, 3, [3, 2, 2]),
    ],
)
def test_split_heads(heads, groups, sizes):
    assert split_heads(heads, groups) == sizes


def test_split_heads_caps_the_group_count():
    assert len(split_heads(4 * MAX_HEAD_GROUPS, 4 * MAX_HEAD_GROUPS)) == MAX_HEAD_GROUPS


def test_head_groups_keep_grouped_query_heads_whole():
    # 8 query heads over 2 K/V heads: a group holds whole K/V heads with their 4 query heads.
    assert head_split((8, 2, 2, 8), 2) == ((4, 1, 1, 4), (4, 1, 1, 4))
    assert head_split((8, 2, 2), 5) == ((4, 1, 1), (4, 1, 1))


def test_exchange_round_trips_through_integers():
    ex = Exchange.make(8, 2, 64, torch.bfloat16, [72, 64, 64], [6, 6, 6], [(0, 0), (8, 0), (8, 0)], 3)
    assert Exchange.decode(ex.encode(), torch.bfloat16) is ex


def test_inbox_keeps_reserved_rows_beside_every_rank_block():
    ex = Exchange.make(4, 1, 2, torch.bfloat16, [5, 3, 3], [2, 2, 2], [(0, 0), (1, 0), (0, 2)], 2)
    assert [ex.inbox_rows(t) for t in range(3)] == [20, 13, 14]
    # Group 0's Q region, then its K region, ...: each [b, rows, h, d].
    assert ex.inbox_offsets[0][1] == 20 * 1 * 2
    assert ex.inbox_offsets[1][0] == (20 + 13 + 14) * 1 * 2


def test_packed_group_holds_each_peer_s_pieces_in_turn():
    world, b, d = 2, 1, 2
    q = torch.arange(b * 4 * 3 * d, dtype=torch.float32).view(b, 3, 4, d).transpose(1, 2)  # [b, H=4, s=3, d]
    k = 100 + torch.arange(b * 2 * 3 * d, dtype=torch.float32).view(b, 3, 2, d).transpose(1, 2)  # [b, H=2, s=3, d]
    packed = pack_group([pieces_from_heads(q, world, 0, 2), pieces_from_heads(k, world, 0, 1)]).view(world, -1)
    for p in range(world):
        q_part = packed[p, : 3 * 2 * d].view(b, 3, 2, d)
        k_part = packed[p, 3 * 2 * d :].view(b, 3, 1, d)
        assert torch.equal(q_part, q[:, 2 * p : 2 * p + 2].transpose(1, 2))
        assert torch.equal(k_part, k[:, p : p + 1].transpose(1, 2))


def test_a_group_projected_alone_packs_like_the_same_heads_cut_from_all():
    world, b, s, d = 2, 1, 3, 2
    full = torch.randn(b, s, world * 3, d)  # [b, s, H, d], heads ordered by rank, 3 per rank
    group = full.view(b, s, world, 3, d)[:, :, :, 1:3].reshape(b, s, world * 2, d)  # heads 1..2 of every rank
    assert torch.equal(
        pack_group([pieces_from_grouped(group, world)]),
        pack_group([pieces_from_heads(full.transpose(1, 2), world, 1, 2)]),
    )


def _plain_call(**overrides):
    q = torch.zeros(1, 8, 4, 16)
    call = dict(
        query=q,
        key=q,
        value=q,
        backend=AttentionBackendType.SDPA,
        ring_world_size=1,
        joint_strategy=None,
        fp8_comms=None,
        extra_inputs=[],
        has_kv_cache=False,
        head_balanced=False,
        # As Wan's runner publishes them: sparse-backend options a dense backend does not read.
        attention_kwargs={"valid_kv_len": 30, "indices_k": None, "thw": (2, 2, 2), "spargeattn_simthreshold": 0.6},
    )
    call.update(overrides)
    return call


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")


def _reason(**overrides):
    call = _plain_call(**overrides)
    return usp_sdma.unsupported_reason(call.pop("query"), call.pop("key"), call.pop("value"), **call)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"joint_strategy": "front"},
        {"joint_strategy": "rear", "attention_kwargs": {}},
        {"extra_inputs": [("gate", torch.zeros(1, 8, 4, 16))]},
        # Grouped-query K/V, and joint Q rows: other heads or rows than Q.
        {"key": torch.zeros(1, 2, 4, 16), "value": torch.zeros(1, 2, 4, 16)},
        {"query": torch.zeros(1, 8, 6, 16)},
    ],
)
def test_ulysses_calls_the_sdma_path_covers(rocm, overrides):
    assert _reason(**overrides) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"ring_world_size": 2},
        {"fp8_comms": object()},
        {"has_kv_cache": True},
        {"head_balanced": True},
        {"value": torch.zeros(1, 8, 4, 32)},
        {"extra_inputs": [("gate", torch.zeros(1, 8, 5, 16))]},
        {"backend": AttentionBackendType.AITER_SPARGE},
        {"attention_kwargs": {"attn_mask": torch.ones(4, 4, dtype=torch.bool)}},
        # A trailing pad sits between the main and the rear joint rows.
        {"joint_strategy": "rear"},
        {"joint_strategy": "front", "is_causal": True},
    ],
)
def test_calls_the_sdma_path_does_not_cover_stay_on_rccl(rocm, overrides):
    assert _reason(**overrides) is not None


def test_cuda_builds_stay_on_rccl(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert _reason() is not None


def test_runner_options_select_the_transport():
    from xfuser.config.args import FlexibleArgumentParser, xFuserArgs

    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "Wan2.2-T2V",
            "--ulysses_a2a",
            "sdma",
            "--ulysses_sdma_head_groups",
            "3",
            "--ulysses_sdma_group_streams",
        ]
    )
    assert (
        args.ulysses_a2a,
        args.ulysses_sdma_head_groups,
        args.ulysses_sdma_group_streams,
        args.ulysses_sdma_overlap_qkv,
    ) == ("sdma", 3, True, False)
    assert parser.parse_args(["--model", "Wan2.2-T2V"]).ulysses_a2a == "default"


@pytest.mark.parametrize(("mode", "head_groups"), [("rccl", 1), ("sdma", 0)])
def test_ulysses_a2a_config_rejects_unknown_modes_and_empty_groups(mode, head_groups):
    from xfuser.config.ulysses_a2a import UlyssesA2AConfig

    with pytest.raises(ValueError):
        UlyssesA2AConfig(mode, head_groups)


@pytest.mark.parametrize(
    "option",
    [{"ulysses_sdma_head_groups": 2}, {"ulysses_sdma_group_streams": True}, {"ulysses_sdma_overlap_qkv": True}],
)
def test_sdma_tuning_options_need_the_sdma_transport(option):
    from xfuser.config.args import xFuserArgs

    with pytest.raises(ValueError, match="need --ulysses_a2a sdma"):
        xFuserArgs(model="test", ulysses_degree=8, **option)


def test_attention_a2a_and_the_sdma_transport_exclude_each_other():
    from xfuser.config.args import xFuserArgs

    common = dict(model="test", ulysses_degree=8, attention_backend="aiter_mxfp8", attention_a2a="mxfp8-e4m3")
    with pytest.raises(ValueError, match="mutually exclusive"):
        xFuserArgs(ulysses_a2a="sdma", **common)
    assert xFuserArgs(**common).attention_a2a == "mxfp8-e4m3"
