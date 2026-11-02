from types import SimpleNamespace

import torch

from xfuser.model_executor.models.transformers.vdn_minimax_h3 import attention
from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.linear_branch import (
    MiniMaxH3VDNLinearBranch,
    VDNH3Layout,
)


class _IdentityGroup:
    def all_reduce(self, tensor):
        return tensor


def _layout():
    # [text 0:6 | audio 6:10 | video 10:50 | tail 50:60 | pad 60:64]
    return VDNH3Layout(
        seq_len=64,
        used=60,
        text_len=6,
        video_start=10,
        num_frames=5,
        tokens_per_frame=8,
        frame_height=2,
        frame_width=4,
    )


def test_video_frame_mean_matches_full_frame_mean(monkeypatch):
    monkeypatch.setattr(attention, "get_sp_group", lambda: _IdentityGroup())
    layout = _layout()
    world, channels = 8, 3
    s_local = layout.seq_len // world
    torch.manual_seed(0)
    x = torch.randn(layout.seq_len, channels)
    processor = attention.VDNMiniMaxH3AttnProcessor(None)

    total = sum(
        processor._video_frame_mean(x[rank * s_local : (rank + 1) * s_local], rank, s_local, layout)
        for rank in range(world)
    )
    expected = (
        x[layout.video_start : layout.video_end].view(layout.num_frames, layout.tokens_per_frame, channels).mean(dim=1)
    )
    torch.testing.assert_close(total, expected)


def test_video_frame_mean_traces_without_graph_breaks(monkeypatch):
    monkeypatch.setattr(attention, "get_sp_group", lambda: _IdentityGroup())
    layout = _layout()
    s_local = 8
    processor = attention.VDNMiniMaxH3AttnProcessor(None)
    compiled = torch.compile(processor._video_frame_mean, fullgraph=True, backend="eager")
    for rank in (0, 3, 7):  # no video rows, all video, no video rows
        x = torch.randn(s_local, 3)
        torch.testing.assert_close(
            compiled(x, rank, s_local, layout),
            processor._video_frame_mean(x, rank, s_local, layout),
        )


def test_ulysses_local_branch_is_built_once(monkeypatch):
    monkeypatch.setattr(attention, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(attention, "get_ulysses_parallel_rank", lambda: 2)
    hybrid = VDNHybridAttentionArchConfig(linear_head_dim=16, short_conv=("k", "v"))
    branch = MiniMaxH3VDNLinearBranch(hybrid, hidden_size=32, num_attention_heads=8, attention_head_dim=16)
    for param in branch.parameters():
        param.data.normal_()
    attn = SimpleNamespace(heads=8, linear_attention=branch)

    local = attention.ensure_ulysses_local_branch(attn)

    assert local is attention.ensure_ulysses_local_branch(attn)
    assert local.heads == 2
    torch.testing.assert_close(local.alpha.A_log, branch.alpha.A_log[4:6])
    torch.testing.assert_close(local.short_conv.k_sp["weight"], branch.short_conv.k_sp["weight"][64:96])


def test_ulysses_local_branch_skipped_without_ulysses(monkeypatch):
    monkeypatch.setattr(attention, "get_ulysses_parallel_world_size", lambda: 1)
    assert attention.ensure_ulysses_local_branch(SimpleNamespace(heads=8)) is None
