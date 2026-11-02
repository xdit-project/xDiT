# SPDX-License-Identifier: Apache-2.0
"""Head-sharded Ulysses parity for the VDN-H3 hybrid attention.

The Ulysses path (`VDNMiniMaxH3AttnProcessor._compute_ulysses`) all-to-alls q/k/v
into a per-head shard, runs both branches on the local heads, and all-to-alls the
result back. Its head slicing and the frame-mean all-reduce live only in that
path, so this spawns world_size=2 and checks the gathered shard outputs match the
single-GPU `_compute` reference on the same weights and inputs.
"""

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.accelerator


class _ToyAttn(nn.Module):
    """Minimal stand-in for the diffusers MiniMaxH3Attention surface the VDN
    processor reads: fused qkv, QK-norm, and the shared output projections."""

    def __init__(self, hidden: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.fused_projections = True
        self.to_qkv = nn.Linear(hidden, 3 * heads * head_dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, hidden, bias=False), nn.Identity()])

    def set_processor(self, processor) -> None:
        self.processor = processor


def _build_inputs(hidden, heads, head_dim, *, radius, text_len, num_frames, gh, gw, device):
    from xfuser.model_executor.models.transformers.vdn_minimax_h3.attention import (
        attach_vdn_modules,
        build_vdn_meta,
    )
    from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
        VDNHybridAttentionArchConfig,
    )
    from xfuser.model_executor.models.transformers.vdn_minimax_h3.window_softmax import set_window_backend

    set_window_backend("sdpa")
    tpf = gh * gw
    used = text_len + num_frames * tpf
    seq_len = used  # no padding; must be divisible by the world size

    # sana_scaled avoids the Cholesky inverse of ``vdn_solve`` (which needs a
    # positive-definite A that random, untrained weights do not guarantee); the
    # head-sharded path under test is identical regardless of the delta rule.
    # radius >= num_frames makes the window dense (full_cover: linear branch off);
    # a smaller radius leaves gaps so ``need_linear`` is True and the Video-Delta
    # branch (and its batched collectives) run.
    hybrid = VDNHybridAttentionArchConfig(
        chunk=1,
        radius=radius,
        anchor_frames="both",
        linear_head_dim=head_dim,
        short_conv=("k", "v"),
        delta_rule="sana_scaled",
    )
    attn = _ToyAttn(hidden, heads, head_dim).to(device).float()
    attach_vdn_modules(attn, hybrid, hidden_size=hidden, num_attention_heads=heads, attention_head_dim=head_dim)
    attn.to(device).float()
    # Shrink random weights hard so the delta scan is strongly contractive and
    # stays well away from the fp divergence that untrained weights hit over
    # frames; parity (the property under test) holds at any magnitude.
    for p in attn.parameters():
        p.mul_(0.02)

    # packed metadata: [text | video]; video position_ids carry the (T, H, W) grid
    pos = torch.zeros(used, 3, dtype=torch.long, device=device)
    row = text_len
    for f in range(num_frames):
        for h in range(gh):
            for w in range(gw):
                pos[row] = torch.tensor([f, h, w], device=device)
                row += 1
    text_indices = torch.arange(text_len, device=device)
    video_indices = torch.arange(text_len, used, device=device)
    audio_indices = torch.empty(0, dtype=torch.long, device=device)
    meta = build_vdn_meta(
        hybrid,
        position_ids=pos,
        text_indices=text_indices,
        audio_indices=audio_indices,
        video_indices=video_indices,
        padded_length=seq_len,
    )
    attn._vdn_meta = meta
    hidden_states = torch.randn(1, seq_len, hidden, device=device)
    return attn, hybrid, hidden_states, seq_len


def _ulysses_parity_worker(rank, world_size, init_method, hidden, heads, head_dim, radius):
    from xfuser.core.distributed import (
        get_ulysses_parallel_rank,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.model_executor.models.transformers.vdn_minimax_h3.attention import (
        VDNMiniMaxH3AttnProcessor,
    )

    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)  # inference-only; the scan uses baddbmm(out=...)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_runtime_state()
    initialize_model_parallel(ring_degree=1, ulysses_degree=world_size)

    # identical weights + inputs on every rank
    torch.manual_seed(1234)
    attn, hybrid, hidden_states, seq_len = _build_inputs(
        hidden, heads, head_dim, radius=radius, text_len=4, num_frames=4, gh=2, gw=2, device=device
    )
    processor = VDNMiniMaxH3AttnProcessor(hybrid)

    # single-GPU reference over the full sequence (no collectives)
    reference = processor._compute(attn, hidden_states, None, attn._vdn_meta)

    # the rank's contiguous sequence shard -> head-sharded Ulysses path
    s_local = seq_len // world_size
    assert get_ulysses_parallel_rank() == rank
    shard = hidden_states[:, rank * s_local : (rank + 1) * s_local, :]
    out_shard = processor(attn, shard, None)

    gathered = [torch.empty_like(out_shard) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, out_shard.contiguous())
    full = torch.cat(gathered, dim=1)

    torch.testing.assert_close(full, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "radius",
    [
        pytest.param(4, id="full_cover"),  # window is dense, linear branch off
        pytest.param(1, id="partial_cover"),  # gaps remain, linear branch + batched collectives run
    ],
)
def test_vdn_ulysses_matches_single_gpu(accelerator_ranks, radius):
    accelerator_ranks(_ulysses_parity_worker, world_size=2, args=(16, 4, 8, radius))
