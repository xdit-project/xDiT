"""Timestep modulation of every block, projected once and shared out over the sequence-parallel ranks.

Each block of a DiT projects the conditioning embedding, one row per sample, through its own
wide linear to get its scales, shifts and gates. Every one of those projections reads all its
weights to produce one row, and every rank computes the same rows. Here the linears are
stacked: each rank projects only its share of the stacked outputs with one GEMM, the ranks
gather the row, and each block's linear returns its slice of it.
"""

from typing import Sequence

import torch
import torch.nn.functional as F

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.model_executor.layers.usp_sdma import plain_linear


class SlicedModulation(torch.nn.Linear):
    """A block's modulation linear that returns its slice of the row the ranks projected together.

    Set by class swap, so the state dict keys stay those of the stock module and its weights stay loaded.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._modulation["row"][:, self._modulation_start : self._modulation_start + self.out_features]


class ShardedModulation(torch.nn.Module):
    """The modulation of `linears`, which all project the same embedding, shared out over the sequence-parallel ranks.

    Holds this rank's share of the stacked weights: a copy, 1 / (sequence-parallel size) of them.
    Calling it with the embedding the linears would get projects the share, gathers the row of
    all ranks, and leaves it for the linears, which are turned into `SlicedModulation`.
    """

    def __init__(self, linears: Sequence[torch.nn.Linear]):
        super().__init__()
        size, rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        total = sum(linear.out_features for linear in linears)
        # Equal shares, as all-gather needs; zero rows fill the last one, past every linear's slice.
        share = -(-total // size)
        first, last = rank * share, min((rank + 1) * share, total)
        weights, biases = [linears[0].weight.data[:0]], [linears[0].bias.data[:0]]
        row = {"row": None}
        start = 0
        for linear in linears:
            begin, end = max(first, start) - start, min(last, start + linear.out_features) - start
            if begin < end:
                weights.append(linear.weight.data[begin:end])
                biases.append(linear.bias.data[begin:end])
            linear._modulation = row
            linear._modulation_start = start
            linear.__class__ = SlicedModulation
            start += linear.out_features
        weight, bias = torch.cat(weights), torch.cat(biases)
        self.register_buffer("weight", F.pad(weight, (0, 0, 0, share - weight.shape[0])), persistent=False)
        self.register_buffer("bias", F.pad(bias, (0, share - bias.shape[0])), persistent=False)
        self._row = row

    def forward(self, embedding: torch.Tensor) -> None:
        local = F.linear(embedding, self.weight, self.bias)
        self._row["row"] = get_sp_group().all_gather(local, dim=-1)


def shard_modulation(
    transformer: torch.nn.Module, linears: Sequence[torch.nn.Module], embedder: torch.nn.Module
) -> int:
    """Share out the modulation `linears` of `transformer`, which project SiLU of what `embedder` returns.

    Nothing changes unless every one of them is a plain linear, with whole and unquantized
    weights. A forward hook on `embedder` projects the row, so once per forward of the
    transformer and inside its compiled graph. Returns the number of linears shared out.
    """
    if not linears or not all(plain_linear(linear) for linear in linears):
        return 0
    transformer.sharded_modulation = ShardedModulation(linears)
    embedder.register_forward_hook(lambda module, args, embedding: transformer.sharded_modulation(F.silu(embedding)))
    return len(linears)
