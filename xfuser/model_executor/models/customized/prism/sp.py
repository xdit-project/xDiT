"""Sequence-parallel plumbing for Prism's two towers.

Prism ships its own Ulysses implementation with uneven ``torch.chunk`` shards.
Here each stream is padded at its end to a multiple of the SP degree and split
evenly instead, the way the other xDiT transformers do it, so the padding is
always a trailing block of keys once a rank holds the full sequence. Attention
then drops that block by length (``valid_kv_len``) rather than by mask, which
every attention backend serves.

Which collective an attention uses follows from the sizes involved:

* video self-attention and the v2a bridge (audio queries over video keys) run
  Ulysses through :func:`usp_attention`, because the video sequence is long;
* audio self-attention and the a2v bridge (video queries over audio keys) gather
  the short audio keys and values onto every rank and attend locally;
* text cross-attention needs no communication: the text is replicated.

Only video self-attention runs on the main attention backend, the one a sparse
backend such as TRITON_BSA replaces; every other call runs on the
cross-attention backend, which defaults to the main one.
"""

from dataclasses import dataclass, field

import torch

from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.model_executor.layers.usp import USP, attention


@dataclass(frozen=True)
class SeqShard:
    """Global length of one stream and the padding appended before the split."""

    length: int
    pad: int

    @classmethod
    def for_length(cls, length: int) -> "SeqShard":
        world_size = get_sequence_parallel_world_size()
        return cls(length=length, pad=(world_size - length % world_size) % world_size)

    def split(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """This rank's equal-sized chunk of ``x`` after zero-padding its end."""
        world_size = get_sequence_parallel_world_size()
        if world_size == 1:
            return x
        if self.pad:
            pad_shape = list(x.shape)
            pad_shape[dim] = self.pad
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
        return x.chunk(world_size, dim=dim)[get_sequence_parallel_rank()]

    def gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Reassemble the full stream from every rank's chunk and drop the padding."""
        if get_sequence_parallel_world_size() == 1:
            return x
        x = get_sp_group().all_gather(x.contiguous(), dim=dim)
        return x.narrow(dim, 0, self.length)


@dataclass(frozen=True)
class PrismSeqInfo:
    video: SeqShard
    audio: SeqShard
    # Published to video self-attention only, which is where a sparse backend
    # such as TRITON_BSA reads the token grid; every other call stays dense.
    video_attention_kwargs: dict = field(default_factory=dict)


def _bhsd(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return x.unflatten(-1, (num_heads, -1)).transpose(1, 2)


def _bsd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).flatten(2)


def cross_backend():
    """Backend for everything but video self-attention."""
    return get_runtime_state().get_cross_attention_backend()


def usp_attention(q, k, v, num_heads: int, kv: SeqShard, attention_kwargs=None, backend=None) -> torch.Tensor:
    """Ulysses attention over SP-sharded ``[B, S_local, H*D]`` queries and keys.

    Heads are zero-padded up to a multiple of the Ulysses degree when they do not
    divide evenly (the 12-head audio space under Ulysses 8). A zero head has zero
    queries, keys and values, so it attends to nothing and is sliced off after.
    """
    attention_kwargs = dict(attention_kwargs or {})
    if kv.pad:
        attention_kwargs["valid_kv_len"] = kv.length
    world_size = get_sequence_parallel_world_size()
    q, k, v = (_bhsd(t, num_heads) for t in (q, k, v))
    padded_heads = -(-num_heads // world_size) * world_size
    if padded_heads != num_heads:
        q, k, v = (
            torch.cat([t, t.new_zeros(t.shape[0], padded_heads - num_heads, *t.shape[2:])], dim=1) for t in (q, k, v)
        )
    out = USP(q, k, v, attention_kwargs=attention_kwargs or None, backend=backend)
    return _bsd(out[:, :num_heads])


def gathered_kv_attention(q, k, v, num_heads: int, kv: SeqShard, backend=None) -> torch.Tensor:
    """Local queries against keys and values gathered from the whole SP group."""
    k = kv.gather(k, dim=1)
    v = kv.gather(v, dim=1)
    return local_attention(q, k, v, num_heads, backend=backend)


def local_attention(q, k, v, num_heads: int, backend=None) -> torch.Tensor:
    """Attention with no sequence parallelism: every key is already on this rank."""
    out = attention(_bhsd(q, num_heads), _bhsd(k, num_heads), _bhsd(v, num_heads), backend=backend)
    return _bsd(out)
