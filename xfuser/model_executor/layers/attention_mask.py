import dataclasses
from typing import NamedTuple

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class AttentionMaskWithMeta:
    """Key-padding mask with pre-computed varlen indices.

    Passed as attention_mask through transformer block and attention stacks.
    SDPA backends use only attn_mask. Varlen backends can use the additional
    metadata to avoid re-computing indices and cumulative lengths per layer.

    Follows the flash-attention unpad_input pattern: the nonzero call that
    produces indices_k is made once per forward pass rather than once per layer.
    """

    attn_mask: torch.Tensor  # [B, 1, 1, S]  bool: True = attend, False = pad
    indices_k: torch.Tensor  # [total_valid_k]  int64 flat valid positions
    cu_seqlens_k: torch.Tensor  # [B+1]  int32 cumulative valid-key counts
    max_seqlen_k: int


def make_attn_mask_with_meta(mask_2d: torch.Tensor) -> AttentionMaskWithMeta:
    """Build AttentionMaskWithMeta from a [B, S] key-padding mask.

    sum and cumsum are graph-break-free. nonzero and max().item() each create
    one graph break. Callers should cache the returned object across denoising
    steps so these breaks occur only once per unique mask.
    """
    seqlens_k = mask_2d.sum(-1, dtype=torch.int32)
    cu_seqlens_k = F.pad(seqlens_k.cumsum(0, dtype=torch.int32), (1, 0))
    indices_k = mask_2d.flatten().nonzero(as_tuple=False).flatten()
    max_seqlen_k = int(seqlens_k.max())
    return AttentionMaskWithMeta(
        attn_mask=mask_2d.to(torch.bool)[:, None, None, :],
        indices_k=indices_k,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
    )


class _MaskMetaEntry(NamedTuple):
    mask: torch.Tensor
    extra: tuple
    meta: AttentionMaskWithMeta


class MaskMetaCache:
    """The AttentionMaskWithMeta built for the last few masks a model saw.

    A denoising loop hands the transformer the same mask tensor at every step, and
    building the metadata costs a host sync, so it is built once per mask. Entries match
    on tensor identity, never on address: the next request's mask commonly lands where
    the previous one was freed. Each entry holds its mask, so a cached mask's address
    cannot be reused while the entry exists. Bounded at ``capacity`` entries, least
    recently used evicted first; classifier-free guidance alternates two masks per step.
    """

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._entries: list[_MaskMetaEntry] = []  # most recently used last

    # The lookup, and the host-syncing build behind it, stay out of compiled graphs.
    @torch.compiler.disable
    def get(self, mask: torch.Tensor, build, *extra) -> AttentionMaskWithMeta:
        """``build(mask, *extra)``, called only when no entry matches.

        ``extra`` is whatever else the metadata depends on, such as a padded length,
        and is compared as part of the key.
        """
        for i, entry in enumerate(self._entries):
            if entry.mask is mask and entry.extra == extra:
                self._entries.append(self._entries.pop(i))
                return entry.meta
        entry = _MaskMetaEntry(mask, extra, build(mask, *extra))
        self._entries = (self._entries + [entry])[-self.capacity :]
        return entry.meta
