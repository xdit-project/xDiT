import dataclasses
import weakref

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


class MaskMetaCache:
    """The AttentionMaskWithMeta built for the last few masks a model saw.

    A denoising loop hands the transformer the same mask tensor at every step,
    and building the metadata costs a host sync, so it is built once per mask.
    Entries are keyed on the mask tensor itself -- a weak reference and its
    version counter -- never on its address: once one request's mask is freed,
    the next request's mask of the same shape commonly lands at that address,
    and an address key would hand it the previous request's valid positions.
    A freed mask's entry can never match, and an in-place write to a cached
    mask bumps its version so that it misses too.

    Bounded at ``capacity`` entries, least recently used evicted: classifier-
    free guidance alternates two masks per step, and a single entry would
    rebuild on every call.
    """

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._entries: list = []  # [(weakref to mask, version, extra, meta)], most recent last

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _version(mask: torch.Tensor):
        # Inference-mode tensors keep no version counter.
        return None if mask.is_inference() else mask._version

    # Weak references and version counters are host-side bookkeeping that
    # Dynamo cannot trace; keep the lookup, and the build behind it, eager.
    @torch.compiler.disable
    def get(self, mask: torch.Tensor, build=make_attn_mask_with_meta, *extra) -> AttentionMaskWithMeta:
        """``build(mask, *extra)``, called only when no entry matches.

        ``extra`` is whatever else the metadata depends on, such as a padded
        length, and is part of the key, so it must be hashable plain values.
        """
        version = self._version(mask)
        live = []
        hit = None
        for entry in self._entries:
            cached = entry[0]()
            if cached is None:
                continue  # its mask was freed; nothing can match it again
            if cached is mask and entry[1] != version:
                continue  # its mask was written in place since
            if cached is mask and entry[2] == extra:
                hit = entry
            else:
                live.append(entry)
        if hit is None:
            hit = (weakref.ref(mask), version, extra, build(mask, *extra))
        self._entries = (live + [hit])[-self.capacity :]
        return hit[3]
