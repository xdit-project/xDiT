import dataclasses

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

    attn_mask: torch.Tensor  # [B, 1, 1, S]  1 = valid key, 0 = padding
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
        attn_mask=mask_2d[:, None, None, :],
        indices_k=indices_k,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
    )
