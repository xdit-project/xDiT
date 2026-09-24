"""Tensor layout helpers shared by kernel modules.

USP hands backends BHSD. Most kernels want BSHD, and several support a varlen
path where K/V are packed with per-sequence offsets. These are plain helpers,
not a framework a kernel has to conform to -- a backend that needs a different
shape just writes what it needs.
"""

from dataclasses import dataclass

import torch

from xfuser.core.attention.spec import VarlenPacking

_BHSD_TO_BSHD = [0, 2, 1, 3]


def to_bshd(*tensors: torch.Tensor, contiguous: bool = False):
    """BHSD -> BSHD. Several kernels require contiguity; pass it explicitly
    rather than guessing, because the cost is real and so is the bug."""
    out = tuple(torch.permute(t, _BHSD_TO_BSHD) for t in tensors)
    if contiguous:
        out = tuple(t.contiguous() for t in out)
    return out[0] if len(out) == 1 else out


def from_bshd(tensor: torch.Tensor) -> torch.Tensor:
    """BSHD -> BHSD."""
    return torch.permute(tensor, _BHSD_TO_BSHD)


def make_contiguous(*tensors: torch.Tensor):
    """Contiguous copies, for kernels that take a layout argument instead of a
    permuted view and so have no to_bshd call to ask for it."""
    out = tuple(t.contiguous() for t in tensors)
    return out[0] if len(out) == 1 else out


@dataclass(frozen=True)
class PackedQKV:
    """Flattened Q with packed K/V, ready for a varlen kernel."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    batch: int
    seq_len: int
    heads: int
    head_dim: int

    def unflatten(self, output: torch.Tensor) -> torch.Tensor:
        """Flat (B*S, H, D) back to BSHD."""
        return output.reshape(self.batch, self.seq_len, self.heads, self.head_dim)


def pack_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packing: VarlenPacking,
) -> PackedQKV:
    """Pack K/V for a varlen kernel. Tensors are BSHD. Q is never filtered:
    all B*S query positions are kept."""
    batch, seq_len, heads, head_dim = query.shape
    flat = (batch * seq_len, heads, head_dim)
    return PackedQKV(
        q=query.reshape(flat),
        k=torch.index_select(key.reshape(flat), 0, packing.indices_k),
        v=torch.index_select(value.reshape(flat), 0, packing.indices_k),
        cu_seqlens_q=torch.arange(
            0, batch + 1, dtype=torch.int32, device=query.device
        ) * seq_len,
        cu_seqlens_k=packing.cu_seqlens_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=packing.max_seqlen_k,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
    )
