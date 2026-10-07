"""Prism block-sparse attention, dense where a call carries no token grid."""

from xfuser.core.attention.backends.sdpa.kernel import sdpa
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.spec import AttnCall

from .bsa import block_sparse_attention_3d

# Cross-attention and the audio stream publish no grid and run dense. This
# backend needs only Triton, so the AITER import cannot be unconditional.
_DENSE = resolve("xfuser.core.attention.backends.aiter.kernel:aiter_attention") or sdpa


def triton_bsa(query, key, value, call: AttnCall):
    """Sparse over the published ``bsa_thw`` grid, dense otherwise.

    USP has already gathered the full sequence for this rank's heads and cut
    trailing pad keys to the grid's length; the gathered queries still carry
    that padding, which is dropped here and restored as zeros afterwards.
    """
    kwargs = call.attention_kwargs
    thw = kwargs.get("bsa_thw")
    # A single latent frame has no temporal blocks to skip; Prism runs it dense too.
    if thw is None or thw[0] <= 1:
        return _DENSE(query, key, value, call)
    seq_len = thw[0] * thw[1] * thw[2]
    if key.shape[2] != seq_len:
        raise ValueError(f"bsa_thw {tuple(thw)} describes {seq_len} keys but the call has {key.shape[2]}.")

    gathered_len = query.shape[2]
    out = block_sparse_attention_3d(
        query[:, :, :seq_len],
        key,
        value,
        tuple(thw),
        chunk_thw=tuple(kwargs.get("bsa_chunk_thw", (4, 4, 4))),
        sparsity=kwargs.get("bsa_sparsity", 0.75),
        cdf_threshold=kwargs.get("bsa_cdf_threshold"),
    )
    if gathered_len > seq_len:
        padded = out.new_zeros(*out.shape[:2], gathered_len, out.shape[3])
        padded[:, :, :seq_len] = out
        out = padded
    return out, None
