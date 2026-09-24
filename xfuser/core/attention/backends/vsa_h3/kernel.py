"""VSA-H3 tiling through FlexAttention."""

from xfuser.core.attention.backends.sdpa.kernel import sdpa
from xfuser.core.attention.numerics.layout import make_contiguous
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.spec import AttnCall

from . import attention as h3

# Calls without VSA-H3 metadata run dense. Unlike AITER_VSA, this backend does
# not require AITER -- it runs on CUDA too -- so aiter/kernel.py, which imports
# aiter at module level, cannot be imported outright here.
_DENSE = resolve("xfuser.core.attention.backends.aiter.kernel:aiter_attention") or sdpa


def flex_vsa_h3(query, key, value, call: AttnCall):
    """FastH3's 64-token VSA-H3. USP has already gathered the sequence; the
    compression gate rides the same exchange. Without that metadata -- the
    MiniMax-H3 token refiner -- this runs dense, the way VSA does without thw.
    """
    kwargs = call.attention_kwargs
    metadata = kwargs.get("vsa_h3_metadata")
    gate = kwargs.get("vsa_h3_gate")
    if metadata is None or gate is None:
        return _DENSE(query, key, value, call)

    sequence_length = metadata.total_seq_length
    gathered_length = query.shape[2]
    query, key, value, gate = (
        t[:, :, :sequence_length] for t in (query, key, value, gate)
    )

    def tile(tensor):
        return make_contiguous(
            h3.tile_h3_vsa_tensor(tensor.transpose(1, 2), metadata).transpose(1, 2)
        )

    sparse, compressed = h3.flex_h3_vsa_attention(
        tile(query), tile(key), tile(value), metadata
    )
    tiled = sparse + compressed.to(sparse.dtype) * tile(gate)
    packed = h3.untile_h3_vsa_tensor(tiled.transpose(1, 2), metadata).transpose(1, 2)

    if gathered_length > sequence_length:
        padded = packed.new_zeros(
            packed.shape[0], packed.shape[1], gathered_length, packed.shape[3]
        )
        padded[:, :, :sequence_length] = packed
        packed = padded

    return packed, None


