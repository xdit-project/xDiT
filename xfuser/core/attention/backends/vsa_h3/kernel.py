"""VSA-H3 on either kernel.

Both select the same key tiles and differ only in how they read them, so
everything around the call is shared. Nothing is permuted here: the padded tile
buffers FlexAttention needs are built inside h3_vsa_attention, which keeps the
gate and the compression branch out of tile order entirely.
"""

import logging

from xfuser.core.attention.backends.sdpa.kernel import sdpa
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.spec import AttnCall
from xfuser.logger import init_logger, log_once

from . import attention as h3

logger = init_logger(__name__)

# Calls without VSA-H3 metadata run dense. Unlike AITER_VSA, this backend does
# not require AITER -- it runs on CUDA too -- so aiter/kernel.py, which imports
# aiter at module level, cannot be imported outright here.
_DENSE = resolve("xfuser.core.attention.backends.aiter.kernel:aiter_attention") or sdpa


def _vsa_h3(query, key, value, call: AttnCall, *, use_triton: bool):
    """USP has already gathered the sequence; the compression gate rides the
    same exchange. Without that metadata -- the MiniMax-H3 token refiner --
    this runs dense, the way VSA does without thw."""
    kwargs = call.attention_kwargs
    metadata = kwargs.get("vsa_h3_metadata")
    gate = kwargs.get("vsa_h3_gate")
    if metadata is None or gate is None:
        return _DENSE(query, key, value, call)

    if use_triton and not h3.h3_vsa_triton_is_usable(query.device):
        log_once(
            logger, ("vsa_h3_triton", str(query.device)),
            f"TRITON_VSA_H3 cannot run its kernel on {query.device}, falling "
            f"back to the FlexAttention path. Select FLEX_VSA_H3 to ask for "
            f"it directly.",
            level=logging.WARNING,
        )
        use_triton = False

    sequence_length = metadata.total_seq_length
    gathered_length = query.shape[2]
    query, key, value, gate = (
        t[:, :, :sequence_length] for t in (query, key, value, gate)
    )

    packed = h3.h3_vsa_attention(
        query, key, value, gate, metadata, use_triton=use_triton
    )

    if gathered_length > sequence_length:
        padded = packed.new_zeros(
            packed.shape[0], packed.shape[1], gathered_length, packed.shape[3]
        )
        padded[:, :, :sequence_length] = packed
        packed = padded

    return packed, None


def flex_vsa_h3(query, key, value, call: AttnCall):
    """Through FlexAttention: portable, and the selection reference."""
    return _vsa_h3(query, key, value, call, use_triton=False)


def triton_vsa_h3(query, key, value, call: AttnCall):
    """Through the hand-written kernel, FlexAttention where it cannot run."""
    return _vsa_h3(query, key, value, call, use_triton=True)
