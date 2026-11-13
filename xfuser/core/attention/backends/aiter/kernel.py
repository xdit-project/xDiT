"""
AITER flash attention.
"""

from typing import Tuple

import torch
from aiter import flash_attn_func, flash_attn_varlen_func
from torch.library import custom_op

from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.spec import AttnCall

# AITER FAv3 on gfx942 supports three different rounding modes:
#   0 = RTNE (round to nearest even)
#   1 = RTNA (round to nearest away)
#   2 = RTZ  (round to zero)
_BF16_ROUNDING_MODE = 2

# The FAv3 hd128 bf16 forward on these archs gives each CU one 256-row query tile.
_V3_TILE_Q = 256
_V3_ARCHS = ("gfx942", "gfx950")
# Split only when the last wave fills at most 1/4 of the CUs.
_TAIL_WAVE_FRACTION = 4
# Shorter K makes the dense tile too cheap to beat the extra launches and merge.
_TAIL_MIN_SEQ_K = 4096
# More K chunks than this only add merge work.
_TAIL_MAX_KV_CHUNKS = 16


# ---------------------------------------------------------------------------
# custom ops
#
# AITER annotates both entry points with `X | None`, which Dynamo cannot wrap
# sourcelessly -- it aborts with "SourcelessBuilder.create does not know how to
# wrap types.UnionType" as soon as it inlines the nested _validate_cu helper,
# so any compiled model reaching either one fails to build a graph. Wrapping
# each makes it a single opaque node.
#
# Registered at module import, which happens at backend selection, outside
# every compiled region -- the same reason Impl resolves there.
# ---------------------------------------------------------------------------


@custom_op("xfuser::aiter_attn", mutates_args=())
def _dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_lse=True,
        return_attn_probs=False,
        how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
    )


@_dense.register_fake
def _dense_fake(query, key, value, dropout_p, is_causal):
    batch, seq_len, heads, _ = query.shape
    return (
        torch.empty_like(query),
        query.new_empty((batch, heads, seq_len), dtype=torch.float32),
    )


@custom_op("xfuser::aiter_attn_varlen", mutates_args=())
def _varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=is_causal,
        return_lse=True,
        return_attn_probs=False,
        how_v3_bf16_cvt=_BF16_ROUNDING_MODE,
    )


@_varlen.register_fake
def _varlen_fake(
    query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, dropout_p, is_causal
):
    total_q, heads, _ = query.shape
    return (
        torch.empty_like(query),
        query.new_empty((heads, total_q), dtype=torch.float32),
    )


def _tail_split_plan(q, k, v, call: AttnCall) -> Tuple[int, int]:
    """(query rows to take off the dense kernel's last wave, K chunks for them), or (0, 0).

    The kernel launches batch * heads * ceil(seq_q / 256) tiles and each runs over all
    of K. When that just exceeds a multiple of the CU count, the last wave holds a few
    tiles that cost as much as a full wave while the other CUs idle. Qwen-Image-Edit at
    Ulysses 8 on a 256-CU MI355X launches 258.
    """
    if call.is_causal or call.dropout_p or q.dtype != torch.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return 0, 0
    props = torch.cuda.get_device_properties(q.device)
    if props.gcnArchName.split(":")[0] not in _V3_ARCHS or k.shape[1] < _TAIL_MIN_SEQ_K:
        return 0, 0
    batch, seq_q, heads, _ = q.shape
    cus = props.multi_processor_count
    # Not divmod: Dynamo cannot trace it on the symbolic sequence lengths of a dynamic graph.
    tiles = batch * heads * -(-seq_q // _V3_TILE_Q)
    full_waves, last_wave = tiles // cus, tiles % cus
    if full_waves == 0 or last_wave == 0 or last_wave * _TAIL_WAVE_FRACTION > cus:
        return 0, 0
    tail_rows = seq_q - full_waves * cus // (batch * heads) * _V3_TILE_Q
    chunks = min(_TAIL_MAX_KV_CHUNKS, cus // (batch * heads * -(-tail_rows // _V3_TILE_Q)))
    if chunks < 2:
        return 0, 0
    return tail_rows, chunks


def _tail_split_attention(q, k, v, tail_rows, chunks):
    """Dense attention with the last `tail_rows` query rows computed split-K.

    Each tail row attends to `chunks` slices of K, run as varlen segments that spread
    over all CUs; the partial outputs are merged with their log-sum-exp.
    """
    batch, seq_q, heads, head_dim = q.shape
    seq_k = k.shape[1]
    main_rows = seq_q - tail_rows
    out_main, lse_main = _dense(q[:, :main_rows].contiguous(), k, v, 0.0, False)

    q_tail = q[:, main_rows:].unsqueeze(1).expand(batch, chunks, tail_rows, heads, head_dim)
    q_tail = q_tail.reshape(batch * chunks * tail_rows, heads, head_dim)
    cu_q = torch.arange(batch * chunks + 1, device=q.device, dtype=torch.int32) * tail_rows
    starts = torch.arange(batch, device=q.device).unsqueeze(1) * seq_k
    bounds = torch.arange(chunks, device=q.device) * seq_k // chunks
    cu_k = torch.cat([(starts + bounds).flatten(), starts.new_full((1,), batch * seq_k)]).to(torch.int32)
    out_tail, lse_tail = _varlen(
        q_tail,
        k.reshape(batch * seq_k, heads, head_dim),
        v.reshape(batch * seq_k, heads, head_dim),
        cu_q,
        cu_k,
        tail_rows,
        -(-seq_k // chunks),
        head_dim**-0.5,
        0.0,
        False,
    )
    lse_tail = lse_tail.view(heads, batch, chunks, tail_rows)
    weights = torch.softmax(lse_tail, dim=2).permute(1, 2, 3, 0).unsqueeze(-1)
    out_tail = out_tail.view(batch, chunks, tail_rows, heads, head_dim).float()
    out_tail = (out_tail * weights).sum(1).to(q.dtype)
    lse_tail = torch.logsumexp(lse_tail, dim=2).permute(1, 0, 2)
    return torch.cat([out_main, out_tail], dim=1), torch.cat([lse_main, lse_tail], dim=2)


def aiter_attention(query, key, value, call: AttnCall):
    """BSHD kernel; packs K/V when the model supplies varlen indices."""
    q, k, v = to_bshd(query, key, value, contiguous=True)

    if call.varlen is None:
        tail_rows, chunks = _tail_split_plan(q, k, v, call)
        if tail_rows:
            out, lse = _tail_split_attention(q, k, v, tail_rows, chunks)
        else:
            out, lse = _dense(q, k, v, call.dropout_p, call.is_causal)
    else:
        p = pack_kv(q, k, v, call.varlen)
        out, lse = _varlen(
            p.q,
            p.k,
            p.v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.head_dim**-0.5,
            call.dropout_p,
            call.is_causal,
        )
        out = p.unflatten(out)
        lse = lse.view(p.heads, p.batch, p.seq_len).permute(1, 0, 2).contiguous()

    return from_bshd(out), lse
