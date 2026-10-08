"""PyTorch's own attention: the generic dispatcher plus its three aten backends,
and cuDNN.

These are the reference implementations -- always present, no vendor library,
no layout conversion (aten takes BHSD directly).
"""

import torch
import torch.nn.functional as F

from xfuser.core.attention.spec import AttnCall

aten = torch.ops.aten
_BIAS_ALIGNMENT = 16


def sdpa(query, key, value, call: AttnCall):
    """Let PyTorch pick the backend."""
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=call.attention_kwargs.get("attn_mask"),
        dropout_p=call.dropout_p,
        is_causal=call.is_causal,
    )
    return output, None


def sdpa_flash(query, key, value, call: AttnCall):
    output, softmax_lse, *_ = aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p=call.dropout_p,
        is_causal=call.is_causal,
    )
    return output, softmax_lse


def sdpa_math(query, key, value, call: AttnCall):
    attn_mask = call.attention_kwargs.get("attn_mask")
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        # The aten op adds whatever mask it is given to the scores, so a
        # boolean one would add 0/1 rather than exclude keys. The public
        # F.scaled_dot_product_attention converts it first; do the same.
        attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill_(~attn_mask, float("-inf"))
    output, attn_weights = aten._scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=call.dropout_p,
        is_causal=call.is_causal,
    )
    return output, attn_weights


def sdpa_efficient(query, key, value, call: AttnCall):
    output, softmax_lse, *_ = aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=_attn_bias(query, key, call.attention_kwargs.get("attn_mask")),
        compute_log_sumexp=True,
        dropout_p=call.dropout_p,
        is_causal=call.is_causal,
    )
    # The kernel pads the log-sum-exp to a multiple of 32; ring merges need one per query.
    return output, softmax_lse[..., : query.shape[2]]


def _attn_bias(query, key, attn_mask):
    """attn_mask as the additive bias the aten kernels take, or None.

    Does what F.scaled_dot_product_attention does before it reaches a kernel: the
    aten ops add whatever they are given to the scores, so a boolean mask becomes
    0 / -inf, and the bias is broadcast to the full score shape. The memory-efficient
    kernel also needs the bias's row stride aligned to 16 elements, so the key
    dimension is padded and sliced back.
    """
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill_(~attn_mask, float("-inf"))
    else:
        attn_mask = attn_mask.to(query.dtype)
    kv_len = attn_mask.shape[-1]
    if kv_len % _BIAS_ALIGNMENT:
        attn_mask = F.pad(attn_mask, (0, _BIAS_ALIGNMENT - kv_len % _BIAS_ALIGNMENT))[..., :kv_len]
    return attn_mask.expand(query.shape[0], query.shape[1], query.shape[2], key.shape[2])


def cudnn(query, key, value, call: AttnCall):
    output, softmax_lse, *_ = aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=_attn_bias(query, key, call.attention_kwargs.get("attn_mask")),
        compute_log_sumexp=True,
        dropout_p=call.dropout_p,
        is_causal=call.is_causal,
    )
    return output, softmax_lse.squeeze(-1)
