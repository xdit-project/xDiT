"""Query-row split of the attention DistVAE's parallel decoder runs whole on every rank."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from distvae.vae import ParallelContext


def attention_rows(attn, hidden_states: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """``attn(hidden_states).narrow(2, start, length)`` for a diffusers self-attention with
    ``AttnProcessor2_0``, computing only the queries of those rows.

    The group norm, keys and values still need every row; the attention, the query and the
    output projection, which grow with the number of queries, run on ``length`` rows only.
    """
    batch_size, channel, height, width = hidden_states.shape
    residual = hidden_states.narrow(2, start, length)
    tokens = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    if attn.group_norm is not None:
        tokens = attn.group_norm(tokens.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(tokens[:, start * width : (start + length) * width])
    key = attn.to_k(tokens)
    value = attn.to_v(tokens)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
    out = attn.to_out[1](attn.to_out[0](out))
    out = out.transpose(-1, -2).reshape(batch_size, channel, length, width)
    if attn.residual_connection:
        out = out + residual
    return out / attn.rescale_output_factor


class RowSplitAttnProcessor:
    """``AttnProcessor2_0`` for a self-attention that every rank of the VAE group runs on the
    whole image: each rank computes an equal band of query rows, and an all-gather rebuilds the
    output. Calls it cannot split this way (cross-attention, masks, a height that does not
    divide by the group size) run whole.
    """

    def __init__(self, parallel_context: ParallelContext):
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.parallel_context = parallel_context
        self.whole = AttnProcessor2_0()

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs
    ):
        world_size = self.parallel_context.world_size
        splittable = (
            hidden_states.ndim == 4
            and hidden_states.shape[2] % world_size == 0
            and encoder_hidden_states is None
            and attention_mask is None
            and temb is None
            and not args
            and not kwargs
        )
        if not splittable:
            return self.whole(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        rows = hidden_states.shape[2] // world_size
        local = attention_rows(attn, hidden_states, self.parallel_context.rank * rows, rows).contiguous()
        # Gloo requires the gather output concatenated along dim 0, not stacked.
        bands = torch.empty((world_size * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(bands, local, group=self.parallel_context.group)
        return bands.view(world_size, *local.shape).permute(1, 2, 0, 3, 4).reshape(hidden_states.shape)


def _splittable_attention(attn) -> bool:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    return (
        type(attn) is Attention
        and type(attn.processor) is AttnProcessor2_0
        and attn.spatial_norm is None
        and not attn.is_cross_attention
    )


def split_replicated_vae_attention(vae) -> int:
    """Split the mid-block attention that DistVAE's 2D decoder runs whole on every rank, and
    return how many attentions were split.

    ``DecoderAdapter`` shards from the up blocks on; its mid block, attention included, decodes
    the whole latent on every rank. At 2048² that is a 65k-token attention per rank.
    """
    from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

    adapter = getattr(vae, "decoder", None)
    if not isinstance(adapter, DecoderAdapter) or adapter.parallel_context.world_size == 1:
        return 0
    mid_block = getattr(adapter.decoder, "mid_block", None)
    resnets = getattr(mid_block, "resnets", None) or ()
    # Adapted resnets would mean the mid block runs on this rank's rows only.
    if any(not type(resnet).__module__.startswith("diffusers.") for resnet in resnets):
        return 0
    split = 0
    for attn in getattr(mid_block, "attentions", None) or ():
        if attn is not None and _splittable_attention(attn):
            attn.set_processor(RowSplitAttnProcessor(adapter.parallel_context))
            split += 1
    return split
