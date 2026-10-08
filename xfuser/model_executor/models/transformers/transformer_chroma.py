"""Chroma transformer with Ulysses sequence parallelism.

Chroma is FLUX-shaped (dual-stream blocks followed by single-stream blocks over
the joint ``[text, image]`` sequence), but unlike FLUX it attends with a text
mask. The diffusers pipeline builds a 1-D mask over ``[text, image]`` in the
model dtype (ones for the prompt tokens plus one pad token and for every image
token, zeros for the remaining pad tokens), and each block forms
``mask[:, None, :, None] * mask[:, None, None, :]``. A floating point mask is an
additive bias to scaled dot-product attention, so the reference output adds
``+1`` to the logits of valid-valid token pairs and ``0`` elsewhere; masked
pad tokens are still attended. This module reproduces that bias exactly,
including under Ulysses, where it must follow the token order each rank sees
after the all-to-all.
"""

from typing import Any, Optional

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_chroma import (
    ChromaTransformer2DModel,
)
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    _get_qkv_projections,
)

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from xfuser.model_executor.layers.usp import USP

# Keyword the wrapper uses to hand the joint-sequence bias to every attention
# processor through the blocks' ``joint_attention_kwargs``.
ATTN_BIAS_KWARG = "xfuser_attn_bias"

# Backends that consume ``attention_kwargs["attn_mask"]``. Any other backend
# would silently drop the bias.
MASK_ATTENTION_BACKENDS = frozenset({AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH})


def _sp_order(tokens: torch.Tensor, num_txt: int, sp_world_size: int) -> torch.Tensor:
    """Reorder ``[text, image]`` tokens into the order Ulysses attends over.

    Rank ``r`` holds ``[text_r, image_r]`` locally and the all-to-all
    concatenates the ranks in order, so the attended sequence is
    ``[text_0, image_0, text_1, image_1, ...]``. Both segment lengths must
    already be divisible by ``sp_world_size``.
    """
    batch = tokens.shape[0]
    text = tokens[:, :num_txt].reshape(batch, sp_world_size, -1)
    image = tokens[:, num_txt:].reshape(batch, sp_world_size, -1)
    return torch.cat([text, image], dim=2).reshape(batch, -1)


def chroma_attention_bias(
    attention_mask: Optional[torch.Tensor],
    num_txt: int,
    num_img: int,
    txt_pad: int = 0,
    img_pad: int = 0,
    sp_world_size: int = 1,
) -> Optional[torch.Tensor]:
    """Build the pairwise attention mask Chroma's blocks would apply.

    Args:
        attention_mask: ``[batch, num_txt + num_img]`` mask over the unpadded
            ``[text, image]`` sequence, as the diffusers pipeline passes it, or
            None.
        txt_pad / img_pad: zero tokens appended to each segment so that it
            splits evenly across ``sp_world_size`` ranks. They are excluded
            from attention entirely, which leaves every real token's output
            unchanged.

    Returns:
        ``[batch, 1, S, S]`` (or a broadcastable key mask when no mask was
        given) in the Ulysses token order of :func:`_sp_order`, or None when
        there is nothing to mask. With no padding and one rank this is exactly
        the tensor diffusers' blocks compute, in the same dtype, so a floating
        point mask stays an additive bias and a boolean mask stays a hard one.
    """
    if attention_mask is None and txt_pad == 0 and img_pad == 0:
        return None

    padded_txt = num_txt + txt_pad

    def pad(tokens: torch.Tensor, value) -> torch.Tensor:
        text, image = tokens[:, :num_txt], tokens[:, num_txt:]
        text_pad = text.new_full((tokens.shape[0], txt_pad), value)
        image_pad = image.new_full((tokens.shape[0], img_pad), value)
        padded = torch.cat([text, text_pad, image, image_pad], dim=1)
        return _sp_order(padded, padded_txt, sp_world_size)

    batch = attention_mask.shape[0] if attention_mask is not None else 1
    device = attention_mask.device if attention_mask is not None else None
    real = pad(torch.ones(batch, num_txt + num_img, dtype=torch.bool, device=device), False)

    if attention_mask is None:
        # Only SP padding to hide: every query sees every real key.
        return real[:, None, None, :]

    if attention_mask.shape[1] != num_txt + num_img:
        raise ValueError(
            "Chroma's attention_mask must cover the text and image tokens, got "
            f"{attention_mask.shape[1]} entries for {num_txt} text and {num_img} "
            "image tokens."
        )

    mask = pad(attention_mask, 0)
    # Same expression as ChromaTransformerBlock / ChromaSingleTransformerBlock.
    pair = mask[:, None, None, :] * mask[:, None, :, None]
    if txt_pad == 0 and img_pad == 0:
        return pair

    real_keys = real[:, None, None, :]
    if pair.dtype == torch.bool:
        # Padded query rows would otherwise mask every key and produce NaN,
        # which then reaches later layers through the padded text tokens.
        padded_queries = ~real[:, None, :, None]
        return (pair | padded_queries) & real_keys
    return pair.masked_fill(~real_keys, float("-inf"))


class xFuserChromaAttnProcessor(FluxAttnProcessor):
    """FLUX attention for Chroma, routed through USP with Chroma's bias.

    Follows the non-PipeFusion path of xFuserFluxAttnProcessor, but instead of
    dropping the mask it passes the pairwise bias from
    :func:`chroma_attention_bias` to an attention backend that applies it.
    """

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        xfuser_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError(
                "xFuserChromaAttnProcessor takes its mask through "
                f"joint_attention_kwargs['{ATTN_BIAS_KWARG}']; a per-block "
                "attention_mask would be in the wrong token order under "
                "sequence parallelism."
            )

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.added_kv_proj_dim is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        backend = None
        attention_kwargs = None
        if xfuser_attn_bias is not None:
            backend = get_runtime_state().attention_backend
            if backend not in MASK_ATTENTION_BACKENDS:
                backend = AttentionBackendType.SDPA
            attention_kwargs = {"attn_mask": xfuser_attn_bias}

        hidden_states = USP(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            attn_layer=attn,
            combine_qkv_a2a=True,
            backend=backend,
            attention_kwargs=attention_kwargs,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


def _pad_tokens(tensor: torch.Tensor, padding: int, dim: int) -> torch.Tensor:
    if padding == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = padding
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


class xFuserChromaTransformer2DWrapper(ChromaTransformer2DModel):
    """ChromaTransformer2DModel with Ulysses sequence parallelism.

    Each rank runs the stock blocks on its shard of the text tokens and of the
    image tokens; the attention processors exchange heads for sequence and
    apply Chroma's attention bias. Classifier-free guidance is not handled
    here: Chroma runs the conditional and unconditional branches as separate
    calls, which xFuserChromaPipeline distributes over the CFG group.
    """

    @classmethod
    def from_config(cls, config, return_unused_kwargs=False, **kwargs):
        """Keep the parent config signature, then install xDiT processors."""
        result = super().from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        model = result[0] if return_unused_kwargs else result
        model.install_xdit_attention_processors()
        return result

    def install_xdit_attention_processors(self):
        for block in (*self.transformer_blocks, *self.single_transformer_blocks):
            block.attn.set_processor(xFuserChromaAttnProcessor())
        return self

    def _check_parallel_config(self) -> None:
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "Chroma supports Ulysses sequence parallelism only; set "
                "--ring_degree 1. Ring attention does not shard Chroma's "
                "attention bias per ring step."
            )
        heads = self.config.num_attention_heads
        ulysses = get_ulysses_parallel_world_size()
        if heads % ulysses != 0:
            raise ValueError(
                f"Chroma's {heads} attention heads must be divisible by the Ulysses degree, got {ulysses}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        sp_world_size = get_sequence_parallel_world_size()
        if sp_world_size > 1:
            self._check_parallel_config()
            if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
                raise NotImplementedError("ControlNet residuals are not supported with sequence parallelism.")

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        num_txt = encoder_hidden_states.shape[1]
        num_img = hidden_states.shape[1]
        txt_pad = -num_txt % sp_world_size
        img_pad = -num_img % sp_world_size

        attn_bias = chroma_attention_bias(attention_mask, num_txt, num_img, txt_pad, img_pad, sp_world_size)
        joint_attention_kwargs = dict(joint_attention_kwargs or {})
        if attn_bias is not None:
            joint_attention_kwargs[ATTN_BIAS_KWARG] = attn_bias

        if sp_world_size > 1:
            rank = get_sequence_parallel_rank()
            hidden_states = _pad_tokens(hidden_states, img_pad, dim=1)
            hidden_states = hidden_states.chunk(sp_world_size, dim=1)[rank]
            img_ids = _pad_tokens(img_ids, img_pad, dim=0)
            img_ids = img_ids.chunk(sp_world_size, dim=0)[rank]
            encoder_hidden_states = _pad_tokens(encoder_hidden_states, txt_pad, dim=1)
            encoder_hidden_states = encoder_hidden_states.chunk(sp_world_size, dim=1)[rank]
            txt_ids = _pad_tokens(txt_ids, txt_pad, dim=0)
            txt_ids = txt_ids.chunk(sp_world_size, dim=0)[rank]

        sample = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=None,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]

        if sp_world_size > 1:
            sample = get_sp_group().all_gather(sample.contiguous(), dim=1)
            sample = sample[:, :num_img]

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)
