import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    _FLEX_BLOCK_SIZE,
    _IMG_TOKENS_PER_SLOT,
    BlockMask,
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
    QwenImage21KVCache,
    QwenImage21KVLayerCache,
    QwenImage21Transformer2DModel,
    _qwenimage21_prefix_segments,
    apply_rotary_emb_qwen,
    build_qwenimage21_block_causal_mask,
)
from diffusers.utils.peft_utils import apply_lora_scale

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from xfuser.model_executor.layers.usp import (
    _combined_qkv_all_to_all,
    _ft_c_output_all_to_all,
    attention,
)
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


def _gather_sequence_scatter_heads(query, key, value, sp_pad):
    """``[B, S/N, H, D]`` per rank -> ``[B, S, H/N, D]``, with the sequence padding trimmed off."""
    if get_ulysses_parallel_world_size() == 1:
        return query, key, value
    query, key, value = _combined_qkv_all_to_all(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    seq_len = query.shape[2] - sp_pad
    return tuple(t[:, :, :seq_len].transpose(1, 2).contiguous() for t in (query, key, value))


def _scatter_sequence_gather_heads(hidden_states, sp_pad):
    """``[B, S, H/N, D]`` -> ``[B, S/N, H, D]`` per rank, re-padding the sequence first."""
    if get_ulysses_parallel_world_size() == 1:
        return hidden_states
    if sp_pad:
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, sp_pad))
    return _ft_c_output_all_to_all(hidden_states.transpose(1, 2)).transpose(1, 2)


def _update_kv_cache(key, value, layer_cache, kv_cache_mode, cache_write_slice):
    if layer_cache is None:
        return key, value
    if kv_cache_mode == "extract" and cache_write_slice is not None:
        # `clone()`, not `contiguous()`: a contiguous prefix slice would pin the whole prefill K/V.
        layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
    elif kv_cache_mode == "cached":
        cached_k, cached_v = layer_cache.get()
        key = torch.cat([cached_k, key], dim=1)
        value = torch.cat([cached_v, value], dim=1)
    return key, value


def _flex_prefill(query, key, value, block_mask):
    seq_len_q = query.shape[1]
    pad_q = math.ceil(seq_len_q / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE - seq_len_q
    pad_kv = math.ceil(key.shape[1] / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE - key.shape[1]
    if pad_q:
        query = F.pad(query, (0, 0, 0, 0, 0, pad_q))
    if pad_kv:
        key = F.pad(key, (0, 0, 0, 0, 0, pad_kv))
        value = F.pad(value, (0, 0, 0, 0, 0, pad_kv))
    out = dispatch_attention_fn(query, key, value, attn_mask=block_mask, dropout_p=0.0, backend="flex")
    return out[:, :seq_len_q]


def _segmented_prefill(query, key, value, segments, key_valid):
    """Exact block-causal prefill: each prefix segment attends to keys ``[0, end)``, text segments with a
    causal triangle over their own keys, then the target attends to everything.

    Mirrors the segment loop in diffusers' ``QwenImage21AttnProcessor`` (diffusers ``bdc2bea``).
    """
    prefix_len = segments[-1][1] if segments else 0
    outputs = []
    for start, end, is_text in segments:
        seg_mask = None
        if is_text:
            seg_len = end - start
            seg_mask = torch.cat(
                [
                    torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                    torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                ],
                dim=1,
            )[None, None]
        if key_valid is not None:
            seg_key_valid = key_valid[:, None, None, :end]
            seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
        outputs.append(
            dispatch_attention_fn(query[:, start:end], key[:, :end], value[:, :end], attn_mask=seg_mask, dropout_p=0.0)
        )
    outputs.append(
        dispatch_attention_fn(
            query[:, prefix_len:],
            key,
            value,
            attn_mask=None if key_valid is None else key_valid[:, None, None, :],
            dropout_p=0.0,
        )
    )
    return torch.cat(outputs, dim=1)


def flex_attention_available() -> bool:
    """Whether ``torch.nn.attention.flex_attention`` imports (torch >= 2.5)."""
    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except ImportError:
        return False
    return True


class _xFuserQwenImage21AttnMixin:
    """Qwen-Image 2.1 attention with Ulysses sequence parallelism and xDiT-backend decode.

    Projections, QK-norm and RoPE run on this rank's slice of the sequence. One all-to-all then gives every
    rank the whole sequence for its share of the heads, so the KV cache, the block-causal prefill and the
    decode all see exactly what they see on a single GPU. With Ulysses off the all-to-alls are no-ops.

    Subclassing the diffusers processors, rather than wrapping them, matters: the model picks which prefill
    metadata to build from ``isinstance`` checks against them.

    Prefill and decode then diverge in this ``__call__``:

    * Eager processor (not a ``QwenImage21FlexAttnProcessor``): step 0 arrives with ``segments`` and no
      ``BlockMask``. ``_segmented_prefill`` calls diffusers' ``dispatch_attention_fn``, which is SDPA.
    * Flex processor, installed only when ``flex_attention_available()`` and the model is compiled: step 0
      arrives as one ``BlockMask``. ``_flex_prefill`` calls ``dispatch_attention_fn(..., backend="flex")``.
    * Decode is the same for both. No mask and no segments, so the call goes through xDiT's ``attention()``
      and ``--attention_backend`` applies. A prompt padding mask stays on ``dispatch_attention_fn``.
    """

    # Padding appended to make the sequence divisible by the Ulysses degree. The block's forward has no
    # slot for it, so `xFuserQwenImage21TransformerWrapper` writes it onto every processor each forward.
    sp_pad = 0

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)
        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
            key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

        query, key, value = _gather_sequence_scatter_heads(query, key, value, self.sp_pad)
        key, value = _update_kv_cache(key, value, layer_cache, kv_cache_mode, cache_write_slice)

        if isinstance(attention_mask, BlockMask):
            hidden_states = _flex_prefill(query, key, value, attention_mask)
        elif segments is not None:
            hidden_states = _segmented_prefill(query, key, value, segments, key_valid)
        elif attention_mask is None:
            # xDiT attention() is [B, H, S, D]; this processor holds [B, S, H, D].
            hidden_states = attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
        else:
            hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0)

        hidden_states = _scatter_sequence_gather_heads(hidden_states, self.sp_pad)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class xFuserQwenImage21AttnProcessor(_xFuserQwenImage21AttnMixin, QwenImage21AttnProcessor):
    """Eager path: exact per-segment SDPA prefill, xDiT-backend decode."""


class xFuserQwenImage21FlexAttnProcessor(_xFuserQwenImage21AttnMixin, QwenImage21FlexAttnProcessor):
    """Compiled path: single flex_attention prefill, xDiT-backend decode."""


class xFuserQwenImage21TransformerWrapper(QwenImage21Transformer2DModel):
    """``QwenImage21Transformer2DModel`` whose block loop runs on a Ulysses slice of the sequence.

    ``forward`` follows diffusers' up to the block loop. The joint sequence, rotary table and per-token
    modulation mask are then padded to a multiple of the Ulysses degree and sharded; everything the attention
    reads over the full sequence (block mask, segments, key-padding mask, cache slice) is left whole, since
    the processors attend over the gathered sequence. The output is gathered before the final norm.

    Sharding uses the sequence-parallel group while the processors' all-to-alls use the Ulysses group;
    they are the same group only because the runner keeps Ring off.

    ``forward`` is a copy of ``QwenImage21Transformer2DModel.forward`` as of diffusers ``bdc2bea``; only the
    sharding block before the block loop, ``target_token_mask=block_modulation_mask`` and the gather after
    it differ.
    """

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size = hidden_states.shape[0]
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError("kv_cache requires `causal_condition=True`.")
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.")

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())

        if kv_cache_mode == "cached":
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            block_segments, block_key_valid = None, None
        else:
            processors = [block.attn.processor for block in self.transformer_blocks]
            needs_block_mask = any(isinstance(processor, QwenImage21FlexAttnProcessor) for processor in processors)
            attention_mask = (
                build_qwenimage21_block_causal_mask(image_ids, joint_key_valid, batch_size, hidden_states.device)
                if needs_block_mask
                else None
            )
            block_segments = (
                None
                if all(isinstance(processor, QwenImage21FlexAttnProcessor) for processor in processors)
                else _qwenimage21_prefix_segments(image_ids, prefix_len)
            )
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            block_key_valid = joint_key_valid

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        sp_pad = (-joint_hidden_states.shape[1]) % sp_world_size
        block_modulation_mask = modulation_mask
        if sp_world_size > 1:
            joint_hidden_states = chunk_and_pad_sequence(joint_hidden_states, sp_rank, sp_world_size, sp_pad, dim=1)
            rotary_emb = chunk_and_pad_sequence(rotary_emb, sp_rank, sp_world_size, sp_pad, dim=0)
            if modulation_mask is not None:
                block_modulation_mask = chunk_and_pad_sequence(modulation_mask, sp_rank, sp_world_size, sp_pad, dim=0)
        for block in self.transformer_blocks:
            block.attn.processor.sp_pad = sp_pad

        for index_block, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                joint_hidden_states = self._gradient_checkpointing_func(
                    block,
                    joint_hidden_states,
                    modulation,
                    rotary_emb,
                    attention_mask,
                    block_modulation_mask,
                    layer_cache,
                    kv_cache_mode,
                    cache_write_slice,
                    block_segments,
                    block_key_valid,
                )
            else:
                joint_hidden_states = block(
                    hidden_states=joint_hidden_states,
                    modulation=modulation,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask,
                    target_token_mask=block_modulation_mask,
                    layer_cache=layer_cache,
                    kv_cache_mode=kv_cache_mode,
                    cache_write_slice=cache_write_slice,
                    segments=block_segments,
                    key_valid=block_key_valid,
                )

        if sp_world_size > 1:
            joint_hidden_states = gather_and_unpad(joint_hidden_states, sp_pad, dim=1)

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        output = self.proj_out(joint_hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
