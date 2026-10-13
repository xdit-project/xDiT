"""xDiT wrapper for the LTX-Video 0.9.x transformer (``LTXVideoTransformer3DModel``).

Parallel layout of one forward pass:

* CFG parallel: the pipelines run classifier-free guidance as one batch of
  ``[negative, positive]``. Each CFG rank keeps its slice of every batch-shaped
  input and the outputs are gathered back in rank order at the end.
* Sequence parallel (Ulysses and/or ring): the video tokens are split along the
  sequence. RoPE is built on the full sequence and then split, so every rank sees
  exactly the positions it would see unsplit; a per-token timestep (image-to-video
  conditioning) is split the same way. The text sequence is never split: cross
  attention lets each rank's video queries attend to the full prompt.

diffusers' own ``_cp_plan`` for this model also splits the text sequence; it is
deliberately not used, and its ``parallel_config`` hooks are never activated.
"""

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_ltx import (
    LTXVideoTransformer3DModel,
    apply_rotary_emb,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    model_parallel_is_initialized,
)
from xfuser.model_executor.layers.usp import USP, attention
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper

__all__ = ["xFuserLTXVideoAttnProcessor", "xFuserLTXVideoTransformer3DWrapper"]


@dataclass(frozen=True)
class _ParallelLayout:
    cfg_world_size: int = 1
    cfg_rank: int = 0
    sp_world_size: int = 1
    sp_rank: int = 0
    ulysses_world_size: int = 1
    ring_world_size: int = 1


def _parallel_layout() -> _ParallelLayout:
    """The CFG and sequence-parallel layout, or a single rank before xDiT is initialized."""
    if not model_parallel_is_initialized():
        return _ParallelLayout()
    return _ParallelLayout(
        cfg_world_size=get_classifier_free_guidance_world_size(),
        cfg_rank=get_classifier_free_guidance_rank(),
        sp_world_size=get_sequence_parallel_world_size(),
        sp_rank=get_sequence_parallel_rank(),
        ulysses_world_size=get_ulysses_parallel_world_size(),
        ring_world_size=get_ring_parallel_world_size(),
    )


@dataclass(frozen=True)
class _TextKeys:
    """Which text tokens each sample's cross attention may attend to.

    The pipelines pad prompts to a fixed length and pass a [batch, text_len]
    mask, which diffusers turns into a -10000 additive bias. Gathering the valid
    keys instead is the same computation, and unlike a mask it is honoured by
    every attention backend, including those that take no mask at all.
    """

    shared: torch.Tensor | None  # one index set used by every sample
    per_sample: tuple[torch.Tensor, ...] | None  # one index set per sample

    @classmethod
    def from_mask(cls, mask: torch.Tensor | None) -> "_TextKeys | None":
        if mask is None:
            return None
        if mask.ndim != 2:
            raise ValueError(
                f"LTX-Video expects encoder_attention_mask of shape [batch, text_len], got {tuple(mask.shape)}."
            )
        valid = mask.to(torch.bool)
        # A sample with no valid key gets an all-equal bias in diffusers, which is
        # attention over every key.
        valid = valid | ~valid.any(dim=1, keepdim=True)
        if bool(valid.all()):
            return None
        if bool((valid == valid[:1]).all()):
            return cls(shared=valid[0].nonzero().flatten(), per_sample=None)
        return cls(
            shared=None,
            per_sample=tuple(row.nonzero().flatten() for row in valid),
        )


def _cross_attention(query, key, value, text_keys: _TextKeys | None) -> torch.Tensor:
    if text_keys is None:
        return attention(query, key, value)
    if text_keys.shared is not None:
        index = text_keys.shared
        return attention(query, key.index_select(2, index), value.index_select(2, index))
    return torch.cat(
        [
            attention(
                query[i : i + 1],
                key[i : i + 1].index_select(2, index),
                value[i : i + 1].index_select(2, index),
            )
            for i, index in enumerate(text_keys.per_sample)
        ],
        dim=0,
    )


class xFuserLTXVideoAttnProcessor:
    """Replacement for diffusers' ``LTXVideoAttnProcessor``.

    QK RMSNorm runs across all heads and RoPE acts on the flattened
    [batch, tokens, inner_dim] layout, both per token, so a sequence shard needs
    no communication before attention. Self attention (``sequence_parallel``)
    goes through USP; cross attention runs locally against the full text.

    The wrapper shares one instance of each kind across all blocks and sets
    ``use_usp`` and ``attention_kwargs`` at the start of every forward.
    """

    def __init__(self, sequence_parallel: bool):
        self.sequence_parallel = sequence_parallel
        self.use_usp = False
        self.attention_kwargs: dict | None = None

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: _TextKeys | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        kv_input = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(kv_input))
        value = attn.to_v(kv_input)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if not self.sequence_parallel:
            hidden_states = _cross_attention(query, key, value, attention_mask)
        elif self.use_usp:
            hidden_states = USP(query, key, value, attn_layer=attn, attention_kwargs=self.attention_kwargs)
        else:
            hidden_states = attention(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserLTXVideoTransformer3DWrapper(LTXVideoTransformer3DModel):
    """``LTXVideoTransformer3DModel`` with CFG and sequence parallelism.

    The constructor is inherited unchanged so diffusers' config handling sees the
    parent signature; ``from_config`` (also reached through ``from_pretrained``)
    installs the xDiT attention processors.
    """

    # Same sequence-parallel split as Wan.
    _chunk_and_pad_sequence = xFuserWanTransformer3DWrapper._chunk_and_pad_sequence
    _gather_and_unpad = xFuserWanTransformer3DWrapper._gather_and_unpad

    def install_xdit_attention_processors(self):
        self._xdit_self_attn_processor = xFuserLTXVideoAttnProcessor(sequence_parallel=True)
        self._xdit_cross_attn_processor = xFuserLTXVideoAttnProcessor(sequence_parallel=False)
        for block in self.transformer_blocks:
            block.attn1.processor = self._xdit_self_attn_processor
            block.attn2.processor = self._xdit_cross_attn_processor
        return self

    @classmethod
    def from_config(cls, config, return_unused_kwargs=False, **kwargs):
        result = super().from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        model = result[0] if return_unused_kwargs else result
        model.install_xdit_attention_processors()
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        rope_interpolation_scale: tuple[float, float, float] | torch.Tensor | None = None,
        video_coords: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if "_xdit_self_attn_processor" not in self.__dict__:
            # Built through __init__ rather than from_config/from_pretrained.
            self.install_xdit_attention_processors()

        layout = _parallel_layout()
        if layout.sp_world_size != layout.ulysses_world_size * layout.ring_world_size:
            # Without yunchang's long-context attention (it needs CUDA or NPU) the
            # sequence-parallel group reports Ulysses and ring degree 1, so USP would
            # attend only within each rank's shard.
            raise RuntimeError(
                f"LTX-Video sequence parallelism of degree {layout.sp_world_size} needs "
                "Ulysses or ring attention, which is unavailable on this host "
                f"(ulysses_degree={layout.ulysses_world_size}, ring_degree={layout.ring_world_size})."
            )
        if self.config.num_attention_heads % layout.ulysses_world_size != 0:
            raise ValueError(
                f"ulysses_degree {layout.ulysses_world_size} must divide the "
                f"{self.config.num_attention_heads} attention heads of LTX-Video."
            )

        # 1. CFG parallel: keep this rank's slice of every batch-shaped input.
        if layout.cfg_world_size > 1:
            full_batch = hidden_states.shape[0]
            if full_batch % layout.cfg_world_size != 0:
                raise ValueError(
                    f"CFG parallel degree {layout.cfg_world_size} must divide the batch "
                    f"size {full_batch}; it needs guidance_scale > 1 so that the pipeline "
                    "batches the negative and positive prompts together."
                )

            def take(x):
                if isinstance(x, torch.Tensor) and x.ndim > 0 and x.shape[0] == full_batch:
                    return x.chunk(layout.cfg_world_size, dim=0)[layout.cfg_rank]
                return x

            hidden_states = take(hidden_states)
            encoder_hidden_states = take(encoder_hidden_states)
            encoder_attention_mask = take(encoder_attention_mask)
            timestep = take(timestep)
            video_coords = take(video_coords)

        batch_size, seq_len = hidden_states.shape[:2]
        pad_amount = (-seq_len) % layout.sp_world_size
        if pad_amount and layout.ring_world_size > 1:
            raise ValueError(
                f"LTX-Video with ring_degree > 1 needs the {seq_len} video tokens to be "
                f"divisible by the sequence-parallel degree {layout.sp_world_size}; "
                "adjust height, width or num_frames, or use Ulysses only."
            )

        # 2. RoPE on the full sequence, then split like the tokens.
        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        if layout.sp_world_size > 1:

            def shard(x):
                return self._chunk_and_pad_sequence(x, layout.sp_rank, layout.sp_world_size, pad_amount, dim=1)

            hidden_states = shard(hidden_states)
            image_rotary_emb = tuple(shard(x) for x in image_rotary_emb)
            # Image-to-video conditions the first frame with a per-token timestep.
            if timestep.ndim == 2 and timestep.shape[1] == seq_len:
                timestep = shard(timestep)

        self._xdit_self_attn_processor.use_usp = layout.sp_world_size > 1
        # Padded keys must not be attended to; after the Ulysses all-to-all the
        # valid ones are the leading seq_len.
        self._xdit_self_attn_processor.attention_kwargs = {"valid_kv_len": seq_len} if pad_amount else None
        text_keys = _TextKeys.from_mask(encoder_attention_mask)

        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    text_keys,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=text_keys,
                )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if layout.sp_world_size > 1:
            output = self._gather_and_unpad(output, pad_amount, dim=1)
        if layout.cfg_world_size > 1:
            output = get_cfg_group().all_gather(output.contiguous(), dim=0)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
