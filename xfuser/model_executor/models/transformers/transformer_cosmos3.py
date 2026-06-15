import math
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from typing import Optional

from xfuser.model_executor.layers.usp import USP, attention
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
)


class xFuserCosmos3AttnProcessor:
    """Replaces Cosmos3AttnProcessor with USP-parallel attention for the generation (DM) pathway.

    Mirrors the stock Cosmos3AttnProcessor exactly (same RoPE, same tensor layout)
    but uses USP for distributed attention when SP is active.
    """

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        # Per-pathway projections: [S, H, D]
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # RoPE: same _rotate_half convention as stock processor
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        try:
            sp_world_size = get_sequence_parallel_world_size()
        except AssertionError:
            sp_world_size = 1

        if sp_world_size > 1:
            # gen_seq is pre-chunked (S/P tokens per rank).
            # USP all-to-all redistributes: [S/P, H, D] → [S, H/P, D]
            # so attention sees full gen sequence with fewer heads.
            # und tokens are replicated (not chunked); we pre-concat
            # them to gen KV so they go through the all-to-all too
            # (small overhead, ~50 tokens).

            # Transpose to BHSD for USP: [S, H, D] -> [1, H, S, D]
            q_und_b = q_und.unsqueeze(0).transpose(1, 2)
            k_und_b = k_und.unsqueeze(0).transpose(1, 2)
            v_und_b = v_und.unsqueeze(0).transpose(1, 2)
            q_gen_b = q_gen.unsqueeze(0).transpose(1, 2)
            k_gen_b = k_gen.unsqueeze(0).transpose(1, 2)
            v_gen_b = v_gen.unsqueeze(0).transpose(1, 2)

            # AR pathway: causal self-attention (no SP, und tokens are short)
            causal_out = attention(q_und_b, k_und_b, v_und_b, dropout_p=0.0, is_causal=True)

            # DM pathway: gen_q × [und_kv; gen_kv]
            # Pre-concat und KV to gen KV. USP's all-to-all will
            # redistribute the combined KV (splitting heads, gathering
            # sequence). The output has the same seq dim as gen_q input.
            k_full_b = torch.cat([k_und_b, k_gen_b], dim=2)
            v_full_b = torch.cat([v_und_b, v_gen_b], dim=2)
            full_out = USP(
                q_gen_b, k_full_b, v_full_b,
                dropout_p=0.0, is_causal=False,
            )
            causal_out = causal_out.squeeze(0).transpose(0, 1).flatten(-2, -1)
            full_out = full_out.squeeze(0).transpose(0, 1).flatten(-2, -1)
        else:
            # Non-distributed: use SDPA with GQA head expansion in BHSD layout.
            # Input tensors are [S, H, D]. SDPA on ROCm requires BHSD [B, H, S, D]
            # for cross-attention with different Q/KV sequence lengths.
            num_kv_groups = attn.num_key_value_groups

            k_und_exp = k_und.repeat_interleave(num_kv_groups, dim=1)
            v_und_exp = v_und.repeat_interleave(num_kv_groups, dim=1)
            # [S, H, D] -> [1, H, S, D] (BHSD)
            causal_out = sdpa(
                q_und.unsqueeze(0).transpose(1, 2),
                k_und_exp.unsqueeze(0).transpose(1, 2),
                v_und_exp.unsqueeze(0).transpose(1, 2),
                is_causal=True,
            ).transpose(1, 2).squeeze(0).flatten(-2, -1)

            all_k = torch.cat([k_und, k_gen], dim=0)
            all_v = torch.cat([v_und, v_gen], dim=0)
            all_k_exp = all_k.repeat_interleave(num_kv_groups, dim=1)
            all_v_exp = all_v.repeat_interleave(num_kv_groups, dim=1)
            full_out = sdpa(
                q_gen.unsqueeze(0).transpose(1, 2),
                all_k_exp.unsqueeze(0).transpose(1, 2),
                all_v_exp.unsqueeze(0).transpose(1, 2),
                is_causal=False,
            ).transpose(1, 2).squeeze(0).flatten(-2, -1)

        und_out = attn.to_out(causal_out)
        gen_out = attn.to_add_out(full_out)
        return und_out, gen_out


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _make_xfuser_cosmos3_transformer_wrapper():
    """Lazily create the wrapper class to avoid import errors when diffusers < 0.37.1."""
    from diffusers.models.transformers.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
        Cosmos3PackedMoTAttention,
    )

    class _AutoCastWrapper(torch.nn.Module):
        """Wraps a module to auto-cast inputs to match parameter dtype."""
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, sample, *args, **kwargs):
            param_dtype = next(self.module.parameters()).dtype
            return self.module(sample.to(param_dtype), *args, **kwargs)

        def parameters(self, *args, **kwargs):
            return self.module.parameters(*args, **kwargs)

    class xFuserCosmos3OmniTransformerWrapper(Cosmos3OmniTransformer):

        def _install_xfuser_processors(self):
            for layer in self.layers:
                layer.self_attn.processor = xFuserCosmos3AttnProcessor()

        def _patch_time_embedder_for_fsdp(self):
            """Wrap time_embedder with auto-cast, fixing FSDP dtype mismatch."""
            self.time_embedder = _AutoCastWrapper(self.time_embedder)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = super().from_pretrained(*args, **kwargs)
            model.__class__ = cls
            model._install_xfuser_processors()
            return model

        def _chunk_and_pad_sequence(self, x, sp_rank, sp_size, pad_amount, dim):
            if pad_amount > 0:
                if dim < 0:
                    dim = x.ndim + dim
                pad_shape = list(x.shape)
                pad_shape[dim] = pad_amount
                x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
            return torch.chunk(x, sp_size, dim=dim)[sp_rank]

        def _gather_and_unpad(self, x, pad_amount, dim):
            x = get_sp_group().all_gather(x, dim=dim)
            size = x.size(dim)
            return x.narrow(dim=dim, start=0, length=size - pad_amount)

        def forward(
            self,
            input_ids,
            text_indexes,
            position_ids,
            und_len,
            sequence_length,
            vision_tokens,
            vision_token_shapes,
            vision_sequence_indexes,
            vision_mse_loss_indexes,
            vision_timesteps,
            vision_noisy_frame_indexes,
            sound_tokens=None,
            sound_token_shapes=None,
            sound_sequence_indexes=None,
            sound_mse_loss_indexes=None,
            sound_timesteps=None,
            sound_noisy_frame_indexes=None,
            action_tokens=None,
            action_token_shapes=None,
            action_sequence_indexes=None,
            action_mse_loss_indexes=None,
            action_timesteps=None,
            action_noisy_frame_indexes=None,
            action_domain_ids=None,
        ):
            try:
                get_runtime_state().increment_step_counter()
                sp_rank = get_sequence_parallel_rank()
                sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                sp_rank = 0
                sp_size = 1

            if sp_size <= 1:
                # No sequence parallelism — delegate to parent forward.
                # Our xFuserCosmos3AttnProcessor handles the attention
                # correctly with USP (which falls through to direct attention
                # when sp_size == 1).
                return Cosmos3OmniTransformer.forward(
                    self,
                    input_ids=input_ids,
                    text_indexes=text_indexes,
                    position_ids=position_ids,
                    und_len=und_len,
                    sequence_length=sequence_length,
                    vision_tokens=vision_tokens,
                    vision_token_shapes=vision_token_shapes,
                    vision_sequence_indexes=vision_sequence_indexes,
                    vision_mse_loss_indexes=vision_mse_loss_indexes,
                    vision_timesteps=vision_timesteps,
                    vision_noisy_frame_indexes=vision_noisy_frame_indexes,
                    sound_tokens=sound_tokens,
                    sound_token_shapes=sound_token_shapes,
                    sound_sequence_indexes=sound_sequence_indexes,
                    sound_mse_loss_indexes=sound_mse_loss_indexes,
                    sound_timesteps=sound_timesteps,
                    sound_noisy_frame_indexes=sound_noisy_frame_indexes,
                    action_tokens=action_tokens,
                    action_token_shapes=action_token_shapes,
                    action_sequence_indexes=action_sequence_indexes,
                    action_mse_loss_indexes=action_mse_loss_indexes,
                    action_timesteps=action_timesteps,
                    action_noisy_frame_indexes=action_noisy_frame_indexes,
                    action_domain_ids=action_domain_ids,
                )

            # SP path: chunk gen_seq across ranks so MLPs/norms run on 1/P
            # of the sequence. USP handles the all-to-all inside attention.
            has_sound = sound_tokens is not None and sound_sequence_indexes is not None
            has_action = action_tokens is not None and action_sequence_indexes is not None

            # Replicate parent's setup: embed, patchify, build joint sequence
            packed_text_embedding = self.embed_tokens(input_ids)
            target_dtype = packed_text_embedding.dtype
            hidden_states = packed_text_embedding.new_zeros(
                size=(sequence_length, self.config.hidden_size)
            )
            hidden_states[text_indexes] = packed_text_embedding

            packed_tokens_vision, original_latent_shapes = (
                self._patchify_and_pack_latents(vision_tokens)
            )
            packed_tokens_vision = self.proj_in(packed_tokens_vision)
            timesteps_vision = vision_timesteps * self.config.timestep_scale
            packed_timestep_embeds_vision = self.time_embedder(
                self.time_proj(timesteps_vision)
            ).to(target_dtype)
            packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision_noisy_frame_indexes,
                token_shapes=vision_token_shapes,
            )
            hidden_states[vision_sequence_indexes] = packed_tokens_vision

            if has_sound:
                packed_tokens_sound = self._pack_sound_latents(
                    sound_tokens, sound_token_shapes
                ).to(target_dtype)
                packed_tokens_sound = (
                    self.audio_proj_in(packed_tokens_sound)
                    + self.audio_modality_embed
                )
                timesteps_sound = sound_timesteps * self.config.timestep_scale
                packed_timestep_embeds_sound = self.time_embedder(
                    self.time_proj(timesteps_sound)
                ).to(target_dtype)
                packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_tokens_sound,
                    packed_timestep_embeds=packed_timestep_embeds_sound,
                    noisy_frame_indexes=sound_noisy_frame_indexes,
                    token_shapes=sound_token_shapes,
                )
                hidden_states[sound_sequence_indexes] = packed_tokens_sound

            if has_action:
                packed_tokens_action, per_token_domain_ids = (
                    self._pack_action_latents(
                        action_tokens, action_token_shapes, action_domain_ids
                    )
                )
                packed_tokens_action = packed_tokens_action.to(target_dtype)
                per_token_domain_ids = per_token_domain_ids.to(
                    device=packed_tokens_action.device
                )
                packed_tokens_action = self.action_proj_in(
                    packed_tokens_action, per_token_domain_ids
                )
                packed_tokens_action = (
                    packed_tokens_action + self.action_modality_embed
                )
                if action_mse_loss_indexes.numel() > 0:
                    timesteps_action = (
                        action_timesteps * self.config.timestep_scale
                    )
                    packed_timestep_embeds_action = self.time_embedder(
                        self.time_proj(timesteps_action)
                    ).to(target_dtype)
                    packed_tokens_action = (
                        self._apply_timestep_embeds_to_noisy_tokens(
                            packed_tokens=packed_tokens_action,
                            packed_timestep_embeds=packed_timestep_embeds_action,
                            noisy_frame_indexes=action_noisy_frame_indexes,
                            token_shapes=action_token_shapes,
                        )
                    )
                hidden_states[action_sequence_indexes] = packed_tokens_action

            # Rotary embeddings for the full joint sequence
            cos, sin = self.rotary_emb(
                position_ids=(
                    position_ids.unsqueeze(0)
                    if position_ids.ndim == 1
                    else position_ids.unsqueeze(1)
                ),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)

            # Split into und (AR, replicated) and gen (DM, to be chunked)
            und_seq = hidden_states[:und_len]
            gen_seq = hidden_states[und_len:]

            und_cos = cos[:und_len]
            und_sin = sin[:und_len]
            gen_cos = cos[und_len:]
            gen_sin = sin[und_len:]

            # Chunk gen_seq and gen rotary embeddings across SP ranks
            gen_len = gen_seq.shape[0]
            pad_amount = (sp_size - (gen_len % sp_size)) % sp_size
            gen_seq = self._chunk_and_pad_sequence(
                gen_seq, sp_rank, sp_size, pad_amount, dim=0
            )
            gen_cos = self._chunk_and_pad_sequence(
                gen_cos, sp_rank, sp_size, pad_amount, dim=0
            )
            gen_sin = self._chunk_and_pad_sequence(
                gen_sin, sp_rank, sp_size, pad_amount, dim=0
            )

            rotary_emb_chunked = (und_cos, und_sin, gen_cos, gen_sin)

            # Run decoder layers on chunked gen_seq
            for decoder_layer in self.layers:
                und_seq, gen_seq = decoder_layer(
                    und_seq, gen_seq, rotary_emb_chunked
                )

            # Gather gen_seq back to full length
            gen_seq = self._gather_and_unpad(gen_seq, pad_amount, dim=0)

            # Final norms and output projections
            und_out = self.norm(und_seq)
            gen_out = self.norm_moe_gen(gen_seq)
            last_hidden_state = torch.cat([und_out, gen_out], dim=0)

            preds_vision_packed = self.proj_out(
                last_hidden_state[vision_mse_loss_indexes]
            )
            preds_vision = self._unpatchify_and_unpack_latents(
                preds_vision_packed,
                token_shapes_vision=vision_token_shapes,
                noisy_frame_indexes_vision=vision_noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )

            preds_sound = None
            if has_sound:
                preds_sound_packed = self.audio_proj_out(
                    last_hidden_state[sound_mse_loss_indexes]
                )
                preds_sound = self._unpack_sound_latents(
                    preds_sound_packed, sound_token_shapes,
                    sound_noisy_frame_indexes,
                )

            preds_action = None
            if has_action:
                per_noisy_domain_ids = [
                    domain_id.reshape(1).expand(len(noisy_idxs))
                    for domain_id, noisy_idxs in zip(
                        action_domain_ids, action_noisy_frame_indexes
                    )
                ]
                per_noisy_domain_ids = torch.cat(
                    per_noisy_domain_ids, dim=0
                ).to(device=last_hidden_state.device)
                preds_action_packed = self.action_proj_out(
                    last_hidden_state[action_mse_loss_indexes],
                    per_noisy_domain_ids,
                )
                preds_action = self._unpack_action_latents(
                    preds_action_packed, action_token_shapes,
                    action_noisy_frame_indexes,
                )

            return preds_vision, preds_sound, preds_action

    return xFuserCosmos3OmniTransformerWrapper


# Lazy singleton
_wrapper_cls = None

def get_cosmos3_transformer_wrapper_class():
    global _wrapper_cls
    if _wrapper_cls is None:
        _wrapper_cls = _make_xfuser_cosmos3_transformer_wrapper()
    return _wrapper_cls
