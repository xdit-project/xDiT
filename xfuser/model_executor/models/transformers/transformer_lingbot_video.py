import torch
from typing import Optional, Union, Dict, Any

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
)


def get_lingbot_video_classes():
    from lingbot_video.transformer_lingbot_video import (
        LingBotVideoTransformer3DModel,
        apply_rotary_emb,
        make_joint_position_ids,
        _cat_interleave,
    )
    return (
        LingBotVideoTransformer3DModel,
        apply_rotary_emb,
        make_joint_position_ids,
        _cat_interleave,
    )


def _usp_attention_forward(attn_module, x, rotary_emb, attention_mask=None, packed_indices=None, parallel_config=None):
    """Replacement forward for LingBotVideoAttention that uses USP for distributed attention.

    For single GPU (sp_world_size=1), USP routes through the configured attention backend
    (AITER, FlashAttention, etc.) rather than the default SDPA dispatch.
    """
    B, S, _ = x.shape
    q = attn_module.to_q(x).unflatten(2, (attn_module.num_heads, attn_module.head_dim))
    k = attn_module.to_k(x).unflatten(2, (attn_module.num_heads, attn_module.head_dim))
    v = attn_module.to_v(x).unflatten(2, (attn_module.num_heads, attn_module.head_dim))

    _, apply_rotary_emb_fn, _, _ = get_lingbot_video_classes()
    q = apply_rotary_emb_fn(attn_module.norm_q(q), rotary_emb)
    k = apply_rotary_emb_fn(attn_module.norm_k(k), rotary_emb)

    # USP expects (B, H, S, D), LingBot attention uses (B, S, H, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = USP(q, k, v)
    out = out.transpose(1, 2)  # back to (B, S, H, D)
    return attn_module.to_out(out.flatten(2, 3).type_as(x))


def _patch_block_bulk_dtype(block):
    """Patch LingBotVideoBlock.forward to use cached dtype instead of .weight.dtype.

    After FP4/FP8 quantization, attn.to_q no longer has .weight, so the stock
    forward's `self.attn.to_q.weight.dtype` crashes. This patches the block to
    use the pre-cached dtype instead.
    """
    original_forward = block.forward

    def patched_forward(x, temb6, rotary_emb, attention_mask=None, moe_padding_mask=None, packed_indices=None, parallel_config=None):
        import torch.nn.functional as F
        from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                f"LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(x.shape)}."
            )
        mod = temb6.view(x.shape[0], x.shape[1], -1) + block.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        bulk_dtype = getattr(block, "_cached_bulk_dtype", torch.bfloat16)
        attn_in = (block.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = block.attn(
            attn_in, rotary_emb, attention_mask,
            packed_indices=packed_indices, parallel_config=parallel_config,
        )
        x = x + (gate_msa * block.norm_post_attn(attn_out)).to(x.dtype)

        ffn_in = (block.norm2(x) * scale_mlp + shift_mlp).to(bulk_dtype)
        if isinstance(block.ffn, LingBotVideoSparseMoeBlock):
            ffn_out = block.ffn(ffn_in, padding_mask=moe_padding_mask)
        else:
            ffn_out = block.ffn(ffn_in)
        ffn_normed = block.norm_post_ffn(ffn_out)
        x = x + (gate_mlp * ffn_normed).to(x.dtype)
        return x

    block.forward = patched_forward


def _reorder_tokens_compile_friendly(tokens, top_scores, top_indices, num_experts):
    """Compile-friendly _reorder_tokens that avoids dynamic-shape torch.where.

    Assumes all top-k slots are active (scores != 0), which holds for standard
    top-k routing where exactly k experts are selected per token.
    """
    num_tokens = tokens.shape[0]
    top_k = top_indices.shape[1]
    flat_scores = top_scores.reshape(-1)
    flat_indices = top_indices.reshape(-1)

    counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
    counts.scatter_add_(0, flat_indices.long(), torch.ones_like(flat_indices, dtype=torch.int64))

    sort_order = torch.argsort(flat_indices, stable=True)
    sorted_positions = sort_order
    sorted_scores = flat_scores[sort_order]
    original_token_idx = sort_order // top_k
    permuted_tokens = tokens[original_token_idx]
    return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k


def _rope_forward_no_rebuild(rope_module, position_ids):
    """RoPE forward that skips the lazy table rebuild check.

    The table must be pre-built via a warmup call before torch.compile.
    Eliminates the data-dependent `max_vals >= axes_lens` graph break.
    """
    if rope_module.freqs_cis is None:
        return rope_module._original_forward(position_ids)
    device = position_ids.device
    if rope_module.freqs_cis[0].device != device:
        rope_module.freqs_cis = [fc.to(device) for fc in rope_module.freqs_cis]
    return torch.cat(
        [rope_module.freqs_cis[i][position_ids[:, i]] for i in range(len(rope_module.axes_dims))],
        dim=-1,
    )


class xFuserLingBotVideoTransformer3DWrapper:
    """Wrapper that adds USP (Ulysses Sequence Parallelism) to LingBotVideoTransformer3DModel.

    Uses the same chunk-blocks-gather pattern as the Wan 2.1 wrapper but applied to
    the joint [video; text] sequence (since LingBot uses self-attention only, no cross-attention).
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        LingBotVideoTransformer3DModel = get_lingbot_video_classes()[0]
        model = LingBotVideoTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        model.__class__ = type(
            "xFuserLingBotVideoTransformer3DWrapper",
            (xFuserLingBotVideoTransformer3DWrapper, LingBotVideoTransformer3DModel),
            {},
        )
        model._install_usp_attention = cls._install_usp_attention.__get__(model)
        model.prepare_for_compile = cls.prepare_for_compile.__get__(model)
        model._install_usp_attention()
        model._usp_attention_installed = True
        return model

    def prepare_for_compile(self, height, width, num_frames, vae_scale_temporal=4, vae_scale_spatial=8):
        """Pre-build RoPE table and patch forward to skip the rebuild check.

        Must be called before torch.compile to eliminate graph breaks from
        the data-dependent rebuild logic in LingBotVideoRotaryEmbedding.
        """
        _, _, make_joint_position_ids_fn, _ = get_lingbot_video_classes()
        pF, pH, pW = self.config.patch_size
        gt = ((num_frames - 1) // vae_scale_temporal + 1) // pF
        gh = (height // vae_scale_spatial) // pH
        gw = (width // vae_scale_spatial) // pW
        max_text_len = 512
        device = next(self.parameters()).device
        pos_ids = make_joint_position_ids_fn(max_text_len, gt, gh, gw, device)
        self.rope(pos_ids)
        self.rope._original_forward = self.rope.forward
        self.rope.forward = lambda pos_ids, _r=self.rope: _rope_forward_no_rebuild(_r, pos_ids)
        # Patch MoE _reorder_tokens to avoid dynamic-shape torch.where graph breaks
        from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock
        for block in self.blocks:
            if isinstance(block.ffn, LingBotVideoSparseMoeBlock):
                block.ffn._reorder_tokens = staticmethod(_reorder_tokens_compile_friendly)
        self._compile_friendly_forward = True

    def cache_expert_weights(self):
        """Pre-transpose expert weights to eliminate per-call .bfloat16().transpose() copies.

        The stock _run_grouped_experts calls w1.bfloat16().transpose(-2,-1) on every
        forward — 6 weight copies per block × 48 layers × 48 steps = 5.6TB of memcpy.
        """
        from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

        for block in self.blocks:
            if not isinstance(block.ffn, LingBotVideoSparseMoeBlock):
                continue
            experts = block.ffn.experts
            # Pre-transpose: [E, I, H] -> [E, H, I] (contiguous), replace originals
            w1T = experts.w1.data.bfloat16().transpose(-2, -1).contiguous()
            w3T = experts.w3.data.bfloat16().transpose(-2, -1).contiguous()
            w2T = experts.w2.data.bfloat16().transpose(-2, -1).contiguous()
            # Replace original weights with transposed versions to free memory
            experts.w1 = torch.nn.Parameter(w1T, requires_grad=False)
            experts.w3 = torch.nn.Parameter(w3T, requires_grad=False)
            experts.w2 = torch.nn.Parameter(w2T, requires_grad=False)

            # Patch _run_grouped_experts to skip .bfloat16().transpose() since weights are pre-transposed
            def _make_fast(moe_block):
                e = moe_block.experts
                def _fast(self_moe, tokens, counts):
                    import torch.nn.functional as F
                    input_shape, padded_tokens, permuted_indices, aligned_counts = self_moe._pad_grouped_tokens(tokens, counts)
                    offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
                    h = F.silu(torch._grouped_mm(padded_tokens.bfloat16(), e.w1, offs=offsets))
                    h = h * torch._grouped_mm(padded_tokens.bfloat16(), e.w3, offs=offsets)
                    out = torch._grouped_mm(h, e.w2, offs=offsets).type_as(padded_tokens)
                    return self_moe._unpad_grouped_tokens(out, input_shape, permuted_indices)
                return _fast
            block.ffn._run_grouped_experts = _make_fast(block.ffn).__get__(block.ffn)

    def _install_usp_attention(self):
        for block in self.blocks:
            block.attn._original_forward = block.attn.forward
            block.attn.forward = lambda *args, _attn=block.attn, **kw: _usp_attention_forward(_attn, *args, **kw)

    def _chunk_and_pad_sequence(self, x, sp_world_rank, sp_world_size, pad_amount, dim):
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
        x = torch.chunk(x, sp_world_size, dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x, pad_amount, dim):
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        sp_world_size = get_sequence_parallel_world_size()
        get_runtime_state().increment_step_counter()

        if sp_world_size <= 1:
            LingBotVideoTransformer3DModel = get_lingbot_video_classes()[0]
            return LingBotVideoTransformer3DModel.forward(
                self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, return_dict,
            )

        _, _, make_joint_position_ids_fn, _ = get_lingbot_video_classes()

        sp_world_rank = get_sequence_parallel_rank()

        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.config.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device

        # For compile-friendliness, use L directly as text_len (pipeline already
        # strips padding for B=1). Avoids .tolist() / .item() graph breaks.
        text_lens_list = [L] * B

        # Patchify
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, n_video, pF * pH * pW * C)
        x = self.patch_embedder(patch_tokens)

        # Text embedding and concatenation
        text = self.text_embedder(encoder_hidden_states)
        joint = torch.cat([x, text], dim=1)  # [video; text]
        joint_seq_len = joint.shape[1]

        # RoPE
        pos_ids = make_joint_position_ids_fn(L, gt, gh, gw, device)
        rotary = self.rope(pos_ids).unsqueeze(0)
        if B > 1:
            rotary = rotary.expand(B, -1, -1)

        # SP chunking — chunk BEFORE computing temb6 to avoid full-seq allocations
        pad_amount = (sp_world_size - (joint_seq_len % sp_world_size)) % sp_world_size
        joint = self._chunk_and_pad_sequence(joint, sp_world_rank, sp_world_size, pad_amount, dim=1)
        rotary = self._chunk_and_pad_sequence(rotary, sp_world_rank, sp_world_size, pad_amount, dim=1)
        local_seq_len = joint.shape[1]
        del x, text, patch_tokens

        # Timestep embedding — compute t_emb once, expand to local chunk only
        timestep_for_embed = timestep.float()
        timestep_proj = self.time_proj(timestep_for_embed)
        t_emb = self.time_embedder(timestep_proj)
        temb_input = t_emb.unsqueeze(1).expand(B, local_seq_len, -1)
        temb6 = self.time_modulation(temb_input.reshape(B * local_seq_len, -1))
        temb6 = temb6.reshape(B, local_seq_len, -1)

        temb6_flat = temb6.reshape(temb6.shape[0] * temb6.shape[1], -1)

        # Transformer blocks
        for block in self.blocks:
            joint = block(joint, temb6_flat, rotary, None, None)

        # Output norm + projection
        final_mod = self.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))

        # Gather across SP ranks
        if sp_world_size > 1:
            projected = self._gather_and_unpad(projected, pad_amount, dim=1)

        # Extract video tokens only
        x_out = projected[:, :n_video]

        # Unpatchify
        Cout = self.config.out_channels
        x_out = x_out.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x_out = x_out.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)

        if not return_dict:
            return (x_out,)
        return Transformer2DModelOutput(sample=x_out)
