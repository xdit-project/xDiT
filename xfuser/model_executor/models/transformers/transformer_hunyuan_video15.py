import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

class xFuserHunyuanVideo15AttnProcessor:

    def __init__(self, attention_kwargs: Optional[Dict[str, Any]] = None):
        self.attention_kwargs = attention_kwargs

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 4. Encoder condition QKV projection and normalization
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if encoder_hidden_states is not None:
            num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
            num_query_tokens = query.shape[1] - num_encoder_hidden_states_tokens
        else:
            num_encoder_hidden_states_tokens = (
                get_runtime_state().max_condition_sequence_length
            )
            num_query_tokens = query.shape[1] - num_encoder_hidden_states_tokens
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        #! ---------------------------------------- ATTENTION ---------------------------------------- 
        if self.attention_kwargs is not None:
            self.attention_kwargs["encoder_sequence_length"] = num_encoder_hidden_states_tokens
            self.attention_kwargs["text_mask"] = attention_mask
        if get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=self.attention_kwargs)
            else:
                query, encoder_query = query.split(
                    [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                )
                key, encoder_key = key.split(
                    [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                )
                value, encoder_value = value.split(
                    [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                )
            
                hidden_states = USP(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                    joint_query=encoder_query,
                    joint_key=encoder_key,
                    joint_value=encoder_value,
                    joint_strategy="rear",
                    attention_kwargs=self.attention_kwargs,
                )
        else:
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=self.attention_kwargs)
        
        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)
  
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

class xFuserHunyuanVideo15Transformer3DWrapper(HunyuanVideo15Transformer3DModel):

    def __init__(
        self,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int|list = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        # YiYi Notes: config based on target_size_config https://github.com/yiyixuxu/hy15/blob/main/hyvideo/pipelines/hunyuan_video_pipeline.py#L205
        target_size: int = 640,  # did not name sample_size since it is in pixel spaces
        task_type: str = "i2v",
        use_meanflow: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(patch_size, list):
            patch_size = 1
        if qk_norm == "rms":
            qk_norm = "rms_norm"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            qk_norm=qk_norm,
            text_embed_dim=text_embed_dim,
            text_embed_2_dim=text_embed_2_dim,
            image_embed_dim=image_embed_dim,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            target_size=target_size,
            task_type=task_type,
            use_meanflow=use_meanflow,
        )
        self.attention_kwargs = attention_kwargs
        for block in self.transformer_blocks:
            block.attn.processor = xFuserHunyuanVideo15AttnProcessor(attention_kwargs=attention_kwargs)

    def _chunk_and_pad_sequence(self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x,
                        torch.zeros(
                            pad_shape,
                            dtype=x.dtype,
                            device=x.device,
                        )], dim=dim)
        x = torch.chunk(x,
                        sp_world_size,
                        dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:


        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if self.attention_kwargs is not None:
            self.attention_kwargs["thw"] = (post_patch_num_frames, post_patch_height, post_patch_width) # Should modify reference in xFuserHunyuanVideo15AttnProcessor.

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)


        # qwen text embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # byt5 text embedding
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # image embed
        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # reorder and combine text tokens: combine valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],  # valid image
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        image[~image_mask],  # invalid image
                        torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                        torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                    ],
                    dim=0,
                )
            )

            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        # sequence parallel
        hidden_states_pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = self._chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=1)
        cos, sin = image_rotary_emb
        cos = self._chunk_and_pad_sequence(cos, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=0)
        sin = self._chunk_and_pad_sequence(sin, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=0)
        image_rotary_emb = (cos, sin)

        any_valid = encoder_attention_mask.to(torch.bool).any(dim=0)
        encoder_hidden_states = encoder_hidden_states[:, any_valid, :]
        encoder_attention_mask = encoder_attention_mask.to(torch.bool)[:, any_valid]
    
        if encoder_hidden_states.shape[1] % sp_world_size != 0:
            if self.attention_kwargs is not None:
                # SSTA requires symmetric [image, text] layout in Q/K/V.
                # Pad text to be divisible by sp_world_size so it can be chunked,
                # ensuring split_text_embed_in_sp=True and avoiding the asymmetric
                # joint_strategy path which SSTA cannot handle.
                enc_rem = encoder_hidden_states.shape[1] % sp_world_size
                enc_pad = sp_world_size - enc_rem
                encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, enc_pad))
                encoder_attention_mask = F.pad(encoder_attention_mask, (0, enc_pad), value=False)
                get_runtime_state().split_text_embed_in_sp = True
                encoder_hidden_states = torch.chunk(
                    encoder_hidden_states,
                    sp_world_size,
                    dim=1)[sp_world_rank]
            else:
                get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states,
                sp_world_size,
                dim=1)[sp_world_rank]

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = self._gather_and_unpad(hidden_states, hidden_states_pad_amount, dim=1)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

