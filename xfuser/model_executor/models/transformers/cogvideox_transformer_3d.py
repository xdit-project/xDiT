from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed, CogVideoXPatchEmbed

from diffusers.models import CogVideoXTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.model_executor.models import xFuserModelBaseWrapper
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
    get_cfg_group,
    get_sp_group,
    get_runtime_state, 
    initialize_runtime_state
)

from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)

# class CogVideoXBlock(nn.Module):
#     r"""
#     Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

#     Parameters:
#         dim (`int`):
#             The number of channels in the input and output.
#         num_attention_heads (`int`):
#             The number of heads to use for multi-head attention.
#         attention_head_dim (`int`):
#             The number of channels in each head.
#         time_embed_dim (`int`):
#             The number of channels in timestep embedding.
#         dropout (`float`, defaults to `0.0`):
#             The dropout probability to use.
#         activation_fn (`str`, defaults to `"gelu-approximate"`):
#             Activation function to be used in feed-forward.
#         attention_bias (`bool`, defaults to `False`):
#             Whether or not to use bias in attention projection layers.
#         qk_norm (`bool`, defaults to `True`):
#             Whether or not to use normalization after query and key projections in Attention.
#         norm_elementwise_affine (`bool`, defaults to `True`):
#             Whether to use learnable elementwise affine parameters for normalization.
#         norm_eps (`float`, defaults to `1e-5`):
#             Epsilon value for normalization layers.
#         final_dropout (`bool` defaults to `False`):
#             Whether to apply a final dropout after the last feed-forward layer.
#         ff_inner_dim (`int`, *optional*, defaults to `None`):
#             Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
#         ff_bias (`bool`, defaults to `True`):
#             Whether or not to use bias in Feed-forward layer.
#         attention_out_bias (`bool`, defaults to `True`):
#             Whether or not to use bias in Attention output projection layer.
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_attention_heads: int,
#         attention_head_dim: int,
#         time_embed_dim: int,
#         dropout: float = 0.0,
#         activation_fn: str = "gelu-approximate",
#         attention_bias: bool = False,
#         qk_norm: bool = True,
#         norm_elementwise_affine: bool = True,
#         norm_eps: float = 1e-5,
#         final_dropout: bool = True,
#         ff_inner_dim: Optional[int] = None,
#         ff_bias: bool = True,
#         attention_out_bias: bool = True,
#     ):
#         super().__init__()

#         # 1. Self Attention
#         self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

#         self.attn1 = Attention(
#             query_dim=dim,
#             dim_head=attention_head_dim,
#             heads=num_attention_heads,
#             qk_norm="layer_norm" if qk_norm else None,
#             eps=1e-6,
#             bias=attention_bias,
#             out_bias=attention_out_bias,
#             processor=CogVideoXAttnProcessor2_0(),
#         )

#         # 2. Feed Forward
#         self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

#         self.ff = FeedForward(
#             dim,
#             dropout=dropout,
#             activation_fn=activation_fn,
#             final_dropout=final_dropout,
#             inner_dim=ff_inner_dim,
#             bias=ff_bias,
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: torch.Tensor,
#         temb: torch.Tensor,
#         image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         text_seq_length = encoder_hidden_states.size(1)

#         # norm & modulate
#         norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
#             hidden_states, encoder_hidden_states, temb
#         )

#         # attention
#         attn_hidden_states, attn_encoder_hidden_states = self.attn1(
#             hidden_states=norm_hidden_states,
#             encoder_hidden_states=norm_encoder_hidden_states,
#             image_rotary_emb=image_rotary_emb,
#         )

#         hidden_states = hidden_states + gate_msa * attn_hidden_states
#         encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

#         # norm & modulate
#         norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
#             hidden_states, encoder_hidden_states, temb
#         )

#         # feed-forward
#         norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
#         ff_output = self.ff(norm_hidden_states)

#         hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
#         encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

#         return hidden_states, encoder_hidden_states


@xFuserTransformerWrappersRegister.register(CogVideoXTransformer3DModel)
class xFuserCogVideoXTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: CogVideoXTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d],
            submodule_name_to_wrap=["attn1"]
        )
    
    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        
        # 3. Position embedding
        seq_length = height * width * num_frames // (self.config.patch_size**2)

        pos_embeds = self.pos_embedding[:, : self.config.max_text_seq_length + seq_length]
        
        hidden_states = hidden_states + pos_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        encoder_hidden_states = hidden_states[:, : self.config.max_text_seq_length]
        hidden_states = hidden_states[:, self.config.max_text_seq_length :]

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                )

        hidden_states = self.norm_final(hidden_states)

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)