# Example for parallelize new models with USP
# run with 
#     torchrun --nproc_per_node=<ulysses_degree x ring-degree> \
#          adding_cogvideox.py <cogvideox-checkpoint-path> \
#          <ulysses_degree> <ring-degree>
# E.g.,
#     torchrun --nproc_per_node=2 \
#          adding_cogvideox.py <cogvideox-checkpoint-path> \
#          2 1
import sys
import functools
from typing import List, Optional, Tuple, Union

import time
import torch

from diffusers import DiffusionPipeline, CogVideoXPipeline

import torch.distributed as dist
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    get_world_group,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

from diffusers.utils import export_to_video

from diffusers.models.attention import Attention
from diffusers.models.attention_processor import (
    CogVideoXAttnProcessor2_0
)
from diffusers.models.embeddings import apply_rotary_emb
from xfuser.model_executor.layers.usp import USP

class xDiTCogVideoXAttnProcessor(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        #! ---------------------------------------- ATTENTION ----------------------------------------
        hidden_states = USP(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, latent_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        rope_h, rope_w = hidden_states.shape[-2] // 2, hidden_states.shape[-1] // 2
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            dim_thw = freqs_cos.shape[-1]
            freqs_cos = freqs_cos.reshape(-1, rope_h, rope_w, dim_thw)
            freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(), dim=-3)[get_sequence_parallel_rank()]
            freqs_cos = freqs_cos.reshape(-1, dim_thw)
            
            freqs_sin = freqs_sin.reshape(-1, rope_h, rope_w, dim_thw)
            freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(), dim=-3)[get_sequence_parallel_rank()]
            freqs_sin = freqs_sin.reshape(-1, dim_thw)
            
            image_rotary_emb = (freqs_cos, freqs_sin)
        
        for block in transformer.transformer_blocks:
            block.attn1.processor = xDiTCogVideoXAttnProcessor()
        
        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            ofs=ofs,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
    
    original_patch_embed_forward = transformer.patch_embed.forward

    @functools.wraps(transformer.patch_embed.__class__.forward)
    def new_patch_embed(
        self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
    ):
        text_embeds = get_sp_group().all_gather(text_embeds.contiguous(), dim=-2)
        image_embeds = get_sp_group().all_gather(image_embeds.contiguous(), dim=-2)
        batch, embed_height, embed_width = image_embeds.shape[0], image_embeds.shape[-2] // 2, image_embeds.shape[-1] // 2
        text_len = text_embeds.shape[-2]
        
        output = original_patch_embed_forward(text_embeds, image_embeds)

        text_embeds = output[:,:text_len,:]
        image_embeds = output[:,text_len:,:].reshape(batch, -1, embed_height, embed_width, output.shape[-1])

        text_embeds = torch.chunk(text_embeds, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        image_embeds = torch.chunk(image_embeds, get_sequence_parallel_world_size(),dim=-3)[get_sequence_parallel_rank()]
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        return torch.cat([text_embeds, image_embeds], dim=1)

    new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
    transformer.patch_embed.forward = new_patch_embed


if __name__ == "__main__":
    dist.init_process_group("nccl")
    init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size()
    )
    initialize_model_parallel(
        sequence_parallel_degree=int(sys.argv[2]) * int(sys.argv[3]),
        ring_degree=int(sys.argv[2]),
        ulysses_degree=int(sys.argv[3]),
        classifier_free_guidance_degree=2,
    )
    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=sys.argv[1],
        torch_dtype=torch.bfloat16,
    )
    local_rank = get_world_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    pipe = pipe.to(device)

    pipe.vae.enable_tiling()

    parallelize_transformer(pipe)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        num_frames=9,
        prompt="A little girl is riding a bicycle at high speed. Focused, detailed, realistic.",
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time

    if local_rank == 0:
        export_to_video(output, "output.mp4", fps=8)
        print(f"epoch time: {elapsed_time:.2f} sec")

    dist.destroy_process_group()
