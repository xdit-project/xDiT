import functools
from typing import List, Optional, Tuple, Union

import logging
import time
import torch

from diffusers import DiffusionPipeline, CogVideoXPipeline

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from diffusers.utils import export_to_video

from xfuser.model_executor.layers.attention_processor import xFuserCogVideoXAttnProcessor2_0

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
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True
        
        if self.config.patch_size_t is None:
            temporal_size = hidden_states.shape[1]
        else:
            temporal_size = hidden_states.shape[1] // self.config.patch_size_t
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb

            def get_rotary_emb_chunk(freqs):
                dim_thw = freqs.shape[-1]
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                freqs = torch.chunk(freqs, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
                freqs = freqs.reshape(-1, dim_thw)
                return freqs

            freqs_cos = get_rotary_emb_chunk(freqs_cos)
            freqs_sin = get_rotary_emb_chunk(freqs_sin)
            image_rotary_emb = (freqs_cos, freqs_sin)
        
        for block in transformer.transformer_blocks:
            block.attn1.processor = xFuserCogVideoXAttnProcessor2_0()
        
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
        batch, num_frames, channels, height, width = image_embeds.shape
        text_len = text_embeds.shape[-2]
        
        output = original_patch_embed_forward(text_embeds, image_embeds)

        text_embeds = output[:,:text_len,:]
        if self.patch_size_t is None:
            image_embeds = output[:,text_len:,:].reshape(batch, num_frames, -1, output.shape[-1])
        else:
            image_embeds = output[:,text_len:,:].reshape(batch, num_frames // self.patch_size_t, -1, output.shape[-1])

        text_embeds = torch.chunk(text_embeds, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        image_embeds = torch.chunk(image_embeds, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        return torch.cat([text_embeds, image_embeds], dim=1)

    new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
    transformer.patch_embed.forward = new_patch_embed

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    
    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"

    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_video_input_parameters(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )
    
    parallelize_transformer(pipe)
    
    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            prompt=input_config.prompt,
            num_inference_steps=1,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
