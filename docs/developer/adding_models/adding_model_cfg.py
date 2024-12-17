# Example for parallelize new models with USP
# run with 
#     torchrun --nproc_per_node=2 \
#          adding_cogvideox.py <cogvideox-checkpoint-path>
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
)

from diffusers.utils import export_to_video

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
        timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        
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
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
    
if __name__ == "__main__":
    dist.init_process_group("nccl")
    init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size()
    )
    initialize_model_parallel(
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
