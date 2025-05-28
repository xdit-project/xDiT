import sys
import functools
from typing import List, Optional, Tuple, Union, Any, Dict

import time
import torch

import diffusers
from diffusers import LuminaPipeline, DiffusionPipeline

import torch.distributed as dist
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    get_world_group,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
)

def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        encoder_mask = torch.chunk(encoder_mask, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        image_rotary_emb = torch.chunk(image_rotary_emb, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]

        output = original_forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
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
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    device = torch.device(f"cuda:{local_rank}")

    initialize_model_parallel(
        classifier_free_guidance_degree=engine_config.parallel_config.cfg_degree,
    )
    pipe = LuminaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)

    pipe.vae.enable_tiling()

    parallelize_transformer(pipe)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).images[0]

    end_time = time.time()
    elapsed_time = end_time - start_time

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if local_rank == 0:
        output.save(f"results/lumina_cfg_{parallel_info}.png")
        print(f"epoch time: {elapsed_time:.2f} sec")

    dist.destroy_process_group()
