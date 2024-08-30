import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group, 
    get_data_parallel_rank, 
    get_data_parallel_world_size,
    get_runtime_state,
)
from diffusers.utils import export_to_video


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model, 
        engine_config=engine_config, 
        torch_dtype=torch.bfloat16, 
    ) 
    if args.enable_sequential_cpu_offload: 
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        pipe.vae.enable_tiling()
    else: 
        pipe = pipe.to(f"cuda:{local_rank}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=49,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6,
    ).frames[0]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    
    if get_data_parallel_rank() == get_data_parallel_world_size() - 1:
        export_to_video(output, "results/output.mp4", fps=8)
        
    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )
    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()