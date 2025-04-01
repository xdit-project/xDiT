import logging
import time
import torch
import torch.distributed
from xfuser import xFuserStableDiffusionXLPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)
from diffusers import StableDiffusionXLPipeline

def main():
    # Initialize argument parser
    parser = FlexibleArgumentParser(description="xFuser SDXL Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    

    # Set runtime configuration
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    # Initialize pipeline
    pipe = xFuserStableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    )

    # Handle device placement
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    # Record initial memory usage
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # Prepare for inference
    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    # Run inference
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=7.5,  # SDXL默认guidance scale
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # Generate parallel configuration info string
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}"
    )

    # Save generated images
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"sdxl_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    # Print performance metrics
    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    
    # Cleanup
    get_runtime_state().destroy_distributed_env()

if __name__ == "__main__":
    main() 