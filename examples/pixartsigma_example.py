import time
import torch
import torch.distributed
from xfuser import xFuserPixArtSigmaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.distributed import (
    get_world_group, 
    get_data_parallel_rank, 
    get_data_parallel_world_size
)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    pipe = xFuserPixArtSigmaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    pipe.prepare_run(input_config)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        use_resolution_binning=input_config.use_resolution_binning,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if input_config.output_type == "pil":
        global_rank = get_world_group().rank
        dp_group_world_size = get_data_parallel_world_size()
        dp_group_index = global_rank // dp_group_world_size
        num_dp_groups = engine_config.parallel_config.dp_degree
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if get_data_parallel_rank() == dp_group_world_size - 1:
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image.save(f"./results/pixart_sigma_result_{image_rank}.png")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )


if __name__ == '__main__':
    main()