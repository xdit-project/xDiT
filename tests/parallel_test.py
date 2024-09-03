from xfuser.parallel import xDiTParallel

import time
import os
import torch
from diffusers import StableDiffusion3Pipeline

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()

    local_rank = get_world_group().local_rank
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")

    paralleler = xDiTParallel(pipe, engine_config, input_config)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    paralleler(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    paralleler.save("results/", "stable_diffusion_3")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
