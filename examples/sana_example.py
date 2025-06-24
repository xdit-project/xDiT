import time
import os
import torch
import warnings
import torch.distributed
from xfuser import xFuserSanaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size

data_type_dict = {
    "Sana_1600M_1024px_diffusers": torch.float16,
    "Sana_1600M_4Kpx_BF16_diffusers": torch.bfloat16,
    "SANA1.5_4.8B_1024px_diffusers": torch.bfloat16,
    "SANA1.5_1.6B_1024px_diffusers": torch.bfloat16,
}

def get_data_type(model_path):
    for model_name, data_type in data_type_dict.items():
        if model_name in model_path:
            return data_type
    warnings.warn(f"Unknown model path: {model_path}, using default data type: torch.float16")
    return torch.float16


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    data_type = get_data_type(engine_config.model_config.model)
    engine_config.runtime_config.dtype = data_type
    pipe = xFuserSanaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=data_type,
    ).to(f"cuda:{local_rank}")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.enable_tiling(tile_sample_min_width=1024, tile_sample_min_height=1024)

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=4.5
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image.save(
                    f"./results/sana_result_{parallel_info}_{image_rank}.png"
                )
                print(
                    f"image {i} saved to ./results/sana_result_{parallel_info}_{image_rank}.png"
                )

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, peak memory: {peak_memory/1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
