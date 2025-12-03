import time
import torch
from diffusers import ZImagePipeline
from xfuser.config.diffusers import has_valid_diffusers_version, get_minimum_diffusers_version

if not has_valid_diffusers_version("zimage"):
    minimum_diffusers_version = get_minimum_diffusers_version("zimage")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Z-Image models.")

from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper
from diffusers import DiffusionPipeline

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
)

def run_pipe(pipe: DiffusionPipeline, input_config):
    # Pipe implementation currently encodes the prompt in-place,
    # causing any subsequent calls to use the already encoded prompt as prompt,
    # causing cascading encodings unless we provide a new list each time.
    prompt = str(input_config.prompt)

    return pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=prompt,
        num_inference_steps=9, # Recommended value
        guidance_scale=0.0, # Recommended value
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).images[0]

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    is_last_process =  get_world_group().rank == get_world_group().world_size - 1

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
        engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
        subfolder="transformer",
    )
    pipe = ZImagePipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(f"cuda:{local_rank}")
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")

        # one full pass to warmup the torch compiler
        output = run_pipe(pipe, input_config)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = run_pipe(pipe, input_config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"
    )
    if input_config.output_type == "pil":
        if is_last_process:
            image_name = f"zimage_result_{parallel_info}_tc_{engine_args.use_torch_compile}.png"
            output.save(f"./results/{image_name}")
            print(f"image saved to ./results/{image_name}")

    if is_last_process:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
