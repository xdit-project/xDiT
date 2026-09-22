import os
import time

import torch

from xfuser import xFuserArgs, xFuserLumina2Pipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    get_world_group,
)


def main():
    parser = FlexibleArgumentParser(description="Lumina-Image-2.0 with xDiT")
    parser.add_argument(
        "--cfg_trunc_ratio",
        type=float,
        default=0.25,
        help="Fraction of denoising steps that use classifier-free guidance.",
    )
    parser.add_argument(
        "--no_cfg_normalization",
        action="store_true",
        help="Disable Lumina2's normalization-based guidance rescaling.",
    )
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16

    local_rank = get_world_group().local_rank
    device = f"cuda:{local_rank}"
    pipe = xFuserLumina2Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model_memory = torch.cuda.max_memory_allocated(device=device)
    pipe.prepare_run(input_config)

    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize(device=device)
    start_time = time.perf_counter()
    output = pipe(
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        height=input_config.height,
        width=input_config.width,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        cfg_trunc_ratio=args.cfg_trunc_ratio,
        cfg_normalization=not args.no_cfg_normalization,
        max_sequence_length=input_config.max_sequence_length,
        output_type=input_config.output_type,
        generator=torch.Generator(device=device).manual_seed(input_config.seed),
    )
    torch.cuda.synchronize(device=device)
    elapsed_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=device)

    if input_config.output_type == "pil" and pipe.is_dp_last_group():
        os.makedirs("results", exist_ok=True)
        dp_group_index = get_data_parallel_rank()
        dp_batch_size = (
            input_config.batch_size + get_data_parallel_world_size() - 1
        ) // get_data_parallel_world_size()
        for i, image in enumerate(output.images):
            image_index = dp_group_index * dp_batch_size + i
            output_path = f"results/lumina2_result_{image_index}.png"
            image.save(output_path)
            print(f"Image saved to {output_path}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"Inference time: {elapsed_time:.2f} sec, "
            f"model memory: {model_memory / 1e9:.2f} GB, "
            f"peak memory: {peak_memory / 1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
