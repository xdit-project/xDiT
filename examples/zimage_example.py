import logging
import os
import time
import torch
from transformers import AutoModel

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)
from xfuser.model_executor.pipelines.pipeline_zimage import xFuserZImagePipeline


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    # Z-Image uses a Qwen3 model as text encoder (not T5).
    # Pre-load it so we can optionally quantize before the pipe is built.
    text_encoder = AutoModel.from_pretrained(
        engine_config.model_config.model,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing Qwen text encoder to fp8")
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

    pipe = xFuserZImagePipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        text_encoder=text_encoder,
    ).to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=input_config.guidance_scale,
        max_sequence_length=input_config.max_sequence_length,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # SP only: no PP or CFG degree
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"
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
                image_name = f"zimage_result_{parallel_info}_{image_rank}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, "
            f"parameter memory: {parameter_peak_memory/1e9:.2f} GB, "
            f"peak memory: {peak_memory/1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
