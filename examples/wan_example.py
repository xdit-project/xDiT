import time
import torch
import numpy as np
from xfuser.config.diffusers import get_minimum_diffusers_version, has_valid_diffusers_version
if not has_valid_diffusers_version("wan"):
    minimum_diffusers_version = get_minimum_diffusers_version("wan")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Wan.")

from xfuser import xFuserWanImageToVideoPipeline, xFuserWanPipeline
from diffusers.utils import export_to_video, load_image

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_sp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    initialize_runtime_state,
    is_dp_last_group,
)

TASK_FPS = {
    "i2v": 16,
    "t2v": 16,
    "ti2v": 24,
}

TASK_FLOW_SHIFT = {
    "i2v": 5,
    "t2v": 12,
    "ti2v": 5,
}


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["i2v", "t2v", "ti2v"],
        help="The task to run."
    )
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    is_i2v_task = args.task == "i2v" or (args.task == "ti2v" and args.img_file_path != None)
    task_pipeline = xFuserWanImageToVideoPipeline if is_i2v_task else xFuserWanPipeline
    pipe = task_pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
        engine_config=engine_config,
        cache_args=None
    )
    pipe.scheduler.config.flow_shift = TASK_FLOW_SHIFT[args.task]
    initialize_runtime_state(pipe, engine_config)
    pipe = pipe.to(f"cuda:{local_rank}")

    if not args.img_file_path and args.task == "i2v":
        raise ValueError("Please provide an input image path via --img_file_path. This may be a local path or a URL.")

    if is_i2v_task:
        image = load_image(args.img_file_path)
        max_area = input_config.height * input_config.width
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        if is_dp_last_group():
            print("Max area is calculated from input height and width values, but the aspect ratio for the output video is retained from the input image.")
            print(f"Input image resolution: {image.height}x{image.width}")
            print(f"Generating a video with resolution: {height}x{width}")
    else: # T2V or TI2V with no image
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = input_config.height // mod_value * mod_value
        width = input_config.width // mod_value * mod_value
        if height != input_config.height or width != input_config.width:
            if is_dp_last_group():
                print(f"Adjusting height and width to be multiples of {mod_value}. New dimensions: {height}x{width}")
        image = None

    def run_pipe(input_config, image):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optional_kwargs = {}
        if image:
            optional_kwargs["image"] = image
        attn_kwargs = {'use_fp8_attn': engine_config.runtime_config.use_fp8_attn,
                       'use_hybrid_fp8_attn': engine_config.runtime_config.use_hybrid_fp8_attn}
        output = pipe(
            height=height,
            width=width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            num_frames=input_config.num_frames,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            attention_kwargs=attn_kwargs,
            **optional_kwargs,
        ).frames[0]
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
        torch.cuda.synchronize()
        if is_dp_last_group():
            print(f"Iteration took {end - start}s, Peak memory: {peak_memory / 1024 ** 2:.2f} MB")
        return output

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")
        if pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="default")

        # one step to warmup the torch compiler
        _ = run_pipe(input_config, image)

    output = run_pipe(input_config, image)
    if is_dp_last_group():
        file_name = f"{args.task}_output.mp4"
        export_to_video(output, file_name, fps=TASK_FPS[args.task])
        print(f"Output video saved to {file_name}")

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
