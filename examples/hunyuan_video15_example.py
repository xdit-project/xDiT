import time
import torch
from xfuser.config.diffusers import has_valid_diffusers_version, get_minimum_diffusers_version

if not has_valid_diffusers_version("hunyuanvideo_15"):
    minimum_diffusers_version = get_minimum_diffusers_version("hunyuanvideo_15")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use HunyuanVideo 1.5 models.")

from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import xFuserHunyuanVideo15Transformer3DWrapper
from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
)

def run_pipe(pipe: DiffusionPipeline, input_config, image, task):
    kwargs = {
        "prompt": input_config.prompt,
        "num_inference_steps": input_config.num_inference_steps, # Recommended value
        "num_frames": input_config.num_frames, # Recommended value
        "generator": torch.Generator(device="cuda").manual_seed(input_config.seed),
    }
    if task == "i2v":
        kwargs["image"] = image
    else: # t2v task
        kwargs["height"] = input_config.height
        kwargs["width"] = input_config.width
    return pipe(**kwargs).frames[0]

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["i2v", "t2v"],
        help="The task to run."
    )
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    is_last_process =  get_world_group().rank == get_world_group().world_size - 1

    image = None
    if args.task == "i2v":
        if not input_config.img_file_path:
            raise ValueError("Image file path must be provided for image-to-video task.")
        image = load_image(args.img_file_path)

    transformer = xFuserHunyuanVideo15Transformer3DWrapper.from_pretrained(
        engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
        subfolder="transformer",
    )
    pipe = HunyuanVideo15Pipeline if args.task == "t2v" else HunyuanVideo15ImageToVideoPipeline
    pipe = pipe.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipe.to(f"cuda:{local_rank}")

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")

        # one full pass to warmup the torch compiler
        output = run_pipe(pipe, input_config, image, args.task)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = run_pipe(pipe, input_config, image, args.task)

    end_time = time.time()

    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"
    )
    if input_config.output_type == "pil":
        if is_last_process:
            video_name = f"hunyuan_video_15_{args.task}_result_{parallel_info}_tc_{engine_args.use_torch_compile}_{input_config.height}x{input_config.width}.mp4"
            export_to_video(output, video_name, fps=24)
            print(f"video saved to {video_name}")

    if is_last_process:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
