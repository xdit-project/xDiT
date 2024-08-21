import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.distributed import (
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
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")

    pipe.enable_model_cpu_offload()

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # prompt_embeds, _ = pipe.encode_prompt(
    #     prompt=input_config.prompt,
    #     do_classifier_free_guidance=True,
    #     num_videos_per_prompt=1,
    #     max_sequence_length=226,
    #     device="cuda",
    #     dtype=torch.float16,
    # )
    
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=49,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type='pt',
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6,
        # prompt_embeds=prompt_embeds,
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if get_data_parallel_rank() == get_data_parallel_world_size() - 1:
        videos = output.frames[0]
        export_to_video(videos, "output.mp4", fps=8)
        
    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )
    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()