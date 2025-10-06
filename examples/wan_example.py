import time
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    dtype = torch.bfloat16
    device = torch.device("cuda")
    pipe = xFuserWanImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(f"cuda:{local_rank}")

    image = load_image(
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        image=image,
        prompt=input_config.prompt,
        #negative_prompt=X,
        num_inference_steps=40,
        num_frames=81,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]
    if pipe.is_dp_last_group():
        export_to_video(output, "i2v_output.mp4")

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
