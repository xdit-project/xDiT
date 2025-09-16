import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline

from xfuser import xFuserArgs
from xfuser.parallel import xDiTParallel
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group


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

    paralleler(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    if input_config.output_type == "pil":
        paralleler.save("results", "stable_diffusion_3")


if __name__ == "__main__":
    main()
