import argparse
import torch

from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import (
    PixArtAlphaPipeline,
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of image",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="pil",
        choices=["latent", "pil"],
        help="latent saves memory, pil will results a memory burst in vae",
    )
    parser.add_argument(
        "--scheduler",
        "-s",
        default="FM-ED",
        type=str,
        choices=["dpm-solver", "ddim", "FM-ED"],
        help="Scheduler to use.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="An astronaut riding a green horse",
    )
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True

    # for DiT the height and width are fixed according to the model

    model_id = args.model_id

    if args.scheduler == "FM-ED":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_id,
            subfolder="scheduler",
        )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        scheduler=scheduler,
    ).to("cuda")

    # warmup
    output = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(42),
        output_type=args.output_type,
    )

    torch.cuda.reset_peak_memory_stats()

    case_name = f"baseline_hw_{args.height}_base"
    if args.output_file:
        case_name = args.output_file + "_" + case_name

    start_time = time.time()
    output = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(42),
        num_inference_steps=args.num_inference_steps,
        output_type=args.output_type,
    )

    end_time = time.time()
    torch.cuda.memory._record_memory_history(enabled=None)

    elapsed_time = end_time - start_time

    peak_memory = torch.cuda.max_memory_allocated(device="cuda")

    print(
        f"{case_name} epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
    )
    if args.output_type == "pil":
        print(f"save images to ./results/{case_name}.png")
        output.images[0].save(f"./results/{case_name}.png")


if __name__ == "__main__":
    main()
