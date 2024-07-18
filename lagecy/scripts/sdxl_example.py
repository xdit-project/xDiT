import torch

from lagecy.pipefuser.pipelines import DistriSDXLPipeline
from lagecy.pipefuser.utils import DistriConfig


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        type=str,
        help="Path or Id to the pretrained model.",
    )
    args = parser.parse_args()

    distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)

    pipeline = DistriSDXLPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        variant="fp16",
        use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    image = pipeline(
        prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        generator=torch.Generator(device="cuda").manual_seed(233),
    ).images[0]

    if distri_config.rank == 0:
        image.save("astronaut.png")
