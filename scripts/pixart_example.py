import argparse
import torch

from distrifuser.pipelines.pixartalpha import DistriPixArtAlphaPipeline
from distrifuser.utils import DistriConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="PixArt-alpha/PixArt-XL-2-1024-MS", type=str, help="Path to the pretrained model.")
    parser.add_argument("--parallelism", "-p", default="patch", type=str, choices=["patch", "naive_patch"],help="Parallelism to use.")
    args = parser.parse_args()

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, 
                                 do_classifier_free_guidance=False, split_batch=False, 
                                 parallelism=args.parallelism)
    pipeline = DistriPixArtAlphaPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        # variant="fp16",
        # use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    output = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    if distri_config.rank == 0:
        output.images[0].save("astronaut.png")

if __name__ == "__main__":
    main()