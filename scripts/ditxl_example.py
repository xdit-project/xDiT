import argparse
import torch

from distrifuser.pipelines.dit import DistriDiTPipeline
from distrifuser.utils import DistriConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="facebook/DiT-XL-2-256", type=str, help="Path to the pretrained model.")
    args = parser.parse_args()

    distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, do_classifier_free_guidance=False, split_batch=False)
    pipeline = DistriDiTPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        # variant="fp16",
        # use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    output = pipeline(
        # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
        words=["panda"],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    if distri_config.rank == 0:
        output.images[0].save("panda.png")

if __name__ == "__main__":
    main()