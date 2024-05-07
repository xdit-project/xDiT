import argparse
import torch

from distrifuser.pipelines.dit import DistriDiTPipeline
from distrifuser.utils import DistriConfig

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="facebook/DiT-XL-2-256",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        default="patch",
        type=str,
        choices=["patch", "naive_patch", "tensor"],
        help="Parallelism to use.",
    )
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=[
            "separate_gn",
            "async_gn",
            "corrected_async_gn",
            "sync_gn",
            "full_sync",
            "no_sync",
        ],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--use_seq_parallel_attn",
        action="store_true",
        default=False,
        help="Enable sequence parallel attention.",
    )
    args = parser.parse_args()

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(
        height=1024,
        width=1024,
        warmup_steps=4,
        do_classifier_free_guidance=True,
        split_batch=False,
        parallelism=args.parallelism,
        mode=args.sync_mode,
        use_cuda_graph=False,
        use_seq_parallel_attn=args.use_seq_parallel_attn,
    )
    pipeline = DistriDiTPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        # variant="fp16",
        # use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

    # warmup
    output = pipeline(
        # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
        prompt=["panda"],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    start_time = time.time()

    output = pipeline(
        # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
        prompt=["panda"],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    end_time = time.time()

    if distri_config.rank == 0:
        elapsed_time = end_time - start_time
        print(f"elapse: {elapsed_time} sec")
        output.images[0].save("panda.png")


if __name__ == "__main__":
    main()
