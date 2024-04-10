import argparse
import torch

from distrifuser.pipelines.pixartalpha import DistriPixArtAlphaPipeline
from distrifuser.utils import DistriConfig
import time
from yunchang import set_seq_parallel_pg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        default="patch",
        type=str,
        choices=["patch", "naive_patch"],
        help="Parallelism to use.",
    )
    parser.add_argument(
        "--use_seq_parallel_attn",
        action="store_true",
        default=False,
        help="Enable sequence parallel attention.",
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

    args = parser.parse_args()

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(
        height=1024,
        width=1024,
        warmup_steps=4,
        do_classifier_free_guidance=False,
        split_batch=False,
        parallelism=args.parallelism,
        mode=args.sync_mode,
        use_seq_parallel_attn=args.use_seq_parallel_attn,
    )

    if distri_config.use_seq_parallel_attn:
        ulysses_degree = distri_config.world_size
        ring_degree = distri_config.world_size // ulysses_degree
        set_seq_parallel_pg(
            ulysses_degree, ring_degree, distri_config.rank, distri_config.world_size
        )

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

    start_time = time.time()

    output = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    end_time = time.time()

    if distri_config.rank == 0:
        elapsed_time = end_time - start_time
        print(f"epoch time: {elapsed_time:.2f}s")
        output.images[0].save("astronaut.png")


if __name__ == "__main__":
    main()
