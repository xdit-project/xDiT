import argparse
import torch

from pipefuser.pipelines.dit import DistriDiTPipeline
from pipefuser.utils import DistriConfig

HAS_LONG_CTX_ATTN = False
try:
    from yunchang import set_seq_parallel_pg

    HAS_LONG_CTX_ATTN = True
except ImportError:
    print("yunchang not found")

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
        "--ulysses_degree",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_use_ulysses_low",
        action="store_true",
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
    )

    pipeline = DistriDiTPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        # variant="fp16",
        # use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

    case_name = f"{args.parallelism}_{args.sync_mode}_u{args.ulysses_degree}_w{distri_config.world_size}"

    # warmup
    output = pipeline(
        # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
        prompt=["panda"],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipeline(
        # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
        prompt=["panda"],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    if distri_config.rank == 0:
        elapsed_time = end_time - start_time
        print(
            f"{case_name}: elapse: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )
        output.images[0].save(f"./results/{case_name}_panda.png")


if __name__ == "__main__":
    main()
