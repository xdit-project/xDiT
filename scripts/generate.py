import argparse
import os

import torch
from datasets import load_dataset
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from tqdm import trange

from distrifuser.pipelines import DistriSDXLPipeline, DistriDiTPipeline, DistriPixArtAlphaPipeline
from distrifuser.utils import DistriConfig
from distrifuser.logger import init_logger
logger = init_logger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Diffuser specific arguments
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--image_size", type=int, nargs="*", default=1024, help="Image size of generation")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["euler", "dpm-solver", "ddim"])

    # DistriFuser specific arguments
    parser.add_argument("--pipeline", type=str, default="dit", choices=["sdxl", "dit", "pixart"])
    parser.add_argument("--model_path", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument(
        "--no_split_batch", action="store_true", help="Disable the batch splitting for classifier-free guidance"
    )
    parser.add_argument("--warmup_steps", type=int, default=4, help="Number of warmup steps")
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=["separate_gn", "stale_gn", "corrected_async_gn", "sync_gn", "full_sync", "no_sync"],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="patch",
        choices=["patch", "tensor", "naive_patch", "pipeline"],
        help="patch parallelism, tensor parallelism or naive patch",
    )
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="alternate",
        choices=["row", "col", "alternate"],
        help="Split scheme for naive patch",
    )
    parser.add_argument("--no_cuda_graph", action="store_true", help="Disable CUDA graph")

    parser.add_argument("--split", nargs=2, type=int, default=None, help="Split the dataset into chunks")

    parser.add_argument("--num_micro_batch", type=int, default=2, help="Number of micro batches")

    # Dataset specific arguments
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["coco", "imagenet"])

    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the dataset")
    parser.add_argument("--end_idx", type=int, default=5000, help="End index of the dataset")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
    else:
        if len(args.image_size) == 1:
            args.image_size = [args.image_size[0], args.image_size[0]]
        else:
            assert len(args.image_size) == 2
    distri_config = DistriConfig(
        height=args.image_size[0],
        width=args.image_size[1],
        do_classifier_free_guidance=args.guidance_scale > 1 if args.pipeline != "dit" else False,
        split_batch= not args.no_split_batch if args.pipeline != "dit" else False, 
        warmup_steps=args.warmup_steps,
        mode=args.sync_mode,
        use_cuda_graph=not args.no_cuda_graph,
        parallelism=args.parallelism,
        split_scheme=args.split_scheme,
        num_micro_batch=args.num_micro_batch,
        scheduler=args.scheduler
    )

    pretrained_model_name_or_path = args.model_path
    if args.scheduler == "euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    elif args.scheduler == "dpm-solver":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    elif args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    else:
        raise NotImplementedError
    
    if args.pipeline == "dit":
        pipeline = DistriDiTPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            distri_config=distri_config,
            scheduler=scheduler,
        )
    elif args.pipeline == "sdxl":
        pipeline = DistriSDXLPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            distri_config=distri_config,
            variant="fp16",
            use_safetensors=True,
            scheduler=scheduler,
        )
    elif args.pipeline == "pixart":
        pipeline = DistriPixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            distri_config=distri_config,
            output_type="pil"
            # scheduler=args.scheduler
        )
    pipeline.set_progress_bar_config(disable=distri_config.rank != 0, position=1, leave=False)

    if args.output_root is None:
        args.output_root = os.path.join(
            "results",
            f"{args.dataset}",
            f"{args.scheduler}-{args.num_inference_steps}",
            f"gpus{distri_config.world_size if args.no_split_batch else distri_config.world_size // 2}-"
            f"warmup{args.warmup_steps}-{args.sync_mode}-{args.num_micro_batch}-{args.parallelism}",
        )
    if distri_config.rank == 0:
        os.makedirs(args.output_root, exist_ok=True)

    if args.dataset == "coco":
        dataset = load_dataset("HuggingFaceM4/COCO", name="2014_captions", split="validation")
    elif args.dataset == "imagenet":
        dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    
    # dataset = dataset.shuffle(seed=42)

    if args.split is not None:
        assert args.split[0] < args.split[1]
        chunk_size = (5000 + args.split[1] - 1) // args.split[1]
        start_idx = args.split[0] * chunk_size
        end_idx = min((args.split[0] + 1) * chunk_size, 5000)
    else:
        start_idx = args.start_idx
        end_idx = args.end_idx 

        

    for i in trange(start_idx, end_idx, disable=distri_config.rank != 0, position=0, leave=False):
        if args.pipeline == "dit":
            prompt = dataset["label"][i]
        elif args.pipeline in ["sdxl", "pixart"]:
            prompt = dataset["sentences_raw"][i][i % len(dataset["sentences_raw"][i])]
        seed = i

        image = pipeline(
            prompt,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        if distri_config.rank == 0:
            output_path = os.path.join(args.output_root, f"{i:05d}.png")
            image.images[0].save(output_path)


if __name__ == "__main__":
    main()
