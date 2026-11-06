"""Measure one pretrained Wan2.2 transport per process, with decoded-output diagnostics."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed.sdma import ulysses
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
from xfuser.model_executor.models.runner_models.wan import xFuserWan22T2VModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="AITER_FP8")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variant", choices=("rccl", "sdma", "sdmaq"), default="sdmaq")
    parser.add_argument("--max-repeat-rmse", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    calls = 0
    original_begin = ulysses.begin_qk

    def counted_begin(*positional, **keywords):
        nonlocal calls
        calls += 1
        return original_begin(*positional, **keywords)

    ulysses.begin_qk = counted_begin
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 4096
    config = xFuserArgs(
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        attention_backend=args.backend, ulysses_degree=int(os.environ["WORLD_SIZE"]),
        use_torch_compile=True, num_iterations=1, warmup_calls=0,
        ulysses_a2a="default" if args.variant == "rccl" else "sdma",
        ulysses_sdma_head_groups=None if args.variant == "rccl" else 1,
        ulysses_sdma_overlap_qkv=args.variant == "sdmaq",
        output_directory=str(args.output),
    )
    model = xFuserWan22T2VModel(config)
    request = model.preprocess_args({
        "prompt": ["A coastal lighthouse above calm water at sunrise, slow camera movement."],
        "dataset_path": None, "seed": 42, "height": 720, "width": 1280,
        "num_frames": 81, "num_inference_steps": args.steps, "input_images": [], "guidance_scale_2": None,
    })
    report = {"backend": args.backend, "variant": args.variant, "steps": args.steps, "seed": 42, "requests": []}
    reference = None
    try:
        model.initialize(request)
        for cycle in range(args.repeats + 1):
            before = calls
            output, timings = model.run(request)
            if args.variant == "sdmaq":
                assert calls - before >= 40 * args.steps, "split-QK Wan blocks did not execute"
            all_timings = [None] * dist.get_world_size()
            dist.all_gather_object(all_timings, timings)
            result = {"warmup": cycle == 0, "rank_seconds": all_timings, "split_qk_calls": calls - before}
            if dist.get_rank() == 0:
                assert output is not None and output.videos is not None and len(output.videos) == 1
                video = np.stack([np.asarray(frame) for frame in output.videos[0]])
                video = video.astype(np.float32) / 255 if video.dtype == np.uint8 else video.astype(np.float32)
                assert video.shape == (81, 720, 1280, 3) and np.isfinite(video).all()
                assert video.min() >= 0 and video.max() <= 1
                if reference is None:
                    reference = video.copy()
                difference = video - reference
                result["max_abs_diff"] = float(np.abs(difference).max())
                result["rmse"] = float(np.sqrt(np.mean(difference * difference)))
                result["shape"] = list(video.shape)
                result["within_first_1_255"] = result["max_abs_diff"] <= 1 / 255
                report["requests"].append(result)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                if cycle == args.repeats:
                    np.save(args.output / "video.npy", video.astype(np.float16))
                    for frame in (0, len(video) // 2, len(video) - 1):
                        image = np.rint(np.clip(video[frame], 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(image).save(args.output / f"frame-{frame}.png")
                if args.max_repeat_rmse is not None:
                    assert result["rmse"] <= args.max_repeat_rmse, result
            del output
            dist.barrier()
    finally:
        ulysses.begin_qk = original_begin
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
