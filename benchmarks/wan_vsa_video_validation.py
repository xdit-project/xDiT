"""Run paired Dense/VSA Wan validations without reloading model weights."""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from xfuser import xFuserArgs
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.runner import xFuserModelRunner


DEFAULT_PROMPTS = (
    "Two anthropomorphic cats in boxing gear fight on a spotlighted stage.",
    "A red sailboat crosses a stormy ocean at sunset, cinematic camera.",
    "A robot chef chops vegetables in a bright modern kitchen.",
)


def _tensor_video(video) -> torch.Tensor:
    frames = video[0] if isinstance(video, list) and len(video) == 1 else video
    array = np.stack([np.asarray(frame) for frame in frames])
    result = torch.from_numpy(array).float()
    if result.max() > 1.0:
        result.div_(255.0)
    return result.permute(0, 3, 1, 2).contiguous()


def _ssim(reference: torch.Tensor, actual: torch.Tensor) -> float:
    channels = reference.shape[1]
    coords = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coords * coords) / (2 * 1.5 * 1.5))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(channels, 1, 11, 11)
    mu_x = F.conv2d(reference, window, padding=5, groups=channels)
    mu_y = F.conv2d(actual, window, padding=5, groups=channels)
    sigma_x = F.conv2d(reference * reference, window, padding=5, groups=channels)
    sigma_y = F.conv2d(actual * actual, window, padding=5, groups=channels)
    sigma_xy = F.conv2d(reference * actual, window, padding=5, groups=channels)
    sigma_x -= mu_x.square()
    sigma_y -= mu_y.square()
    sigma_xy -= mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def _metrics(reference, actual, lpips_metric=None) -> dict:
    reference = _tensor_video(reference)
    actual = _tensor_video(actual)
    mse = float((reference - actual).square().mean())
    result = {
        "mae": float((reference - actual).abs().mean()),
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": _ssim(reference, actual),
    }
    if lpips_metric is not None:
        lpips_metric.reset()
        sampled_reference = F.interpolate(
            reference[::8], size=(224, 224), mode="bilinear",
            align_corners=False,
        )
        sampled_actual = F.interpolate(
            actual[::8], size=(224, 224), mode="bilinear",
            align_corners=False,
        )
        for start in range(0, len(sampled_reference), 2):
            lpips_metric.update(
                sampled_reference[start:start + 2],
                sampled_actual[start:start + 2],
            )
        result["lpips_alex_frame_stride_8"] = float(lpips_metric.compute())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))

    config = xFuserArgs(
        model="Wan2.1-T2V", attention_backend="AITER_VSA",
        cross_attention_backend="AITER", dit_parallel_size=4,
        ulysses_degree=4, ring_degree=1, height=args.height,
        width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps, prompt=args.prompts[0],
        seed=args.seeds[0], guidance_scale=6.0, flow_shift=8.0,
        output_directory="/tmp/wan-vsa-validation", warmup_steps=0,
        num_iterations=args.iterations,
        vsa_drop_rates=[0.25, 0.40], vsa_prob_threshold=0.9,
        input_images=[],
    )
    runner = xFuserModelRunner(vars(config))
    runner.model.settings.model_name = args.model_path
    first_input = runner.preprocess_args(vars(config))
    runner.initialize(first_input)
    runtime = get_runtime_state()
    rows = []
    lpips_metric = None
    if rank == 0:
        try:
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True, sync_on_compute=False
            )
        except (ImportError, RuntimeError):
            pass

    for prompt in args.prompts:
        for seed in args.seeds:
            run_args = dict(first_input, prompt=prompt, seed=seed)
            runtime.attention_backend = AttentionBackendType.AITER
            dense, dense_time = runner.run(run_args)
            runtime.attention_backend = AttentionBackendType.AITER_VSA
            sparse, sparse_time = runner.run(run_args)
            if rank == 0:
                rows.append({
                    "prompt": prompt,
                    "seed": seed,
                    "metrics": _metrics(
                        dense.videos, sparse.videos, lpips_metric
                    ),
                    "dense_seconds": dense_time[-1],
                    "vsa_seconds": sparse_time[-1],
                    "speedup": dense_time[-1] / sparse_time[-1],
                })

    if rank == 0:
        summary = {
            "mean_psnr": float(np.mean([r["metrics"]["psnr"] for r in rows])),
            "mean_ssim": float(np.mean([r["metrics"]["ssim"] for r in rows])),
            "mean_speedup": float(np.mean([r["speedup"] for r in rows])),
        }
        if lpips_metric is not None:
            summary["mean_lpips_alex_frame_stride_8"] = float(np.mean([
                r["metrics"]["lpips_alex_frame_stride_8"] for r in rows
            ]))
        with open(args.output, "w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle, indent=2)
    runner.cleanup()


if __name__ == "__main__":
    main()
