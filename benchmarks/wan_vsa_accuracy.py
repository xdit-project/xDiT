"""Trace Wan diffusion accuracy per timestep.

Run with torchrun. Capture a dense reference first, then compare VSA online:

  torchrun --nproc_per_node=4 benchmarks/wan_vsa_accuracy.py capture ...
  torchrun --nproc_per_node=4 benchmarks/wan_vsa_accuracy.py compare \
      --reference dense.pt --attention-backend AITER_VSA ...
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from xfuser import xFuserArgs
from xfuser.runner import xFuserModelRunner


def _metric(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    reference = reference.to(actual.device, torch.float32)
    actual = actual.float()
    delta = actual - reference
    return {
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ),
        "cosine": float(
            F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0)
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "compare"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--attention-backend", default="AITER")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--vsa-drop-rates", type=float, nargs="+")
    parser.add_argument("--vsa-prob-threshold", type=float, default=0.9)
    parser.add_argument("--ulysses-degree", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "compare" and not args.reference:
        parser.error("--reference is required in compare mode")
    return args


def main():
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    config = xFuserArgs(
        model="Wan2.1-T2V",
        attention_backend=args.attention_backend,
        cross_attention_backend="AITER",
        dit_parallel_size=args.ulysses_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=1,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        prompt=args.prompt,
        negative_prompt=None,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        output_directory="/tmp/wan-vsa-accuracy",
        warmup_steps=0,
        vsa_drop_rates=args.vsa_drop_rates,
        vsa_prob_threshold=args.vsa_prob_threshold,
        input_images=[],
    )
    runner = xFuserModelRunner(vars(config))
    runner.model.settings.model_name = args.model_path
    input_args = runner.preprocess_args(vars(config))
    runner.initialize(input_args)
    runner.model.pipe.transformer.attention_kwargs[
        "vsa_collect_density"
    ] = True

    reference = None
    if rank == 0 and args.mode == "compare":
        reference = torch.load(args.reference, map_location="cpu")["trace"]

    scheduler = runner.model.pipe.scheduler
    original_step = scheduler.step
    trace = []
    rows = []

    def traced_step(*step_args, **step_kwargs):
        output = original_step(*step_args, **step_kwargs)
        model_output = (
            step_args[0] if step_args else step_kwargs["model_output"]
        )
        timestep = (
            step_args[1] if len(step_args) > 1 else step_kwargs["timestep"]
        )
        latent = output[0] if isinstance(output, tuple) else output.prev_sample
        index = len(trace) if args.mode == "capture" else len(rows)
        if rank == 0:
            transformer = runner.model.pipe.transformer
            schedule = transformer.attention_kwargs
            metadata = {
                "step": index,
                "timestep": float(timestep.reshape(-1)[0]),
                "vsa_step_index": schedule.get("vsa_step_index"),
                "vsa_num_steps": schedule.get("vsa_num_steps"),
                "drop_rate": schedule.get("vsa_effective_drop_rate"),
                "use_dense": schedule.get("vsa_use_dense"),
            }
            density = schedule.get("vsa_last_density")
            if density is not None:
                metadata["last_layer_density"] = float(density)
            if args.mode == "capture":
                trace.append(
                    {
                        **metadata,
                        "noise": model_output.detach().float().cpu(),
                        "latent": latent.detach().float().cpu(),
                    }
                )
            else:
                expected = reference[index]
                rows.append(
                    {
                        **metadata,
                        "noise": _metric(expected["noise"], model_output),
                        "latent": _metric(expected["latent"], latent),
                    }
                )
        return output

    scheduler.step = traced_step
    _, timings = runner.run(input_args)
    if rank == 0:
        if args.mode == "capture":
            torch.save({"trace": trace, "timings": timings}, args.output)
        else:
            with open(args.output, "w") as handle:
                json.dump({"rows": rows, "timings": timings}, handle, indent=2)
    runner.cleanup()


if __name__ == "__main__":
    main()
