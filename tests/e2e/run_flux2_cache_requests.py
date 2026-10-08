"""Compare consecutive FLUX.2 FBCache requests with fresh pipeline instances.

Run on one accelerator with a local FLUX.2 Klein checkpoint, for example:
    PYTHONPATH=. python tests/e2e/run_flux2_cache_requests.py \
        --model /path/to/FLUX.2-klein-4B --output-dir /tmp/flux2-cache-requests
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
from xfuser.model_executor.cache.adapters import apply_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    rendezvous = tempfile.TemporaryDirectory()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{rendezvous.name}/init",
    )
    initialize_model_parallel()

    def load():
        pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(
            "cuda:0"
        )
        pipe.set_progress_bar_config(disable=True)
        apply_cache("fbcache", num_steps=8, pipe=pipe, cache_config='{"residual_diff_threshold": 0.4}')
        return pipe

    def request(pipe, prompt, size):
        full_steps = []
        last_block = pipe.transformer.transformer_blocks[0].transformer_blocks[-1]
        hook = last_block.register_forward_hook(lambda *unused: full_steps.append(1))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            image = pipe(
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=8,
                guidance_scale=1.0,
                generator=torch.Generator("cuda:0").manual_seed(42),
                output_type="np",
            ).images[0]
            assert np.isfinite(image).all(), "The decoded image must contain only finite values"
        finally:
            hook.remove()
        torch.cuda.synchronize()
        return image, {
            "size": size,
            "full_steps": len(full_steps),
            "elapsed_seconds": time.perf_counter() - start,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }

    report = {
        "model": args.model,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "steps": 8,
        "threshold": 0.4,
        "requests": [],
    }
    try:
        pipe = load()
        warmup, _ = request(pipe, "A ginger cat sitting on a sunny windowsill.", 256)
        Image.fromarray((warmup * 255).round().astype(np.uint8)).save(args.output_dir / "first.png")
        for name, prompt, size in [
            ("same_shape", "A golden retriever playing in a green meadow.", 256),
            ("changed_shape", "A mountain lake at sunrise, with clear reflections.", 384),
        ]:
            reused, result = request(pipe, prompt, size)
            fresh_pipe = load()
            fresh, fresh_result = request(fresh_pipe, prompt, size)
            np.testing.assert_array_equal(reused, fresh)
            assert result["full_steps"] == fresh_result["full_steps"]
            assert 0 < result["full_steps"] < 8, "The request must compute and also exercise cache hits"
            result.update(name=name, max_abs_diff=float(np.max(np.abs(reused - fresh))), fresh=fresh_result)
            report["requests"].append(result)
            for suffix, image in [("reused", reused), ("fresh", fresh)]:
                Image.fromarray((image * 255).round().astype(np.uint8)).save(args.output_dir / f"{name}_{suffix}.png")
            print(json.dumps(result), flush=True)
            del fresh_pipe
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.distributed.destroy_process_group()
        rendezvous.cleanup()


if __name__ == "__main__":
    main()
