"""Compare full-model PipeFusion DDPM outputs and RNG state with a reference run.

Record a reference from the original PR tree, then repeat on the updated tree:

    HF_HUB_OFFLINE=1 PYTHONPATH=. python -m torch.distributed.run \
        --standalone --nproc_per_node=2 \
        tests/e2e/model_executor/schedulers/hunyuandit_pipefusion_noise.py \
        --model /path/to/HunyuanDiT-v1.2-Diffusers/snapshot --output-dir /tmp/reference

Use the same command on the updated tree with a new output directory and
``--reference-dir /tmp/reference``. Both runs must use the same model snapshot,
hardware, dependencies, and attention backend. Model files must already exist;
this script never downloads them. It writes images, exact tensor/RNG artifacts,
and per-rank timings and peak GPU memory. Run directly with torchrun, not pytest.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


CASES = {
    "cfg_one_image": (6.0, 1),
    "cfg_two_images": (6.0, 2),
    "no_cfg": (1.0, 1),
}
PROMPT = "一只可爱的橘猫坐在窗台上，阳光明媚"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("XDIT_HUNYUANDIT_MODEL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    args = parser.parse_args()
    if not args.model or not Path(args.model).is_dir():
        parser.error("--model or XDIT_HUNYUANDIT_MODEL must name an existing local model snapshot")
    if args.reference_dir is not None and args.output_dir.resolve() == args.reference_dir.resolve():
        parser.error("output and reference directories must differ")
    return args


def decode_images(pipe, latents, device):
    # Match HunyuanDiTPipeline's decode and postprocess after retaining its exact
    # final latent output. These are real pipeline components, with no mocks.
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    decoded, nsfw = pipe.run_safety_checker(decoded, device, latents.dtype)
    denormalize = [True] * decoded.shape[0] if nsfw is None else [not flagged for flagged in nsfw]
    return pipe.image_processor.postprocess(decoded, output_type="pil", do_denormalize=denormalize)


def assert_reference(artifact, reference_path):
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    assert artifact.keys() == reference.keys(), f"artifact keys differ from {reference_path}"
    for name, tensor in artifact.items():
        expected = reference[name]
        assert tensor.shape == expected.shape, f"{name}: {tensor.shape} != {expected.shape}"
        assert tensor.dtype == expected.dtype, f"{name}: {tensor.dtype} != {expected.dtype}"
        assert torch.equal(tensor, expected), f"{name} differs from {reference_path}"


@torch.no_grad()
def main():
    args = parse_args()
    assert torch.cuda.is_available(), "HunyuanDiT E2E requires real CUDA or ROCm devices"
    assert int(os.environ.get("WORLD_SIZE", "1")) == 2, "launch with --nproc_per_node=2"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from xfuser import xFuserArgs, xFuserHunyuanDiTPipeline
    from xfuser.core.distributed import get_runtime_state, get_world_group, is_pipeline_last_stage

    engine_config, _ = xFuserArgs(
        model=args.model,
        pipefusion_parallel_degree=2,
        ulysses_degree=1,
        num_pipeline_patch=4,
        warmup_steps=1,
        attention_backend="sdpa",
    ).create_config()
    engine_config.runtime_config.dtype = torch.float16
    rank = get_world_group().rank
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": str(Path(args.model).resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "diffusers": importlib.metadata.version("diffusers"),
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "yunchang": importlib.metadata.version("yunchang"),
        "optional_backends": {
            name: importlib.util.find_spec(name) is not None
            for name in ("flash_attn", "flash_attn_interface", "flashinfer", "xformers", "torchao")
        },
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "prompt": PROMPT,
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "seed": 42,
        "pipefusion_degree": 2,
        "ulysses_degree": 1,
        "patches": 4,
        "warmup_steps": 1,
        "attention_backend": "sdpa",
        "case": args.case,
    }
    if args.reference_dir is not None:
        reference_metadata = json.loads((args.reference_dir / "run.json").read_text())
        for key, value in metadata.items():
            if key != "commit":
                assert value == reference_metadata[key], f"reference run has a different {key}"
    if rank == 0:
        (args.output_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    try:
        pipe = xFuserHunyuanDiTPipeline.from_pretrained(
            args.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
        for name, (guidance, image_count) in cases.items():
            generator = torch.Generator(device=device).manual_seed(42)
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            memory_before_call = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = pipe(
                prompt=PROMPT,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=guidance,
                num_images_per_prompt=image_count,
                generator=generator,
                output_type="latent",
            )
            latents = images = None
            if is_pipeline_last_stage():
                assert output is not None, f"{name}: last pipeline stage returned no output"
                latents = output.images
                assert latents.shape == (image_count, pipe.transformer.config.in_channels, 128, 128)
                assert torch.isfinite(latents).all(), f"{name}: non-finite final latents"
                assert latents.float().std() > 0, f"{name}: constant final latents"
                images = decode_images(pipe, latents, device)
                assert len(images) == image_count, f"{name}: wrong image count"
                assert all(image.size == (1024, 1024) for image in images), f"{name}: wrong image size"
            else:
                assert output is None, f"{name}: an intermediate stage returned an image"
            torch.cuda.synchronize(device)
            metrics = {
                "case": name,
                "rank": rank,
                "guidance_scale": guidance,
                "images_per_prompt": image_count,
                "elapsed_seconds": time.perf_counter() - started,
                "memory_before_call_bytes": memory_before_call,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            artifact = {
                "generator_state": generator.get_state(),
                "next_random_draw": torch.randn(32, device=device, generator=generator).cpu(),
            }
            if latents is not None:
                artifact["final_latents"] = latents.cpu()
                artifact["images"] = torch.from_numpy(np.stack([np.asarray(image) for image in images]))
                assert artifact["images"].amax() > artifact["images"].amin(), f"{name}: constant image output"
                for index, image in enumerate(images):
                    image.save(args.output_dir / f"{name}_{index}.png")
            artifact_name = f"{name}_rank{rank}.pt"
            if args.reference_dir is not None:
                assert_reference(artifact, args.reference_dir / artifact_name)
                metrics["exact_reference_match"] = True
            torch.save(artifact, args.output_dir / artifact_name)
            (args.output_dir / f"{name}_rank{rank}.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)
            torch.distributed.barrier()
            del output, latents, images, artifact
    finally:
        get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
