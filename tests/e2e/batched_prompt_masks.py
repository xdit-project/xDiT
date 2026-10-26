"""Batches of prompts of different lengths: stock Diffusers against the xDiT wrapper, on pretrained weights.

Run directly with Python; the script spawns its own processes:
    python tests/e2e/batched_prompt_masks.py --model z-image --ulysses-degree 2

The stock pipeline runs on one GPU with two prompts of different lengths. The
same pipeline with the xDiT transformer wrapper (--attention-backend, else the
platform's default) then runs the same two prompts on --ulysses-degree GPUs. Every transformer call of both runs is
recorded. The summary gives the relative L2 distance of each call's output and
the PSNR of each decoded sample, then runs the stock transformer on every input
the wrapper saw (a "same-input" error). That separates a mask that is missing
or misplaced, whose error appears on the first call, from float32 rounding that
the sampler amplifies over the steps. --noise-floor adds the independent
trajectory of the stock pipeline with SDPA's math kernel instead of its default
one, which is that rounding alone. The run fails if a same-input error exceeds
--tolerance.

Models: z-image (Tongyi-MAI/Z-Image, guidance 4 batches the prompt with its
negative prompt), qwen-image (Qwen/Qwen-Image, true CFG 4) and hunyuan-video-1.5
(the 720p text-to-video checkpoint; use --even-text for a prompt whose text
shards over two ranks). float32 runs turn TF32 off.
"""

import argparse
import contextlib
import json
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_SHORT = "A red fox in the snow."
_LONG = (
    "A weathered fisherman mends a bright orange net on the wooden pier of a small harbor at sunrise, "
    "gulls circling above the moored boats, soft golden light, misty water, detailed photograph, 35mm film."
)
_FOX_SHORT = "A red fox walks slowly through a snowy forest."
_FOX_LONG = (
    "A red fox walks slowly through a snowy forest at dawn, pale golden light filtering between the pine trunks, "
    "fresh powder kicked up by its paws, realistic %s."
)
_MODELS = {
    "z-image": dict(checkpoint="Tongyi-MAI/Z-Image", prompts=(_SHORT, _LONG), size=(512, 512), steps=50),
    "qwen-image": dict(checkpoint="Qwen/Qwen-Image", prompts=(_SHORT, _LONG), size=(512, 512), steps=50),
    "hunyuan-video-1.5": dict(
        checkpoint="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        prompts=(_FOX_SHORT, _FOX_LONG % "wildlife cinematography"),
        size=(496, 848),
        steps=20,
    ),
}


def _classes(model):
    if model == "z-image":
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper

        return ZImagePipeline, ZImageTransformer2DModel, xFuserZImageTransformer2DWrapper
    if model == "qwen-image":
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
        from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper

        return QwenImagePipeline, QwenImageTransformer2DModel, xFuserQwenImageTransformerWrapper
    from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15Transformer3DModel
    from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (
        xFuserHunyuanVideo15Transformer3DWrapper,
    )

    return HunyuanVideo15Pipeline, HunyuanVideo15Transformer3DModel, xFuserHunyuanVideo15Transformer3DWrapper


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return {key: _to_cpu(item) for key, item in value.items()} if isinstance(value, dict) else value


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value.float().cpu()]
    return [t for item in value for t in _tensors(item)] if isinstance(value, (list, tuple)) else []


def _relative_l2(actual, expected):
    actual, expected = _tensors(actual), _tensors(expected)
    error = sum(((a - e) ** 2).sum().item() for a, e in zip(actual, expected))
    return (error / sum((e**2).sum().item() for e in expected)) ** 0.5


def _psnr(actual, expected):
    mse = float(((actual - expected) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(1 / mse))


def _configure(options):
    if options.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return getattr(torch, options.dtype)


def _worker(rank, world_size, role, options, init_method, output):
    """Run the pipeline as `role` ("stock", "stock-math" or "wrapper"), saving its calls and samples from rank 0."""
    from huggingface_hub import snapshot_download

    torch.cuda.set_device(rank)
    dtype = _configure(options)
    spec = _MODELS[options.model]
    checkpoint = snapshot_download(spec["checkpoint"])
    pipeline_class, _, wrapper_class = _classes(options.model)
    device = torch.device("cuda", rank)
    kwargs = dict(torch_dtype=dtype)
    if role == "wrapper":
        from xfuser.config.args import xFuserArgs
        from xfuser.core.distributed import (
            get_runtime_state,
            init_distributed_environment,
            initialize_model_parallel,
            initialize_runtime_state,
        )

        init_distributed_environment(
            rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
        )
        initialize_model_parallel(ulysses_degree=world_size, ring_degree=1)
        kwargs["transformer"] = wrapper_class.from_pretrained(checkpoint, subfolder="transformer", torch_dtype=dtype)
    pipe = pipeline_class.from_pretrained(checkpoint, **kwargs).to(device)
    pipe.set_progress_bar_config(disable=rank != 0)
    if role == "wrapper":
        args = xFuserArgs(
            model=checkpoint, ulysses_degree=world_size, ring_degree=1, attention_backend=options.attention_backend
        )
        engine_config, _ = args.create_config()
        initialize_runtime_state(pipeline=pipe, engine_config=engine_config)
        if rank == 0:
            print(f"wrapper attention backend: {get_runtime_state().attention_backend.name}", flush=True)
    video = options.model == "hunyuan-video-1.5"
    if video:
        # Decoding a batch of two needs slicing.
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    calls = []
    pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(dict(args=_to_cpu(args), kwargs=_to_cpu(kwargs))), with_kwargs=True
    )
    pipe.transformer.register_forward_hook(
        lambda module, args, kwargs, result: calls[-1].update(out=_to_cpu(result)), with_kwargs=True
    )
    prompts = list(spec["prompts"])
    if options.model == "hunyuan-video-1.5" and options.even_text:
        prompts[1] = _FOX_LONG % "cinematography"
    height, width = spec["size"]
    arguments = dict(prompt=prompts, height=height, width=width, output_type="np")
    arguments["num_inference_steps"] = options.steps or spec["steps"]
    arguments["generator"] = torch.Generator("cpu").manual_seed(options.seed)
    if options.model == "z-image":
        arguments["guidance_scale"] = 4.0
    elif options.model == "qwen-image":
        arguments.update(negative_prompt=[" "] * len(prompts), true_cfg_scale=4.0)
    else:
        arguments["num_frames"] = options.frames
    kernel = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if role == "stock-math" else None
    with torch.no_grad(), kernel or contextlib.nullcontext():
        result = pipe(**arguments)
    if rank == 0:
        samples = np.asarray(result.frames if video else result.images, dtype=np.float32)
        torch.save(dict(calls=calls, samples=samples / 255.0 if samples.max() > 1.5 else samples), output)


def _spawn(world_size, role, options, output):
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    mp.spawn(_worker, args=(world_size, role, options, f"tcp://127.0.0.1:{port}", output), nprocs=world_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(_MODELS), required=True)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--steps", type=int, help="Denoising steps; default per model")
    parser.add_argument("--frames", type=int, default=9, help="Video frames (HunyuanVideo-1.5)")
    parser.add_argument(
        "--even-text", action="store_true", help="HunyuanVideo-1.5: a 40-token prompt, sharded over two ranks"
    )
    parser.add_argument(
        "--attention-backend", help="The wrapper's attention backend, e.g. SDPA or CUDNN; default per platform"
    )
    parser.add_argument("--noise-floor", action="store_true", help="Also run stock with SDPA's math kernel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Largest same-input relative L2 error")
    parser.add_argument("--output", help="Keep the recorded calls and samples here")
    options = parser.parse_args()
    if torch.cuda.device_count() < options.ulysses_degree:
        parser.error(f"This run needs {options.ulysses_degree} accelerator devices.")

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(options.output or scratch)
        directory.mkdir(parents=True, exist_ok=True)
        _spawn(1, "stock", options, directory / "stock.pt")
        if options.noise_floor:
            _spawn(1, "stock-math", options, directory / "stock-math.pt")
        _spawn(options.ulysses_degree, "wrapper", options, directory / "wrapper.pt")
        stock = torch.load(directory / "stock.pt", weights_only=False)
        math_calls = (
            torch.load(directory / "stock-math.pt", weights_only=False)["calls"] if options.noise_floor else None
        )
        wrapper = torch.load(directory / "wrapper.pt", weights_only=False)

    assert len(stock["calls"]) == len(wrapper["calls"])
    trajectory = [_relative_l2(w["out"], s["out"]) for s, w in zip(stock["calls"], wrapper["calls"])]
    # The stock transformer on every input the wrapper saw.
    _, stock_class, _ = _classes(options.model)
    from huggingface_hub import snapshot_download

    dtype = _configure(options)
    transformer = stock_class.from_pretrained(
        snapshot_download(_MODELS[options.model]["checkpoint"]), subfolder="transformer", torch_dtype=dtype
    ).to("cuda")

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, (list, tuple)):
            return type(value)(to_device(item) for item in value)
        return {key: to_device(item) for key, item in value.items()} if isinstance(value, dict) else value

    with torch.no_grad():
        same_input = [
            _relative_l2(call["out"], transformer(*to_device(call["args"]), **to_device(call["kwargs"])))
            for call in wrapper["calls"]
        ]
    noise_floor = None
    if math_calls is not None:
        floor = [_relative_l2(m["out"], s["out"]) for s, m in zip(stock["calls"], math_calls)]
        noise_floor = dict(first=floor[0], last=floor[-1], max=max(floor))
    summary = dict(
        model=options.model,
        ulysses_degree=options.ulysses_degree,
        dtype=options.dtype,
        calls=len(trajectory),
        independent_trajectory=dict(first=trajectory[0], last=trajectory[-1], max=max(trajectory)),
        stock_math_kernel_trajectory=noise_floor,
        same_input=dict(first=same_input[0], last=same_input[-1], max=max(same_input)),
        sample_psnr_db=[_psnr(w, s) for s, w in zip(stock["samples"], wrapper["samples"])],
    )
    print(json.dumps(summary, indent=2))
    assert max(same_input) <= options.tolerance, f"same-input error {max(same_input):.2e} exceeds {options.tolerance}"


if __name__ == "__main__":
    main()
