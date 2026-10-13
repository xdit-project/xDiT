"""xDiT's parallel pipeline paths must run with classifier-free guidance turned off.

``guidance_scale <= 1`` disables CFG: the prompt embeddings then hold only the
conditional batch. PixArt-alpha, PixArt-Sigma and Sana nevertheless doubled the latent
batch on their xDiT path (taken under sequence parallelism or PipeFusion), so the
transformer saw twice as many latents as prompts and crashed. Each case runs the xDiT
path on one CPU rank and compares it with the diffusers pipeline it wraps.
"""

import importlib
import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo

PROMPT_TOKENS = 6


def _pixart(sigma):
    from diffusers import DDIMScheduler, PixArtAlphaPipeline, PixArtSigmaPipeline, PixArtTransformer2DModel

    transformer = PixArtTransformer2DModel(
        sample_size=8,
        num_layers=2,
        patch_size=2,
        attention_head_dim=8,
        num_attention_heads=3,
        caption_channels=32,
        in_channels=4,
        cross_attention_dim=24,
        out_channels=8,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )
    pipeline_cls = PixArtSigmaPipeline if sigma else PixArtAlphaPipeline
    pipeline = pipeline_cls(
        tokenizer=None, text_encoder=None, vae=None, transformer=transformer, scheduler=DDIMScheduler()
    )
    return pipeline, {"caption_channels": 32, "height": 64, "width": 64}


def _sana():
    from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel

    transformer = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        sample_size=8,
        patch_size=1,
    )
    pipeline = SanaPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    return pipeline, {"caption_channels": 8, "height": 256, "width": 256}


PIPELINES = {
    "pixart_alpha": (lambda: _pixart(sigma=False), "pipeline_pixart_alpha", "xFuserPixArtAlphaPipeline"),
    "pixart_sigma": (lambda: _pixart(sigma=True), "pipeline_pixart_sigma", "xFuserPixArtSigmaPipeline"),
    "sana": (_sana, "pipeline_sana", "xFuserSanaPipeline"),
}


def _call_kwargs(torch, caption_channels, height, width, num_images_per_prompt):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(1, PROMPT_TOKENS, caption_channels, generator=generator),
        "prompt_attention_mask": torch.ones(1, PROMPT_TOKENS, dtype=torch.long),
        "negative_prompt": None,
        "latents": torch.randn(num_images_per_prompt, 4, 8, 8, generator=generator),
        "height": height,
        "width": width,
        "num_inference_steps": 2,
        "num_images_per_prompt": num_images_per_prompt,
        "guidance_scale": 1.0,
        "use_resolution_binning": False,
        "output_type": "latent",
    }


def _worker(rank, world_size, init_method, result_queue, name, num_images_per_prompt):
    dist = None
    try:
        from unittest.mock import patch

        import torch
        import torch.distributed as dist

        import xfuser.envs as envs
        from xfuser.config.args import xFuserArgs
        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper

        build, module_name, wrapper_name = PIPELINES[name]
        # Torch is a CUDA build even on CPU-only machines; keep xDiT on the CPU.
        with patch.object(envs, "_is_cuda", lambda: False), patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
            parallel_state.initialize_model_parallel(backend="gloo")
            engine_config, _ = xFuserArgs(model="tiny", attention_backend="sdpa").create_config()
            engine_config.runtime_config.dtype = torch.float32

            torch.manual_seed(0)
            pipeline, shape = build()
            kwargs = _call_kwargs(torch, num_images_per_prompt=num_images_per_prompt, **shape)
            expected = pipeline(**kwargs)[0]

            wrapper_cls = getattr(
                importlib.import_module(f"xfuser.model_executor.pipelines.{module_name}"), wrapper_name
            )
            wrapper = wrapper_cls(pipeline, engine_config)
            # One rank has no parallelism, so the wrapper would hand the call straight to
            # diffusers; send it down the path sequence parallelism and PipeFusion take.
            with patch.object(xFuserPipelineBaseWrapper, "use_naive_forward", lambda self: False):
                actual = wrapper(**kwargs)[0]

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        result_queue.put(("returned", rank, None))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_single_rank(torch, init_method, args, timeout=300):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_worker, args=(0, 1, init_method, result_queue, *args))
    process.start()
    deadline = time.monotonic() + timeout
    result = None
    while result is None and time.monotonic() < deadline:
        try:
            result = result_queue.get(timeout=1)
        except queue.Empty:
            if not process.is_alive():
                break
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    return result


@pytest.mark.slow
@pytest.mark.parametrize("num_images_per_prompt", [1, 2])
@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_parallel_path_matches_diffusers_without_cfg(tmp_path, name, num_images_per_prompt):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}", (name, num_images_per_prompt))

    assert result is not None, "the rank neither reported nor exited in time"
    assert result[0] == "returned", result[2]
