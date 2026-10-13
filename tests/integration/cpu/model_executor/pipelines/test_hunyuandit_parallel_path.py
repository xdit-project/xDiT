"""xDiT's HunyuanDiT pipeline runs its parallel path on current diffusers.

The parallel path built the image RoPE with ``get_2d_rotary_pos_embed`` and its
default ``output_type="np"``, which diffusers refuses from 0.33.0 on, so every
CFG, Ulysses, ring or PipeFusion run raised a ValueError before its first
denoising step. The case runs that path on one CPU rank, with classifier-free
guidance and a tiny transformer, and compares it with the diffusers pipeline it
wraps.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo

TEXT_TOKENS = 6
TEXT_TOKENS_2 = 5


def _pipeline():
    from diffusers import DDIMScheduler, HunyuanDiT2DModel, HunyuanDiTPipeline

    transformer = HunyuanDiT2DModel(
        sample_size=8,
        num_layers=2,
        patch_size=2,
        attention_head_dim=8,
        num_attention_heads=3,
        in_channels=4,
        cross_attention_dim=32,
        cross_attention_dim_t5=32,
        pooled_projection_dim=16,
        hidden_size=24,
        activation_fn="gelu-approximate",
        text_len=TEXT_TOKENS,
        text_len_t5=TEXT_TOKENS_2,
    )
    return HunyuanDiTPipeline(
        vae=None,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        scheduler=DDIMScheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        text_encoder_2=None,
        tokenizer_2=None,
    )


def _call_kwargs(torch):
    generator = torch.Generator().manual_seed(1)
    kwargs = {}
    for prefix in ("", "negative_"):
        kwargs[f"{prefix}prompt_embeds"] = torch.randn(1, TEXT_TOKENS, 32, generator=generator)
        kwargs[f"{prefix}prompt_attention_mask"] = torch.ones(1, TEXT_TOKENS, dtype=torch.long)
        kwargs[f"{prefix}prompt_embeds_2"] = torch.randn(1, TEXT_TOKENS_2, 32, generator=generator)
        kwargs[f"{prefix}prompt_attention_mask_2"] = torch.ones(1, TEXT_TOKENS_2, dtype=torch.long)
    return {
        **kwargs,
        "latents": torch.randn(1, 4, 8, 8, generator=generator),
        "height": 64,
        "width": 64,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
        "use_resolution_binning": False,
        "output_type": "latent",
    }


def _worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        from unittest.mock import patch

        import torch
        import torch.distributed as dist

        import xfuser.envs as envs
        from xfuser.config.args import xFuserArgs
        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
        from xfuser.model_executor.pipelines.pipeline_hunyuandit import xFuserHunyuanDiTPipeline

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
            engine_config, _ = xFuserArgs(model="tiny", attention_backend="SDPA").create_config()
            engine_config.runtime_config.dtype = torch.float32

            torch.manual_seed(0)
            pipeline = _pipeline()
            expected = pipeline(**_call_kwargs(torch))[0]

            wrapper = xFuserHunyuanDiTPipeline(pipeline, engine_config)
            # One rank has no parallelism, so the wrapper would hand the call straight to
            # diffusers; send it down the path CFG, sequence parallelism and PipeFusion take.
            with patch.object(xFuserPipelineBaseWrapper, "use_naive_forward", lambda self: False):
                actual = wrapper(**_call_kwargs(torch))[0]

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        result_queue.put(("returned", rank, None))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_single_rank(torch, init_method, timeout=300):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_worker, args=(0, 1, init_method, result_queue))
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
def test_parallel_path_matches_diffusers(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}")

    assert result is not None, "the rank neither reported nor exited in time"
    assert result[0] == "returned", result[2]
