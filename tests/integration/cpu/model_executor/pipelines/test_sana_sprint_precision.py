"""xDiT's Sana Sprint loop must compute the SCM timestep in float32 like diffusers.

Diffusers keeps the timestep and the SCM input scaling in float32 and casts only
the transformer inputs to the transformer's dtype. xDiT's loop, which sequence
parallelism takes, cast the timestep to the prompt dtype first, so a bfloat16
pipeline denoised with rounded timesteps. One CPU rank compares the xDiT loop
with diffusers in bfloat16.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo


def _tiny_sana_sprint(torch):
    from diffusers import SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler

    torch.manual_seed(0)
    transformer = SanaTransformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        sample_size=32,
        qk_norm="rms_norm_across_heads",
        guidance_embeds=True,
    ).to(torch.bfloat16)
    return SanaSprintPipeline(
        tokenizer=None, text_encoder=None, vae=None, transformer=transformer, scheduler=SCMScheduler()
    )


def _call_kwargs(torch):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(1, 6, 8, generator=generator).to(torch.bfloat16),
        "prompt_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "latents": torch.randn(1, 4, 8, 8, generator=generator),
        "height": 256,
        "width": 256,
        "num_inference_steps": 2,
        "output_type": "latent",
        "use_resolution_binning": False,
        "generator": torch.Generator().manual_seed(3),
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
        from xfuser.model_executor.pipelines.pipeline_sana_sprint import xFuserSanaSprintPipeline

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
            engine_config.runtime_config.dtype = torch.bfloat16

            expected = _tiny_sana_sprint(torch)(**_call_kwargs(torch))[0]
            wrapper = xFuserSanaSprintPipeline(_tiny_sana_sprint(torch), engine_config)
            # One rank has no parallelism, so the wrapper would hand the call straight
            # to diffusers; send it down the path sequence parallelism takes.
            with patch.object(xFuserPipelineBaseWrapper, "use_naive_forward", lambda self: False):
                actual = wrapper(**_call_kwargs(torch))[0]

        result_queue.put(("returned", rank, (actual - expected).abs().max().item()))
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
def test_sana_sprint_bfloat16_loop_matches_diffusers(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}")

    assert result is not None, "the rank neither reported nor exited in time"
    status, _, difference = result
    assert status == "returned", difference
    # Same kernels, same dtype: anything but rounding noise means the loops differ.
    assert difference < 1e-5, f"max difference {difference}"
