"""xDiT's FLUX.2 loop must not drop classifier-free guidance for klein base checkpoints.

Diffusers runs an unconditional branch for FLUX.2 klein checkpoints that are not
step-distilled (the "base" ones) whenever guidance_scale > 1. The loop xDiT runs
under PipeFusion and sequence parallelism has no such branch, so it returned an
unguided image without a word. It must refuse those calls, while step-distilled
klein, for which diffusers ignores the guidance scale, keeps matching diffusers.
One CPU rank runs the loop on a tiny klein model.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo


def _tiny_klein(is_distilled):
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )

    torch.manual_seed(0)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=256,
        axes_dims_rope=[4, 4, 4, 4],
        guidance_embeds=False,
    )
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    return Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        is_distilled=is_distilled,
    )


def _call_kwargs(torch):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(1, 8, 16, generator=generator),
        "height": 16,
        "width": 16,
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "generator": torch.Generator().manual_seed(0),
        "output_type": "pt",
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
        from xfuser.model_executor.pipelines.pipeline_flux2_klein import xFuserFlux2KleinPipeline

        results = {}
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

            # One rank has no parallelism, so the wrapper would hand the call straight
            # to diffusers; send it down the loop PipeFusion and sequence parallelism take.
            with patch.object(xFuserPipelineBaseWrapper, "use_naive_forward", lambda self: False):
                try:
                    xFuserFlux2KleinPipeline(_tiny_klein(is_distilled=False), engine_config)(**_call_kwargs(torch))
                    results["base"] = "returned an image"
                except NotImplementedError:
                    results["base"] = "refused"

                expected = _tiny_klein(is_distilled=True)(**_call_kwargs(torch)).images
                wrapper = xFuserFlux2KleinPipeline(_tiny_klein(is_distilled=True), engine_config)
                actual = wrapper(**_call_kwargs(torch)).images
                results["distilled"] = (actual - expected).abs().max().item()
        result_queue.put(("returned", rank, results))
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
def test_klein_base_guidance_is_refused_and_distilled_klein_matches_diffusers(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}")

    assert result is not None, "the rank neither reported nor exited in time"
    status, _, results = result
    assert status == "returned", results
    assert results["base"] == "refused"
    assert results["distilled"] < 1e-4, results
