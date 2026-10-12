"""SDXL's CFG-parallel UNet must give the images the diffusers pipeline gives.

With ``--use_cfg_parallel`` the two CFG ranks each run part of the UNet batch and
gather the results. Diffusers stacks that batch as ``[unconditional, conditional]``
for every image of every prompt, so the split has to follow that layout for any
batch size, run the whole batch when guidance is off, and work again after the
image size changes. Two Gloo ranks run a tiny SDXL against the diffusers pipeline.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo

WORLD_SIZE = 2
PROMPT_TOKENS = 7

# name -> (guidance_scale, prompts, num_images_per_prompt, image size)
CASES = {
    "one_image": (5.0, 1, 1, 64),
    "two_images_per_prompt": (5.0, 1, 2, 64),
    "two_prompts": (5.0, 2, 1, 64),
    "smaller_image_after_larger": (5.0, 1, 1, 32),
    "guidance_off": (1.0, 1, 1, 64),
}


def _tiny_sdxl():
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,
        cross_attention_dim=64,
        norm_num_groups=1,
    )
    scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1,
        beta_schedule="scaled_linear",
        timestep_spacing="leading",
    )
    return StableDiffusionXLPipeline(
        vae=None,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        unet=unet,
        scheduler=scheduler,
    )


def _call_kwargs(torch, guidance_scale, prompts, num_images_per_prompt, size):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(prompts, PROMPT_TOKENS, 64, generator=generator),
        "pooled_prompt_embeds": torch.randn(prompts, 32, generator=generator),
        "negative_prompt_embeds": torch.randn(prompts, PROMPT_TOKENS, 64, generator=generator),
        "negative_pooled_prompt_embeds": torch.randn(prompts, 32, generator=generator),
        "latents": torch.randn(prompts * num_images_per_prompt, 4, size // 8, size // 8, generator=generator),
        "height": size,
        "width": size,
        "num_inference_steps": 2,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
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
        from xfuser.model_executor.pipelines.pipeline_stable_diffusion_xl import xFuserStableDiffusionXLPipeline

        # Torch is a CUDA build even on CPU-only machines; keep xDiT on the CPU.
        with patch.object(envs, "_is_cuda", lambda: False), patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
            parallel_state.initialize_model_parallel(backend="gloo", classifier_free_guidance_degree=world_size)
            engine_config, _ = xFuserArgs(model="tiny", use_cfg_parallel=True, attention_backend="sdpa").create_config()
            engine_config.runtime_config.dtype = torch.float32

            reference = _tiny_sdxl()
            wrapper = xFuserStableDiffusionXLPipeline(_tiny_sdxl(), engine_config)
            differences = {}
            for name, case in CASES.items():
                kwargs = _call_kwargs(torch, *case)
                expected = reference(**kwargs).images
                try:
                    actual = wrapper(**kwargs).images
                except Exception as error:  # noqa: BLE001 - report which case broke
                    differences[name] = f"{type(error).__name__}: {error}"
                    break
                differences[name] = (actual - expected).abs().max().item()
        result_queue.put(("returned", rank, differences))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_ranks(torch, init_method, timeout=300):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, WORLD_SIZE, init_method, result_queue))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + timeout
    results = []
    while len(results) < WORLD_SIZE and time.monotonic() < deadline:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    return results


@pytest.mark.slow
def test_cfg_parallel_unet_matches_diffusers(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    results = _run_ranks(torch, f"file://{tmp_path / 'dist-init'}")

    assert len(results) == WORLD_SIZE, f"only {len(results)} of {WORLD_SIZE} ranks reported: {results}"
    for status, rank, differences in results:
        assert status == "returned", differences
        assert list(differences) == list(CASES), f"rank {rank} stopped early: {differences}"
        for name, difference in differences.items():
            assert difference < 1e-3, f"rank {rank}, {name}: max difference {difference}"
