"""Without parallelism the wrapper must run diffusers with the scheduler diffusers expects.

xDiT wraps the pipeline's scheduler for its parallel paths, and with no
parallelism enabled it hands the call to the diffusers pipeline. ConsisID (and
CogVideoX) decide how to call ``scheduler.step`` with ``isinstance(self.scheduler,
CogVideoXDPMScheduler)``, which the wrapper fails, so the single-GPU run called
the DPM scheduler with the DDIM argument order. One CPU rank compares the
wrapper with the diffusers pipeline on a tiny ConsisID.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo


def _tiny_consisid():
    import torch
    from diffusers import (
        AutoencoderKLCogVideoX,
        CogVideoXDPMScheduler,
        ConsisIDPipeline,
        ConsisIDTransformer3DModel,
    )

    torch.manual_seed(0)
    transformer = ConsisIDTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=8,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=32,
        num_layers=1,
        sample_width=2,
        sample_height=2,
        sample_frames=9,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=16,
        use_rotary_positional_embeddings=True,
        use_learned_positional_embeddings=True,
        cross_attn_interval=1,
        is_kps=False,
        is_train_face=True,
        cross_attn_dim_head=1,
        cross_attn_num_heads=1,
        LFE_id_dim=2,
        LFE_vit_dim=2,
        LFE_depth=5,
        LFE_dim_head=8,
        LFE_num_heads=2,
        LFE_num_id_token=1,
        LFE_num_querie=1,
        LFE_output_dim=21,
        LFE_ff_mult=1,
        LFE_num_scale=1,
    )
    torch.manual_seed(0)
    vae = AutoencoderKLCogVideoX(
        in_channels=3,
        out_channels=3,
        down_block_types=("CogVideoXDownBlock3D",) * 4,
        up_block_types=("CogVideoXUpBlock3D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=2,
        temporal_compression_ratio=4,
    )
    return ConsisIDPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=CogVideoXDPMScheduler(),
    )


def _call_kwargs(torch):
    from PIL import Image

    generator = torch.Generator().manual_seed(1)
    return {
        "image": Image.new("RGB", (16, 16)),
        "prompt_embeds": torch.randn(1, 16, 32, generator=generator),
        "negative_prompt_embeds": torch.randn(1, 16, 32, generator=generator),
        "id_vit_hidden": [torch.ones([1, 2, 2])],
        "id_cond": torch.ones(1, 2),
        "height": 16,
        "width": 16,
        "num_frames": 8,
        "num_inference_steps": 3,
        "guidance_scale": 6.0,
        "max_sequence_length": 16,
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
        from xfuser.model_executor.pipelines.pipeline_consisid import xFuserConsisIDPipeline

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

            expected = _tiny_consisid()(**_call_kwargs(torch)).frames
            actual = xFuserConsisIDPipeline(_tiny_consisid(), engine_config)(**_call_kwargs(torch)).frames

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
def test_wrapper_without_parallelism_matches_diffusers(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2", reason="ConsisIDPipeline requires OpenCV")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}")

    assert result is not None, "the rank neither reported nor exited in time"
    status, _, difference = result
    assert status == "returned", difference
    assert difference < 1e-5, f"max difference {difference}"
