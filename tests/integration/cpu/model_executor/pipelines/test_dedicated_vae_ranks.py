"""Spawned regression: dedicated VAE ranks decode latents received from the DiT ranks.

Rank 0 plays the last DiT rank and sends latents the way ``send_to_vae_decode`` does.
Ranks 1 and 2 form the VAE group and run ``xFuserVAEWrapper.execute`` with a DistVAE
row-parallel decoder, exactly as the Ray VAE worker calls it (outside any grad context).
"""

import queue
import time
import traceback
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_DIT_SIZE = 1
_VAE_SIZE = 2
_WORLD_SIZE = _DIT_SIZE + _VAE_SIZE
_LATENT_SHAPE = (1, 4, 8, 8)


def _tiny_vae(torch):
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=16,
    ).eval()


def _latents(torch):
    generator = torch.Generator().manual_seed(1)
    return torch.randn(_LATENT_SHAPE, generator=generator)


def _send_latents(torch, dist, latents, dst):
    dist.send(torch.tensor([latents.dim()], dtype=torch.int), dst=dst)
    dist.send(torch.tensor(latents.shape, dtype=torch.int), dst=dst)
    dist.send(latents, dst=dst)


def _worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist
        from diffusers.image_processor import VaeImageProcessor

        dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=world_size)
        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.pipelines import base_pipeline

        parallel_state.init_vae_group(_DIT_SIZE, _VAE_SIZE, "gloo")
        latents = _latents(torch)
        if rank < _DIT_SIZE:
            _send_latents(torch, dist, latents, dst=_DIT_SIZE)
            result_queue.put(("sent", rank, None))
            return

        vae = _tiny_vae(torch)
        image_processor = VaeImageProcessor(vae_scale_factor=2)
        with torch.no_grad():
            expected = image_processor.postprocess(vae.decode(latents, return_dict=False)[0], output_type="pt")

        world_group = SimpleNamespace(rank=rank, local_rank=rank)
        dit_config = SimpleNamespace(pp_degree=1, sp_degree=_DIT_SIZE, cfg_degree=1, dp_degree=1, tp_degree=1)
        engine_config = SimpleNamespace(runtime_config=SimpleNamespace(dtype=torch.float32))
        with (
            mock.patch.object(base_pipeline, "get_world_group", return_value=world_group),
            mock.patch.object(base_pipeline, "get_device", return_value=torch.device("cpu")),
        ):
            wrapper = base_pipeline.xFuserVAEWrapper(
                vae,
                engine_config=engine_config,
                dit_parallel_config=dit_config,
                use_parallel=True,
                image_processor=image_processor,
            )
            image = wrapper.execute(output_type="pt")

        max_error = (image.float() - expected.float()).abs().max().item()
        result_queue.put(("decoded", rank, (tuple(image.shape), tuple(expected.shape), max_error)))
    except Exception:  # report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, _WORLD_SIZE, init_method, result_queue))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    results = []
    deadline = time.monotonic() + timeout
    while len(results) < _WORLD_SIZE and time.monotonic() < deadline:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)
    return processes, hung, results


def test_dedicated_vae_ranks_decode_like_the_plain_vae(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("distvae")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(torch, f"file://{tmp_path / 'vae-ranks-init'}", timeout=120)

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(error[2] for error in errors)
    assert not hung, f"VAE decode hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    decoded = {rank: facts for kind, rank, facts in results if kind == "decoded"}
    assert sorted(decoded) == [1, 2]
    for image_shape, expected_shape, max_error in decoded.values():
        assert image_shape == expected_shape == (1, 3, 16, 16)
        assert max_error < 1e-4
