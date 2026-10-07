"""The xFuserModel runner drives a legacy xFuserPipelineBaseWrapper under data parallelism.

Each case spawns Gloo ranks, initializes a runner around a stub legacy wrapper
with a tiny real VAE, runs it, and checks that the last rank, which saves the
outputs, receives one image per prompt, decoded from that prompt's latents.
"""

import queue
import time
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_PROMPTS = ["a", "b", "c", "d"]


def _tiny_vae():
    import torch
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        block_out_channels=(32,),
        layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        latent_channels=4,
        norm_num_groups=32,
        sample_size=16,
    ).eval()


def _latents(prompts):
    """Return one distinct latent per prompt, so each image identifies its prompt."""
    import torch

    return torch.stack([torch.full((4, 8, 8), float(_PROMPTS.index(p) + 1)) / 4 for p in prompts])


def _worker(rank, world_size, init_method, result_queue, degrees):
    dist = None
    try:
        import os

        # The runner's log() reads the rank from the launcher's environment.
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size))
        import torch
        import torch.distributed as dist

        from xfuser.config.config import RuntimeConfig
        from xfuser.core.distributed import parallel_state, runtime_state
        from xfuser.core.distributed.parallel_state import (
            get_pipeline_parallel_rank,
            get_pipeline_parallel_world_size,
            is_dp_last_group,
        )
        from xfuser.core.distributed.runtime_state import DiTRuntimeState
        from xfuser.model_executor.models.runner_models import base_model
        from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput, xFuserModel
        from xfuser.model_executor.models.runner_models.vae_manager import VAEManager
        from xfuser.model_executor.pipelines import base_pipeline
        from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper

        use_parallel_vae = degrees["use_parallel_vae"]

        class _LegacyPipeline(xFuserPipelineBaseWrapper):
            """Follows the decode and output flow of the SD3/FLUX.1 wrappers."""

            @xFuserPipelineBaseWrapper.enable_data_parallel
            def __call__(self, prompt=None):
                latents = _latents(prompt)
                if get_pipeline_parallel_rank() != get_pipeline_parallel_world_size() - 1:
                    latents = None  # earlier PipeFusion stages hold no final latents
                image = None
                if use_parallel_vae:
                    latents = self.gather_broadcast_latents(latents)
                    image = self.vae.decode(latents, return_dict=False)[0]
                elif is_dp_last_group():
                    image = self.vae.decode(latents, return_dict=False)[0]
                if self.is_dp_last_group():
                    return SimpleNamespace(images=list(image))
                return None

        class _Runner(xFuserModel):
            def _load_model(self):
                raise NotImplementedError

            def _load_model_checked(self):
                pipe = _LegacyPipeline.__new__(_LegacyPipeline)
                pipe.engine_config = SimpleNamespace(runtime_config=RuntimeConfig(use_parallel_vae=use_parallel_vae))
                vae = _tiny_vae()
                if use_parallel_vae and not pipe.use_naive_forward():
                    vae = pipe._convert_vae(vae)  # what the wrapper's constructor does
                pipe.module = SimpleNamespace(vae=vae)
                return pipe

            def _run_pipe(self, input_args):
                output = self.pipe(prompt=input_args["prompt"])
                images = output.images if output else []
                return DiffusionOutput(images=images, pipe_args=input_args)

        cpu = torch.device("cpu")
        with (
            patch.object(parallel_state, "set_device"),
            patch.object(base_pipeline, "get_device", lambda local_rank: cpu),
            patch.object(torch.cuda, "Event", Mock()),
            patch.object(torch.cuda, "synchronize", lambda: None),
            patch.object(base_model, "initialize_runtime_state", lambda *args: None),
            torch.no_grad(),
        ):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
            parallel_state.initialize_model_parallel(
                backend="gloo",
                data_parallel_degree=degrees["dp"],
                classifier_free_guidance_degree=degrees["cfg"],
                sequence_parallel_degree=degrees["ulysses"],
                ulysses_degree=degrees["ulysses"],
                pipeline_parallel_degree=degrees["pp"],
                use_parallel_vae=use_parallel_vae,
            )
            state = DiTRuntimeState.__new__(DiTRuntimeState)
            # The runner may load the pipeline in a dtype other than the engine
            # config's default, as SD3.5 does (bfloat16 against float16).
            state.runtime_config = RuntimeConfig(dtype=torch.float16, use_parallel_vae=use_parallel_vae)
            state.parallel_config = SimpleNamespace(dp_degree=degrees["dp"], vae_parallel_size=0)
            runtime_state._RUNTIME = state

            runner = object.__new__(_Runner)
            runner.config = SimpleNamespace(
                data_parallel_degree=degrees["dp"],
                use_parallel_vae=use_parallel_vae,
                use_torch_compile=False,
                cache_method=None,
                enable_tiling=False,
                num_iterations=1,
                warmup_calls=0,
                determinism_check=0,
                batch_size=None,
                create_config=lambda: (None, None),
            )
            runner.settings = SimpleNamespace(model_output_type="image")
            runner.loader = Mock()
            runner._vae_manager = VAEManager(
                config=runner.config,
                capabilities=SimpleNamespace(use_parallel_vae_encoder=False),
                settings=runner.settings,
            )
            runner._post_load_and_state_initialization = lambda input_args: None
            runner._enable_options = lambda: None
            runner._validate_args = lambda input_args: None
            runner.initialize({})

            # A finite elapsed time keeps run()'s timing log formattable.
            torch.cuda.Event.return_value.elapsed_time.return_value = 1.0
            output, _ = runner.run({"prompt": list(_PROMPTS)})

        # Plain arrays: tensors would be shared through memory this process frees on exit.
        images = None if output is None else [image.numpy() for image in output.images]
        result_queue.put(("returned", rank, images))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, degrees, *, world_size, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, init_method, result_queue, degrees))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    results = []
    deadline = time.monotonic() + timeout
    while len(results) < world_size and time.monotonic() < deadline:
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


@pytest.mark.parametrize(
    "dp, cfg, ulysses, pp, use_parallel_vae",
    [
        pytest.param(2, 1, 1, 1, False, id="dp2"),
        pytest.param(2, 1, 2, 1, False, id="dp2-ulysses2"),
        pytest.param(2, 2, 1, 1, False, id="dp2-cfg2"),
        pytest.param(1, 1, 2, 1, True, id="ulysses2-parallel-vae"),
        pytest.param(2, 1, 2, 1, True, id="dp2-ulysses2-parallel-vae"),
        pytest.param(1, 1, 1, 2, True, id="pipefusion2-parallel-vae"),
    ],
)
def test_last_rank_receives_one_image_per_prompt(tmp_path, dp, cfg, ulysses, pp, use_parallel_vae):
    torch = pytest.importorskip("torch")
    pytest.importorskip("distvae")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    world_size = dp * cfg * ulysses * pp
    degrees = {"dp": dp, "cfg": cfg, "ulysses": ulysses, "pp": pp, "use_parallel_vae": use_parallel_vae}
    processes, hung, results = _run_spawned(
        torch,
        f"file://{tmp_path / 'legacy-wrapper-init'}",
        degrees,
        world_size=world_size,
        timeout=180,
    )

    assert not hung, f"ranks hung; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * world_size
    images = {rank: images for _, rank, images in results}

    # Each DP group runs the prompts the runner gave it; the last rank concatenates the groups.
    expected_order = [prompt for group in range(dp) for prompt in _PROMPTS[group::dp]]
    with torch.no_grad():
        expected = _tiny_vae().decode(_latents(expected_order), return_dict=False)[0]
    last = images[world_size - 1]
    assert last is not None and len(last) == len(_PROMPTS)
    torch.testing.assert_close(torch.stack([torch.from_numpy(image) for image in last]), expected, atol=1e-5, rtol=1e-5)
