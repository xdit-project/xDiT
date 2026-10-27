"""The xFuserModel runner drives a legacy xFuserPipelineBaseWrapper under data parallelism.

Each case spawns Gloo ranks, initializes a runner around a stub legacy wrapper
with a tiny real VAE, runs it, and checks that the last rank, which saves the
outputs, receives one image per prompt, decoded from that prompt's latents.
"""

import traceback
from functools import partial
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


def _worker(rank, world_size, init_method, result_queue, *, degrees, use_parallel_vae):
    dist = None
    try:
        import os

        # The runner's log() reads the rank from the launcher's environment.
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size))
        import torch
        import torch.distributed as dist

        from xfuser import xFuserArgs
        from xfuser.config import config as config_module
        from xfuser.core.distributed import parallel_state, runtime_state
        from xfuser.core.distributed.parallel_state import (
            get_pipeline_parallel_rank,
            get_pipeline_parallel_world_size,
            is_dp_last_group,
        )
        from xfuser.core.distributed.runtime_state import DiTRuntimeState
        from xfuser.model_executor.models.runner_models.base_model import (
            DiffusionOutput,
            ModelCapabilities,
            ModelSettings,
            xFuserModel,
        )
        from xfuser.model_executor.pipelines import base_pipeline
        from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper

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
            settings = ModelSettings(model_name="test/model", model_output_type="image")
            capabilities = ModelCapabilities(pipefusion_parallel_degree=True, use_parallel_vae=True)

            def _load_model(self):
                raise NotImplementedError

            def _post_load_and_state_initialization(self, input_args):
                """Skip placing the pipeline on an accelerator."""

            def _load_model_checked(self):
                # What the wrapper's constructor does: build the runtime state, then shard the decoder.
                state = DiTRuntimeState.__new__(DiTRuntimeState)
                state.runtime_config = self.engine_config.runtime_config
                state.parallel_config = self.engine_config.parallel_config
                runtime_state._RUNTIME = state
                pipe = _LegacyPipeline.__new__(_LegacyPipeline)
                vae = _tiny_vae()
                if (
                    use_parallel_vae
                    and not pipe.use_naive_forward()
                    and not self.engine_config.runtime_config.runner_managed_parallel_vae
                ):
                    vae = pipe._convert_vae(vae)
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
            # Only the optional yunchang package is missing; no attention runs here.
            patch.object(config_module, "HAS_LONG_CTX_ATTN", True),
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
                sequence_parallel_degree=degrees["ulysses"],
                ulysses_degree=degrees["ulysses"],
                pipeline_parallel_degree=degrees["pp"],
                use_parallel_vae=use_parallel_vae,
            )
            args = xFuserArgs(
                model="test/model",
                data_parallel_degree=degrees["dp"],
                ulysses_degree=degrees["ulysses"],
                pipefusion_parallel_degree=degrees["pp"],
                use_parallel_vae=use_parallel_vae,
                output_directory=str(os.path.dirname(init_method.removeprefix("file://"))),
            )
            runner = _Runner(args)
            runner.initialize({})

            # A finite elapsed time keeps run()'s timing log formattable.
            torch.cuda.Event.return_value.elapsed_time.return_value = 1.0
            input_args = {"prompt": list(_PROMPTS), "dataset_path": None, "height": 64, "width": 64}
            output, _ = runner.run(input_args)

        # Plain arrays: tensors would be shared through memory this process frees on exit.
        images = None if output is None else [image.numpy() for image in output.images]
        result_queue.put(("returned", rank, images))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "dp, ulysses, pp, use_parallel_vae",
    [
        pytest.param(2, 2, 1, False, id="dp2-ulysses2"),
        pytest.param(2, 2, 1, True, id="dp2-ulysses2-parallel-vae"),
        pytest.param(1, 1, 2, True, id="pipefusion2-parallel-vae"),
    ],
)
def test_last_rank_receives_one_image_per_prompt(tmp_path, run_spawned, dp, ulysses, pp, use_parallel_vae):
    torch = pytest.importorskip("torch")
    pytest.importorskip("distvae")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    world_size = dp * ulysses * pp
    degrees = {"dp": dp, "ulysses": ulysses, "pp": pp}
    processes, hung, _, results = run_spawned(
        torch,
        partial(_worker, degrees=degrees, use_parallel_vae=use_parallel_vae),
        f"file://{tmp_path / 'legacy-wrapper-init'}",
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
