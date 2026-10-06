"""xDiT's StableDiffusion3 wrapper must honour the call arguments diffusers takes.

The wrapper runs its own denoising loop under sequence parallelism, PipeFusion
and CFG parallelism, and hands the call to diffusers otherwise. Its loop took
any keyword through ``**kwargs`` and dropped it, so ``sigmas``, ``mu`` and
``max_sequence_length`` changed nothing and skip-layer guidance was silently
off. Its own ``timesteps`` argument, which diffusers does not take, raised once
the call reached diffusers. One CPU rank compares the wrapper with the diffusers
pipeline it wraps.

Naive calls must also expose the original scheduler to diffusers and user
callbacks, then restore xDiT's scheduler for subsequent parallel calls.
"""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo

PROMPT_TOKENS = 7


def _tiny_sd3(dynamic_shifting=False):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline

    torch.manual_seed(0)
    transformer = SD3Transformer2DModel(
        sample_size=16,
        patch_size=2,
        in_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        caption_projection_dim=32,
        joint_attention_dim=32,
        pooled_projection_dim=64,
        out_channels=4,
    )
    scheduler = (
        FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        if dynamic_shifting
        else FlowMatchEulerDiscreteScheduler(shift=3.0)
    )
    return StableDiffusion3Pipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        text_encoder_3=None,
        tokenizer_3=None,
    )


def _call_kwargs(torch, num_images_per_prompt=1, **overrides):
    generator = torch.Generator().manual_seed(1)
    kwargs = {
        "prompt_embeds": torch.randn(1, PROMPT_TOKENS, 32, generator=generator),
        "pooled_prompt_embeds": torch.randn(1, 64, generator=generator),
        "negative_prompt_embeds": torch.randn(1, PROMPT_TOKENS, 32, generator=generator),
        "negative_pooled_prompt_embeds": torch.randn(1, 64, generator=generator),
        "latents": torch.randn(num_images_per_prompt, 4, 16, 16, generator=generator),
        "height": 128,
        "width": 128,
        "num_inference_steps": 3,
        "guidance_scale": 5.0,
        "num_images_per_prompt": num_images_per_prompt,
        "output_type": "latent",
    }
    kwargs.update(overrides)
    return kwargs


# name -> (dynamic shifting scheduler, call overrides)
MATCHES_DIFFUSERS = {
    "sigmas": (False, {"sigmas": [1.0, 0.7, 0.4]}),
    # More sigmas than ``num_inference_steps``: the sigmas set the step count.
    "sigmas_longer_than_num_inference_steps": (False, {"sigmas": [1.0, 0.85, 0.7, 0.55, 0.4]}),
    "dynamic_shifting_default_mu": (True, {}),
    "dynamic_shifting_explicit_mu": (True, {"mu": 0.3}),
    "guidance_off": (False, {"guidance_scale": 1.0}),
}

NAIVE_CALLS = {
    "naive_forward": {},
    "naive_forward_explicit_no_timesteps": {"timesteps": None},
}

# name -> (call overrides, exception the parallel path must raise)
REJECTED = {
    "max_sequence_length_over_512": ({"max_sequence_length": 600}, "ValueError"),
    "skip_layer_guidance": ({"skip_guidance_layers": [0]}, "NotImplementedError"),
}


def _worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        from unittest.mock import patch

        import torch
        import torch.distributed as dist

        import xfuser.envs as envs
        from xfuser.config.args import xFuserArgs
        from xfuser.core.distributed import get_runtime_state, parallel_state
        from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
        from xfuser.model_executor.pipelines.pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline

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

            def wrap(dynamic_shifting=False):
                return xFuserStableDiffusion3Pipeline(_tiny_sd3(dynamic_shifting), engine_config)

            for name, overrides in NAIVE_CALLS.items():
                pipeline = _tiny_sd3()
                original_scheduler = pipeline.scheduler
                wrapped = xFuserStableDiffusion3Pipeline(pipeline, engine_config)
                wrapped_scheduler = pipeline.scheduler
                callback_steps = []

                def check_scheduler(callback_pipeline, step, timestep, callback_kwargs):
                    assert callback_pipeline.scheduler is original_scheduler, "callback received xDiT's scheduler"
                    callback_steps.append(step)
                    return callback_kwargs

                try:
                    expected = _tiny_sd3()(**_call_kwargs(torch)).images
                    actual = wrapped(**_call_kwargs(torch, callback_on_step_end=check_scheduler, **overrides)).images
                    assert callback_steps, "the scheduler check never ran"
                    assert pipeline.scheduler is wrapped_scheduler, "xDiT's scheduler was not restored"
                    results[name] = (actual - expected).abs().max().item()
                except Exception as error:  # noqa: BLE001
                    results[name] = f"{type(error).__name__}: {error}"

            pipeline = _tiny_sd3()
            original_scheduler = pipeline.scheduler
            wrapped = xFuserStableDiffusion3Pipeline(pipeline, engine_config)
            wrapped_scheduler = pipeline.scheduler

            def fail_callback(callback_pipeline, step, timestep, callback_kwargs):
                assert callback_pipeline.scheduler is original_scheduler, "callback received xDiT's scheduler"
                raise RuntimeError("callback failed")

            try:
                with pytest.raises(RuntimeError, match="^callback failed$"):
                    wrapped(**_call_kwargs(torch, callback_on_step_end=fail_callback))
                assert pipeline.scheduler is wrapped_scheduler, "xDiT's scheduler was not restored after an error"
                results["naive_callback_error"] = "raised"
            except Exception as error:  # noqa: BLE001
                results["naive_callback_error"] = f"{type(error).__name__}: {error}"

            # One rank has no parallelism, so the wrapper hands calls to diffusers
            # unless they need its own loop; ``timesteps`` is such a call. Diffusers
            # takes no ``timesteps``, but the same schedule as sigmas (t / 1000).
            try:
                expected = _tiny_sd3()(**_call_kwargs(torch, sigmas=[1.0, 0.8, 0.6])).images
                actual = wrap()(**_call_kwargs(torch, timesteps=[1000.0, 800.0, 600.0])).images
                results["timesteps_without_parallelism"] = (actual - expected).abs().max().item()
            except Exception as error:  # noqa: BLE001 - report which case broke
                results["timesteps_without_parallelism"] = f"{type(error).__name__}: {error}"

            # Send the remaining calls down the path parallel runs take.
            with patch.object(xFuserPipelineBaseWrapper, "use_naive_forward", lambda self: False):
                for name, (dynamic_shifting, overrides) in MATCHES_DIFFUSERS.items():
                    kwargs = _call_kwargs(torch, **overrides)
                    expected = _tiny_sd3(dynamic_shifting)(**kwargs).images
                    try:
                        actual = wrap(dynamic_shifting)(**kwargs).images
                        results[name] = (actual - expected).abs().max().item()
                    except Exception as error:  # noqa: BLE001
                        results[name] = f"{type(error).__name__}: {error}"
                    if "sigmas" in overrides:
                        results[f"{name}_runtime_steps"] = get_runtime_state().input_config.num_inference_steps
                for name, (overrides, expected_error) in REJECTED.items():
                    try:
                        wrap()(**_call_kwargs(torch, **overrides))
                        results[name] = "no error"
                    except Exception as error:  # noqa: BLE001
                        results[name] = "raised" if type(error).__name__ == expected_error else repr(error)
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
def test_sd3_wrapper_honours_diffusers_call_arguments(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    result = _run_single_rank(torch, f"file://{tmp_path / 'dist-init'}")

    assert result is not None, "the rank neither reported nor exited in time"
    status, _, results = result
    assert status == "returned", results
    failures = {}
    for name in [*NAIVE_CALLS, "timesteps_without_parallelism", *MATCHES_DIFFUSERS]:
        if not isinstance(results[name], float) or results[name] > 1e-4:
            failures[name] = results[name]
        overrides = MATCHES_DIFFUSERS.get(name, (None, {}))[1]
        if "sigmas" in overrides and results[f"{name}_runtime_steps"] != len(overrides["sigmas"]):
            failures[f"{name}_runtime_steps"] = results[f"{name}_runtime_steps"]
    for name in ["naive_callback_error", *REJECTED]:
        if results[name] != "raised":
            failures[name] = results[name]
    assert not failures, failures
