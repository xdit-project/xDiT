"""Compare a complete, locally available pretrained SD3 pipeline with diffusers.

Run explicitly with ``SD3_E2E_MODEL=/path/to/checkpoint python -m pytest
tests/e2e/model_executor/pipelines/test_sd3_pretrained.py -s``. No checkpoint is
downloaded. ``SD3_E2E_OUTPUT_DIR`` retains PNGs, tensors, logs and JSON reports;
``SD3_E2E_MODES`` selects comma-separated stock, naive, ulysses, cfg and
pipefusion_sync modes. Stock always runs first in its own process.

For a published derivative, set ``SD3_E2E_MODEL_CLASSIFICATION`` to
``full_pretrained_derivative`` and record its repository and pinned commit in
``SD3_E2E_MODEL_ID`` / ``SD3_E2E_MODEL_REVISION``. Model-card inference settings
can be supplied with ``SD3_E2E_QUALITY_STEPS`` and
``SD3_E2E_GUIDANCE_SCALE`` (defaults: 20 and 5). Small checkpoints require
``SD3_E2E_ALLOW_TINY=1``; their small runs are labelled tiny_preflight and cannot
be reported as full-model E2E validation.

Numerical regression defaults to float32 with TF32 disabled. Set
``SD3_E2E_DTYPE=bfloat16`` to investigate production-precision drift; the
retained metrics and images still use the same strict comparison assertions.
"""

import contextlib
import json
import os
from pathlib import Path
import time
import traceback

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.accelerator, pytest.mark.multi_gpu, pytest.mark.slow]

_MODES = {"stock": 1, "naive": 1, "ulysses": 2, "cfg": 2, "pipefusion_sync": 2}
_PROMPT = "A red fox sitting beside a clear mountain lake, pine trees, warm morning sunlight, detailed photograph"
_NEGATIVE_PROMPT = "blurry, distorted, low quality, text, watermark"
_PARALLEL_LIMITS = {"relative_l2": 1e-4, "cosine_min": 0.99999999, "image_mae": 1e-4}
_NAIVE_LIMITS = {"relative_l2": 1e-5, "cosine_min": 0.999999, "image_mae": 1e-5}


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cases(quality_steps):
    return [
        {"name": "quality", "num_inference_steps": quality_steps},
        {"name": "sigmas_3_requested_2", "num_inference_steps": 2, "sigmas": [1.0, 0.7, 0.4], "dynamic": False},
        {
            "name": "sigmas_5_requested_3",
            "num_inference_steps": 3,
            "sigmas": [1.0, 0.85, 0.7, 0.55, 0.4],
            "dynamic": False,
        },
        {"name": "dynamic_default_mu", "num_inference_steps": 3, "dynamic": True},
        {"name": "dynamic_explicit_mu", "num_inference_steps": 3, "dynamic": True, "mu": 0.3},
        {"name": "text_length_63", "num_inference_steps": 3, "max_sequence_length": 63},
        {"name": "text_length_64", "num_inference_steps": 3, "max_sequence_length": 64},
        {"name": "custom_timesteps", "num_inference_steps": 2, "timesteps": [1000.0, 800.0, 600.0], "dynamic": False},
    ]


def _compare(torch, latents, image, reference, limits):
    # Use float64 for reductions: float32 dot/norm roundoff can reject even
    # bit-identical full-resolution latents at the strict naive tolerance.
    expected = reference["latents"].double()
    actual = latents.double()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert image.shape == reference["image"].shape, (image.shape, reference["image"].shape)
    difference = (actual - expected).flatten()
    expected_flat, actual_flat = expected.flatten(), actual.flatten()
    expected_norm = expected_flat.norm().clamp_min(1e-12)
    denominator = (expected_flat.norm() * actual_flat.norm()).clamp_min(1e-12)
    metrics = {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected_norm).item(),
        "cosine": (torch.dot(actual_flat, expected_flat) / denominator).clamp(-1, 1).item(),
        "image_mae": (image - reference["image"]).abs().mean().item(),
    }
    failures = []
    if metrics["relative_l2"] > limits["relative_l2"]:
        failures.append("latent relative L2 exceeds tolerance")
    if metrics["cosine"] < limits["cosine_min"]:
        failures.append("latent cosine similarity is below tolerance")
    if metrics["image_mae"] > limits["image_mae"]:
        failures.append("decoded image MAE exceeds tolerance")
    return metrics, failures


def _run_pipeline(rank, world_size, init_method, mode, settings, report, report_path):
    import numpy as np
    from PIL import Image
    import torch
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype_name = settings.get("dtype", "float32")
    dtype = getattr(torch, dtype_name)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report.update(
        torch=torch.__version__,
        diffusers=diffusers.__version__,
        cuda=torch.version.cuda,
        hip=torch.version.hip,
        device=torch.cuda.get_device_name(rank),
        dtype=dtype_name,
        offload="model_cpu_offload",
        world_size=world_size,
    )
    load_start = time.perf_counter()
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        settings["model_path"],
        torch_dtype=dtype,
        local_files_only=True,
    )
    required_components = (
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
    )
    for name in required_components:
        assert getattr(pipeline, name, None) is not None, f"full pipeline is missing {name}"
    assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    parameter_count = sum(parameter.numel() for parameter in pipeline.transformer.parameters())
    tiny = parameter_count < 100_000_000
    assert not tiny or settings["allow_tiny"], (
        "checkpoint has fewer than 100M transformer parameters; only an explicit "
        "SD3_E2E_ALLOW_TINY=1 may run this as tiny_preflight"
    )
    classification = "tiny_preflight" if tiny else settings["classification"]
    report.update(
        classification=classification,
        transformer_parameters=parameter_count,
        load_seconds=time.perf_counter() - load_start,
    )
    scheduler_config = dict(pipeline.scheduler.config)
    vae_scale_factor = pipeline.vae_scale_factor
    channels = pipeline.transformer.config.in_channels
    resolution = 64 if tiny else 1024
    cases = _cases(3 if tiny else settings["quality_steps"])
    wrapped = None
    if mode != "stock":
        from xfuser.config.args import xFuserArgs
        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.pipelines.pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline

        parallel_state.init_distributed_environment(
            backend="nccl",
            distributed_init_method=init_method,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
        )
        config_args = dict(model=settings["model_path"], attention_backend="sdpa")
        if mode == "ulysses":
            config_args["ulysses_degree"] = 2
        elif mode == "cfg":
            config_args["use_cfg_parallel"] = True
        elif mode == "pipefusion_sync":
            config_args.update(
                pipefusion_parallel_degree=2,
                num_pipeline_patch=2,
                warmup_steps=max(settings["quality_steps"], 5),
            )
        engine_config, _ = xFuserArgs(**config_args).create_config()
        engine_config.runtime_config.dtype = dtype
        wrapped = xFuserStableDiffusion3Pipeline(pipeline, engine_config)
    callable_pipeline = wrapped if wrapped is not None else pipeline
    pipeline.set_progress_bar_config(disable=True)
    # Attach offload hooks to the final transformer wrapper, on this rank's GPU.
    pipeline.enable_model_cpu_offload(device=device)
    text_lengths = []

    def observe_t5(module, inputs):
        text_lengths.append(int(inputs[0].shape[-1]))

    observation = pipeline.text_encoder_3.register_forward_pre_hook(observe_t5)
    failures = []
    output_dir = Path(settings["output_dir"])
    try:
        for case in cases:
            text_lengths.clear()
            scheduler_options = {"use_dynamic_shifting": case["dynamic"]} if "dynamic" in case else {}
            original_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config, **scheduler_options)
            pipeline.scheduler = (
                wrapped._convert_scheduler(original_scheduler) if wrapped is not None else original_scheduler
            )
            call_scheduler = pipeline.scheduler
            schedule_length = len(case.get("sigmas", case.get("timesteps", []))) or case["num_inference_steps"]
            callback_steps, callback_timesteps = [], []
            final_latents = None

            def observe_step(callback_pipeline, step, timestep, callback_kwargs):
                nonlocal final_latents
                callback_steps.append(int(step))
                callback_timesteps.append(float(timestep.detach().cpu()))
                if mode == "naive" and "timesteps" not in case:
                    assert callback_pipeline.scheduler is original_scheduler, (
                        "naive callback received scheduler wrapper"
                    )
                if step == schedule_length - 1:
                    final_latents = callback_kwargs["latents"].detach().float().cpu().clone()
                return callback_kwargs

            call = dict(
                prompt=_PROMPT,
                negative_prompt=_NEGATIVE_PROMPT,
                height=resolution,
                width=resolution,
                guidance_scale=settings["guidance_scale"],
                generator=torch.Generator(device="cpu").manual_seed(42),
                output_type="np",
                callback_on_step_end=observe_step,
                max_sequence_length=case.get("max_sequence_length", 256),
                num_inference_steps=case["num_inference_steps"],
            )
            for name in ("sigmas", "mu", "timesteps"):
                if name in case:
                    call[name] = case[name]
            if mode == "stock" and "timesteps" in call:
                call["sigmas"] = [timestep / 1000.0 for timestep in call.pop("timesteps")]
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = callable_pipeline(**call)
            torch.cuda.synchronize(device)
            result = {
                "name": case["name"],
                "resolution": resolution,
                "requested_steps": case["num_inference_steps"],
                "schedule_steps": schedule_length,
                "max_sequence_length": call["max_sequence_length"],
                "guidance_scale": call["guidance_scale"],
                "seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "t5_input_lengths": list(text_lengths),
                "callback_steps": callback_steps,
                "callback_timesteps": callback_timesteps,
                "scheduler_config": dict(original_scheduler.config),
            }
            if mode == "ulysses":
                from xfuser.core.distributed import get_sp_group

                # The SD3 callback observes each rank's local rows before the
                # pipeline gathers them for VAE decoding. Reconstruct the same
                # full latent here so the comparison uses the stock shape.
                assert final_latents is not None, "Ulysses did not capture final latents"
                latent_rows = get_sp_group().all_gather(final_latents.to(device), separate_tensors=True)
                final_latents = torch.cat(latent_rows, dim=-2).cpu()
            assert pipeline.scheduler is call_scheduler, "scheduler wrapper was not restored"
            assert text_lengths and set(text_lengths) == {call["max_sequence_length"]}, result
            if wrapped is not None and (mode != "naive" or "timesteps" in case):
                from xfuser.core.distributed import get_runtime_state

                result["runtime_steps"] = get_runtime_state().input_config.num_inference_steps
                assert result["runtime_steps"] == schedule_length, result
            if output is not None:
                assert callback_steps == list(range(schedule_length)), result
                assert final_latents is not None and torch.isfinite(final_latents).all(), result
                assert tuple(final_latents.shape) == (
                    1,
                    channels,
                    resolution // vae_scale_factor,
                    resolution // vae_scale_factor,
                ), final_latents.shape
                image_array = np.asarray(output.images, dtype=np.float32)
                assert image_array.shape == (1, resolution, resolution, 3), image_array.shape
                assert np.isfinite(image_array).all(), "decoded image is non-finite"
                assert image_array.min() >= 0 and image_array.max() <= 1, "image is outside RGB [0, 1]"
                spatial_std = float(image_array.std(axis=(1, 2)).mean())
                assert spatial_std > 0.01 and float(np.ptp(image_array)) > 0.05, "decoded image is degenerate"
                image_tensor = torch.from_numpy(image_array.copy())
                stem = f"{mode}-{case['name']}"
                Image.fromarray(np.rint(image_array[0] * 255).astype(np.uint8)).save(output_dir / f"{stem}.png")
                torch.save({"latents": final_latents, "image": image_tensor}, output_dir / f"{stem}.pt")
                result.update(
                    latent_shape=list(final_latents.shape),
                    image_spatial_std=spatial_std,
                    image_path=str(output_dir / f"{stem}.png"),
                    output_rank=rank,
                )
                if mode != "stock":
                    reference = torch.load(output_dir / f"stock-{case['name']}.pt", weights_only=True)
                    limits = _NAIVE_LIMITS if mode == "naive" and "timesteps" not in case else _PARALLEL_LIMITS
                    result["tolerances"] = limits
                    result["comparison"], case_failures = _compare(
                        torch,
                        final_latents,
                        image_tensor,
                        reference,
                        limits,
                    )
                    stock_report = json.loads((output_dir / "stock-rank0.json").read_text())
                    stock_case = next(item for item in stock_report["cases"] if item["name"] == case["name"])
                    if not np.allclose(callback_timesteps, stock_case["callback_timesteps"], rtol=1e-6, atol=1e-4):
                        case_failures.append("scheduler timesteps differ from stock diffusers")
                    result["failures"] = case_failures
                    failures.extend(f"{case['name']}: {failure}" for failure in case_failures)
            report["cases"].append(result)
            _write_json(report_path, report)
            # Every rank releases its full transformer, including ranks that do
            # not decode an image and therefore bypass SD3's final hook cleanup.
            pipeline.maybe_free_model_hooks()
            torch.cuda.empty_cache()
    finally:
        observation.remove()
        pipeline.maybe_free_model_hooks()
    if mode != "stock":
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    assert not failures, "\n".join(failures)


def _worker(rank, world_size, init_method, mode, settings):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output_dir = Path(settings["output_dir"])
    report_path = output_dir / f"{mode}-rank{rank}.json"
    report = {
        "status": "running",
        "mode": mode,
        "rank": rank,
        "cases": [],
        "model_path": settings["model_path"],
        "model_id": settings["model_id"],
        "model_revision": settings["model_revision"],
        "prompt": _PROMPT,
        "negative_prompt": _NEGATIVE_PROMPT,
        "seed": 42,
    }
    _write_json(report_path, report)
    with (output_dir / f"{mode}-rank{rank}.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                _run_pipeline(rank, world_size, init_method, mode, settings, report, report_path)
            except BaseException:
                report.update(status="failed", error=traceback.format_exc())
                _write_json(report_path, report)
                raise
            report["status"] = "passed"
            _write_json(report_path, report)


def _launch(torch, mode, settings, tmp_path):
    world_size = _MODES[mode]
    context = torch.multiprocessing.get_context("spawn")
    for rank in range(world_size):
        # A reused artifact directory must not expose a previous failure while
        # the freshly spawned rank is still importing its dependencies.
        (Path(settings["output_dir"]) / f"{mode}-rank{rank}.json").unlink(missing_ok=True)
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                f"file://{tmp_path / (mode + '-init')}",
                mode,
                settings,
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + settings["timeout_seconds"]
    errors = []
    try:
        while any(process.is_alive() for process in processes):
            for rank, process in enumerate(processes):
                path = Path(settings["output_dir"]) / f"{mode}-rank{rank}.json"
                if path.exists():
                    report = json.loads(path.read_text())
                    if report["status"] == "failed":
                        errors.append(report["error"])
                if process.exitcode not in (None, 0):
                    errors.append(f"{mode} rank {rank} exited with code {process.exitcode}")
            if errors:
                break
            if time.monotonic() > deadline:
                errors.append(f"{mode} exceeded {settings['timeout_seconds']} seconds")
                break
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
    reports = []
    for rank, process in enumerate(processes):
        path = Path(settings["output_dir"]) / f"{mode}-rank{rank}.json"
        if path.exists():
            report = json.loads(path.read_text())
            reports.append(report)
            if report["status"] != "passed":
                errors.append(report.get("error", f"{mode} rank {rank} did not finish"))
        else:
            errors.append(f"{mode} rank {rank} produced no report")
        if process.exitcode != 0:
            errors.append(f"{mode} rank {rank} exited with code {process.exitcode}")
    assert not errors, f"Artifacts: {settings['output_dir']}\n" + "\n".join(dict.fromkeys(errors))
    assert len(reports) == world_size
    for name in (case["name"] for case in reports[0]["cases"]):
        assert (
            sum("output_rank" in case for report in reports for case in report["cases"] if case["name"] == name) == 1
        ), f"{mode}/{name} did not produce exactly one image"
    return reports


def test_pretrained_sd3_matches_diffusers(tmp_path):
    model = os.environ.get("SD3_E2E_MODEL")
    if not model:
        pytest.skip("set SD3_E2E_MODEL to a complete local pretrained SD3/SD3.5 checkpoint")
    model_path = Path(model).expanduser().resolve()
    assert (model_path / "model_index.json").is_file(), f"not a local diffusers checkpoint: {model_path}"
    modes = list(
        dict.fromkeys(
            mode.strip()
            for mode in os.environ.get(
                "SD3_E2E_MODES",
                "naive,ulysses,cfg,pipefusion_sync",
            ).split(",")
        )
    )
    assert modes and all(mode in _MODES for mode in modes), f"invalid SD3_E2E_MODES: {modes}"
    torch = pytest.importorskip("torch")
    required_devices = max(_MODES[mode] for mode in modes)
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_devices:
        pytest.skip(f"requires {required_devices} accelerator devices")
    if any(mode != "stock" for mode in modes) and not torch.distributed.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")
    output_dir = Path(os.environ.get("SD3_E2E_OUTPUT_DIR", str(tmp_path / "sd3-artifacts"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "model_id": os.environ.get("SD3_E2E_MODEL_ID", str(model_path)),
        "model_revision": os.environ.get("SD3_E2E_MODEL_REVISION", "unspecified local checkpoint"),
        "classification": os.environ.get("SD3_E2E_MODEL_CLASSIFICATION", "full_pretrained"),
        "allow_tiny": os.environ.get("SD3_E2E_ALLOW_TINY") == "1",
        "quality_steps": int(os.environ.get("SD3_E2E_QUALITY_STEPS", "20")),
        "guidance_scale": float(os.environ.get("SD3_E2E_GUIDANCE_SCALE", "5")),
        "dtype": os.environ.get("SD3_E2E_DTYPE", "float32"),
        "timeout_seconds": int(os.environ.get("SD3_E2E_TIMEOUT_SECONDS", "3600")),
    }
    assert settings["quality_steps"] > 0 and settings["timeout_seconds"] > 0
    assert settings["dtype"] in {"float32", "bfloat16"}, settings["dtype"]
    assert settings["guidance_scale"] > 1, "CFG coverage requires guidance scale > 1"
    assert settings["classification"] in {"full_pretrained", "full_pretrained_derivative"}
    summary = {
        "settings": settings,
        "runs": {},
        "status": "running",
        "quality_review": "Review the retained PNGs; numerical parity does not establish prompt quality.",
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    try:
        for mode in ["stock", *(mode for mode in modes if mode != "stock")]:
            summary["runs"][mode] = _launch(torch, mode, settings, tmp_path)
            _write_json(summary_path, summary)
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        _write_json(summary_path, summary)
        raise
    summary["status"] = "passed"
    _write_json(summary_path, summary)
    print(f"SD3 E2E {summary['runs']['stock'][0]['classification']} artifacts: {output_dir}")
