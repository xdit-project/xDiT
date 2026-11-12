# Outputs, timing, and profiling

## Timing and warmup

`--num_iterations` repeats generation. `--warmup_calls` performs complete untimed
pipeline calls before those iterations. This differs from `--warmup_steps`,
which counts synchronous denoising steps in the PipeFusion schedule, and
`--profile_warmup`, which configures the profiler's sampling schedule.

If no explicit or CUDA graph capture warmup runs, the runner discards the first
timing when it has multiple measurements. `timings.json` contains
a list of timings in seconds; batched inference records each batch separately.
`--use_torch_compile` can also perform compilation warmup during initialization.

`--profile` runs PyTorch profiling with `--profile_wait`, `--profile_warmup`, and
`--profile_active`, measured in full pipeline calls (one complete batch traversal
when batching). The profiler runs for the sum of these values instead of `--num_iterations`.
It does not apply `--warmup_calls`. It saves a compressed Chrome trace for each rank.
`--profile_capture_phase` is a separate option for supported model compilation
paths; it records CUDA graph capture under the conditions described by those paths.

`--determinism_check` compares timed outputs against the first using exact
equality. Use `--determinism_check_report_ranks` to select which ranks save
failure outputs. This retains complete outputs and adds memory and synchronization costs. The
total elapsed time includes these costs.

```bash
xdit --model FLUX.1-dev --prompt "A botanical garden" \
    --ulysses_degree 4 --warmup_calls 1 --num_iterations 5 \
    --output_directory ./benchmark_results
```

For profiling, replace the iteration controls with `--profile`. Use distinct
output directories when comparing configurations.

## Compilation

Add `--use_torch_compile` to a supported model command. Initial compilation adds
startup cost and can use extra memory. Reuse the same input shapes to reduce
the compilation cost per run.

## Dataset inference

The CSV must contain a `prompt` column, and `--batch_size` is required:

```bash
xdit --model FLUX.1-dev --dataset_path ./prompts.csv --batch_size 4 \
    --ulysses_degree 4 --output_directory ./dataset_outputs
```

## Output files

The runner saves outputs under `--output_directory` (default: the current directory).
The default saver writes PNG images or MP4 videos, plus `timings.json` for a normal
run. Profiling writes `profile_trace_rank_{rank}.json.gz`; capture-phase traces go
under `capture_traces/`. Models with custom video/audio savers may write additional
files or use different output names.

Output names normally include the model's output prefix, parallel degrees,
compilation setting, dimensions, optional task, and output index. Reusing the same
output directory and configuration can overwrite files, including `timings.json`.

## Programmatic usage

Run the following as a Python script under `torchrun --nproc_per_node=4`.
The runner constructor initializes the distributed configuration. `initialize()`
loads the model. Use the same devices and model access required by the CLI.

```python
from xfuser.runner import xFuserModelRunner

config = {
    "model": "FLUX.1-dev",
    "prompt": "A beautiful garden with flowers",
    "height": 1024,
    "width": 1024,
    "ulysses_degree": 4,
    "num_inference_steps": 50,
    "seed": 42,
    "output_directory": "./outputs",
    "input_images": [],
}

runner = xFuserModelRunner(config)
input_args = runner.preprocess_args(config)
runner.initialize(input_args)
output, timings = runner.run(input_args)
runner.save(output=output, timings=timings)
runner.cleanup()
```

To profile, replace the `run` and `save` calls with:

```python
output, timings, profile = runner.profile(input_args)
runner.save(profile=profile)
```

The profiling path returns an empty timing list. Save `output` separately if
needed, and use `run()` for benchmark timings.
