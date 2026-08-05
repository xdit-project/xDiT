# External GPU Validation Handoff

## Current status: NOT RUN

This repository contains a reproducible validation plan, not GPU results. The
matrix was prepared on a system where the GPU end-to-end jobs were deliberately
not run. A result is evidence only after an operator executes a case on the
declared hardware and attaches its JSONL record, log, and generated output.

The artifacts are:

- `tests/gpu_validation/matrix.json`: machine-readable case definitions and
  expected outcomes.
- `tools/gpu_validation.py`: dependency-light planner, runner, monitor, and
  JSONL recorder.
- `tests/core/test_gpu_validation_handoff.py`: schema, coverage, command, filter,
  serialization, and expected-rejection tests.

## Setup

Use a clean checkout of the commit being validated. Install xDiT and the
backend-specific PyTorch, TorchAO, AITER, Diffusers, and Transformers versions
under test. Authentication remains external to the artifact: use the standard
Hugging Face CLI or environment on the validation host. Never put a token in
the matrix, command line, result, or attachment.

Before a run, capture the environment and inspect the plan:

```bash
git status --short
python tools/gpu_validation.py --list
python tools/gpu_validation.py --dry-run --tag smoke
```

The checked-in cases use four-step 512×512 workloads to make first-pass
validation practical. Increase dimensions, frames, or steps only as a separate
follow-up; preserve the original case ID and command when reporting this
matrix.

Transformers `4.x` and `5.x` in the matrix are environment requirements, not
runner-managed installations. Before execution, the runner probes the installed
Transformers major, CUDA or ROCm platform, accelerator architecture, and the
required TorchAO or AITER package. A mismatch writes an
`environment_mismatch` record with `execution: "NOT RUN"` and does not launch
xDiT. The probe preserves the visible device count and requires at least
`world_size` matching accelerators. It applies numeric
`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HIP_VISIBLE_DEVICES`
selection where the platform tooling exposes an ordered device list. The
selected variable and value are recorded under `validation_probe.visibility`.
There is no bypass flag: select the matching environment or matrix case.

## Selecting and executing cases

Filters compose. Repeated tags are ANDed; repeated models are ORed.

```bash
# One case, dry-run (dry-run is also the default)
python tools/gpu_validation.py \
  --case rdna4-flux2-fp8-eager-te-tf5 \
  --dry-run

# All RDNA4 smoke cases
python tools/gpu_validation.py \
  --backend rdna4_aiter \
  --tag smoke \
  --dry-run

# Execute and append one structured record
python tools/gpu_validation.py \
  --case rdna4-flux2-fp8-eager-te-tf5 \
  --execute \
  --results gpu-validation-results/results.jsonl
```

`--model` accepts the matrix model name, model family, or checkpoint value.
Use `--continue-on-error` for a batch where later cases should run after a
failure. The batch still exits nonzero if any executed case fails or an expected
rejection is missing, late, or wrong. Exit code 2 means cases were not run due
to selection or environment mismatch; exit code 1 takes precedence when a
batch also contains an execution failure.

Each case uses its `timeout_seconds` value or the matrix default. Pass
`--timeout-seconds N` to override that limit for every selected case. A timeout
terminates the case's isolated process group, kills any processes that remain
after the grace period, and writes a `timed_out` failed record. With
`--continue-on-error`, later cases still run and the batch exits nonzero.
Timeout values must be finite positive numbers.

`--list`, `--dry-run`, and `--execute` are mutually exclusive action modes.
Omitting all three retains the safe dry-run behavior.

Every command receives a new output directory:
`<output-root>/<case-id>/<UTC timestamp>-<UUID>`. Execution reserves that
directory atomically and fails if it already exists. A prior case directory is
never reused, so stale artifacts cannot satisfy a later case. The output root
is resolved once against the validation runner's caller working directory. The
same absolute run directory is passed to xDiT, reserved, scanned, and recorded,
even though the child process itself starts from the repository root.

Local cases mean an offline, pre-populated Hugging Face cache. They preserve
the registered `--model` alias and add `HF_HUB_OFFLINE=1` plus `HF_HOME`:

```bash
export XDIT_LOCAL_HF_HOME=/absolute/path/to/huggingface-cache
python tools/gpu_validation.py \
  --case blackwell-flux2-nvfp4-local-cache \
  --execute
```

Cases with external images or custom state dicts use named environment
placeholders such as `XDIT_WAN_HIGH_NOISE`. The runner stops before execution
if any `${NAME}` is unset or empty. It writes a structured
`environment_mismatch` record with `execution: "NOT RUN"` and continues when
`--continue-on-error` is set. Result commands retain `${NAME}` rather than its
expanded value, and matching values are redacted from captured child output;
paths and credentials are never hardcoded or logged.

## What a result records

Schema version 2 appends one JSON object per executed or environment-mismatched
case to the selected JSONL file:

- case ID, expected outcome, exact expanded command, UTC timestamp, and exit
  status;
- commit SHA, dirty-worktree flag, Python/package versions, platform, and
  available `nvidia-smi` or `rocm-smi` device information;
- wall time and model-load duration, measured between the runner's
  initialization markers;
- first-forward state: `not_reached`, `failed`, or `succeeded`. Here
  `succeeded` means the inference call reached `Running model...` and the
  process exited successfully;
- sampled peak RSS summed across the launcher process tree and sampled cgroup
  memory. Cgroup memory is scope-wide and may include unrelated processes;
- sampled peak GPU memory. NVIDIA measurements are filtered to the process
  tree. ROCm SMI exposes a device-global value, recorded with
  `gpu_memory_scope: device_global`. Missing tooling produces `null`, never a
  fabricated value;
- generated output paths, byte counts, SHA-256 hashes, validation log path,
  reference path, and quality notes.

An expected inference case passes only with exit status zero, a successful
inference call, and at least one newly generated, non-empty artifact in its
fresh run directory with a recorded SHA-256 hash. An expected preflight case
does not require an output artifact; it passes only when the process exits
nonzero before `Running model...` and its log matches the declared
`error_pattern`. A late failure, wrong rejection, missing rejection, or missing
output is a failed validation.

Generated artifacts use a central allowlist,
`GENERATED_ARTIFACT_EXTENSIONS` in `tools/gpu_validation.py`. The current
extensions are `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.mp4`, `.webm`, and
`.mov`. JSON, JSONL, logs, timing files, checkpoints, and other metadata never
count as successful inference output. Extend the central allowlist and its
tests when a runner intentionally adds another generated media type.

## Quality comparison

1. Run a non-quantized reference with the same checkpoint revision, prompt or
   input image, seed, dimensions, frame count, and inference steps.
2. Keep the reference output with the validation bundle and pass its path with
   `--reference`.
3. Record visible differences, NaNs/artifacts, text-conditioning regressions,
   temporal instability, or acceptance criteria with `--quality-note`.
4. Compare hashes only for determinism across identical configurations.
   Quantized and reference outputs are not expected to have identical hashes.
5. Treat visual acceptance as an operator decision; the recorder does not
   infer image or video quality.

Example:

```bash
python tools/gpu_validation.py \
  --case blackwell-flux2-nvfp4-eager \
  --execute \
  --reference references/flux2-bf16-seed1234.png \
  --quality-note "No visible composition loss; small texture change."
```

## Attaching completed results

Attach these files together to the issue or pull request:

1. the JSONL result file;
2. each case's `validation.log`;
3. generated outputs and referenced baseline outputs;
4. the exact matrix file used, if it differs from the checked-in commit.

Report the validated commit SHA and whether the worktree was dirty. Do not edit
the checked-in `validation_status: "NOT RUN"` to summarize partial external
runs; result JSONL records are the source of truth. State skipped cases and
missing metrics explicitly. Passing unit tests or dry-runs do not constitute
GPU end-to-end validation.
