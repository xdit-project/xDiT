# External GPU Validation Handoff

## Current status: partially run

The checked-in `validation_status` stays **NOT RUN**, because most of the matrix
has not been executed anywhere and a result is only evidence once an operator
attaches its JSONL record, log, and generated output.

What has run: thirty-six cases on 8× MI355X (`gfx950`), covering the
memory-efficient load paths in bf16 and FP8 across six image models, all passing
and all scored against an unquantized render.
[Memory-efficient load results](meta_load_results.md) reports them, including two
bugs the sweep found in combinations nothing had run before. Those records live on
that node and are not checked in.

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

Each case is sampled the way its own model is meant to be sampled, from that
model's entry in `.ci/benchmark_configs`, which the matrix carries under
`sampling` alongside the source it was read from. There is no global default to
fall back on: a model with no entry raises `UnknownSampling` and reports as a
case that cannot run, rather than being rendered at someone else's step count.
This matters beyond fairness — four steps at 512×512 left Z-Image an
unconverged blob, and a model's step count and resolution also decide how much a
numeric change moves its output, which is what the quality gate below measures.

The matrix treats Transformers `4.x` and `5.x` as environment requirements; the
runner does not install them. Before execution, it probes the installed
Transformers major, CUDA or ROCm platform, accelerator architecture, and the
required TorchAO or AITER package. A mismatch writes an
`environment_mismatch` record with `execution: "NOT RUN"` and does not launch
xDiT. The probe preserves the visible device count and requires at least
`world_size` matching accelerators. It applies numeric
`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HIP_VISIBLE_DEVICES`
selection where the platform tooling exposes an ordered device list. The
selected variable and value are recorded under `validation_probe.visibility`.
There is no bypass flag: select the matching environment or matrix case.

A case's `hardware.accelerator` is enforced, not documentation, and an
unrecognised token is a validation error rather than a case that silently never
matches. On ROCm the accepted tokens are:

| Token | Matches | Use it when |
| --- | --- | --- |
| `gfx942`, `gfx950` | that arch alone | behaviour is specific to it, as with the FP4 rejections that hold only where AITER ships no FP4 kernels |
| `gfx942_or_gfx950` | either datacentre arch | behaviour is the same on both, which `envs._on_mi3xx` already treats as one class |
| `non_rdna4_rocm` | any ROCm that is not RDNA4 | the gate is RDNA4-versus-rest and older parts such as gfx90a are acceptable |
| `gfx1200_or_gfx1201` | RDNA4 | the case needs the AITER block-scale path |

Pin a case to a single arch only when its behaviour genuinely differs there.
Where a case merely happened to run on one machine first, record that under
`observed_on_<arch>` and leave the accelerator token as wide as the behaviour
allows, so the case stays eligible on the other hardware.

## Where the cases come from

`tests/gpu_validation/matrix.json` is generated, not hand-edited:

```bash
python tools/generate_validation_matrix.py           # rewrite the matrix
python tools/generate_validation_matrix.py --check    # fail if it is stale
```

Hand-authoring did not scale. Fifty runners carry a usable load declaration between
them, covering 444 placement-and-quantization combinations before any hardware
profile or rank count multiplies that, and each hand-written case restated a
runner's `LoadDeclaration` and could drift from it. The enumeration is now derived
from those declarations, so two inputs stay hand-written:

`tests/gpu_validation/profiles.json` holds what cannot be derived: which
quantizations a hardware profile may attempt, how many ranks each placement gets,
and which models need an input image the harness cannot supply.

`tests/gpu_validation/curated_cases.json` holds cases whose expected outcome is a
claim about the world rather than about the code, which in practice means every
expected rejection: that AITER ships no FP4 kernels for gfx942, that FSDP2 cannot
shard uint8 MXFP4 before torch 2.12. Deriving those from the same probe functions
the runner calls would leave the suite asserting only that the code agrees with
itself. A curated case wins any collision with a generated one, so its ID, and
therefore its result history, survives regeneration.

Generated cases only ever expect success, which is not circular: the declaration
decides what is worth attempting and the GPU decides whether it works. Generation
is also blind to the local HF cache on purpose. The matrix is a plan, and which
weights a machine happens to hold is an execution-time fact; filtering on it would
make the file differ per node and make case IDs come and go. Use
`tools/cache_inventory.py` to see what the current machine can actually reach.

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
`<output-root>/<run-id>/<case-id>`, where the run id is
`<UTC timestamp>-<UUID>` unless `--run-id` names it. Run first and case second so
one invocation produces one directory holding every case it ran, rather than a
sibling directory per model. A sweep runs one case per invocation, which would
still be a directory per case, so pass the same `--run-id` to each invocation of
a sweep and the whole sweep is one folder to review.

Execution reserves that directory atomically and fails if it already exists. A
prior case directory is never reused, so stale artifacts cannot satisfy a later
case; reusing a `--run-id` across invocations is safe because the case name
differs, and re-running the same case under the same run id is refused. The
output root is resolved once against the validation runner's caller working
directory. The same absolute run directory is passed to xDiT, reserved, scanned,
and recorded, even though the child process itself starts from the repository
root.

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

## Reading the results

`tools/validation_report.py` turns those records into a report meant for people
rather than for parsing:

```bash
python tools/validation_report.py \
  --results gpu-validation-results/results.jsonl
```

It states whether the run is green, gives the outcome, load time, wall time and
peak device and host memory per case, and then lists which combinations for each
model on this hardware have still not run. The denominator counts only cases
this machine's accelerator can run, so cases needing other hardware are not
reported as outstanding work. Pass `--format markdown` for a version to paste
into a report, and pass several `--results` files to combine runs; the latest
record for a case ID wins, so a re-run supersedes an earlier attempt.

Two readings the report deliberately keeps apart. A case the matrix expects to
be refused shows as `rej` rather than `ok`: the guard firing is a pass, but the
combination still does not work on that hardware. And the post-load column is
wall minus load and compile, so it covers VAE decode, saving and teardown as well
as inference; the runner does not time inference on its own.

Two of its columns come from the quality checks. `spread` is the run's own
artifact measurement, shown as `BLANK` when it failed, and `-` when nobody
measured that run — the report renders records and does not read images, so a run
recorded before the measurement existed stays blank until it is re-run. `vs ref`
is the SSIM from the `--score-quality` pass, so it describes a different, compile-
free run of the same case rather than the run whose timings are on that row, and
a score marked `info` was not allowed to fail its case.

Generated artifacts use a central allowlist,
`GENERATED_ARTIFACT_EXTENSIONS` in `tools/gpu_validation.py`. The current
extensions are `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.mp4`, `.webm`, and
`.mov`. JSON, JSONL, logs, timing files, checkpoints, and other metadata never
count as successful inference output. Extend the central allowlist and its
tests when a runner intentionally adds another generated media type.

## Quality comparison

Two checks run on a case's output. One needs no reference and gates everything;
the other needs a reference and gates only where a score is evidence.

### Did it draw anything

Every executed case measures the spread of its own artifact. A uniform frame is
`failed_blank_output` regardless of exit status, model, or expected outcome. This
exists because an FP8 Qwen-Image run wrote a pure black 2048×2048 frame and was
reported as passed: reference comparison had nothing to say about that model, so
nothing looked at the image at all. The measure is spread rather than mean, since
a night scene is legitimately dark but no legitimate render is uniform, and the
floor sits two orders of magnitude below the flattest real render measured. An
artifact with no still-image reader, a video, records as unmeasured rather than
being credited with having drawn something.

### Does it match an unquantized render

`--score-quality` compares each case against its model's `eager`, unquantized,
single-rank case and fails the case when it diverges:

```bash
python tools/gpu_validation.py \
  --case gen-mi3xx-flux-1-dev-fp8-te-fsdp-w8 \
  --execute --score-quality \
  --results gpu-validation-results/results.jsonl
```

Four things about that path are deliberate:

- **It disables `torch.compile` for every run in the pass,** because a compiled
  run is not reproducible enough to compare: the same case has scored 0.98
  against itself when compile picked different kernels. Those runs are marked
  `scoring_run` and their timings stay out of the performance columns.
- **The reference is matched on attributes, not by name** — eager placement, no
  quantization, no offload, one rank — so a regenerated matrix cannot leave a case
  scoring against the wrong thing. It is reused from `--results` when a
  compile-free run of it is already recorded, so scoring one case does not re-run
  an eight-rank reference to judge it. Within one invocation, references run
  first.
- **Two floors, because the gate judges two different changes.** A case that only
  moves weights should render the same image and is held to 0.90 SSIM; a case
  that quantizes them moves texture while keeping content and is held to 0.60.
  One floor cannot do both: loose enough for FP8, it would pass a shard that
  rendered a different picture. Both are calibrated on measurements recorded in
  `tools/image_quality.py`, and a floor is only meaningful next to the sampling it
  was measured at — the previous single floor went stale exactly that way when
  each model started being sampled its own way.
- **Only a model measured to reproduce its sample is gated on the score.** Others
  record it as an observation, shown as `info`. Base Z-Image renders two good
  pictures of its prompt that score 0.6254 against each other, so on models like
  it a low score says a different sample came out, not that the load was wrong.
  `IDENTITY_STABLE_MODELS` holds the list and the argument.

Pass `--ssim-min` to override the floor a case would otherwise get. Treat the
result as a gross-correctness check: it answers whether the model still drew the
picture, not whether quantized quality is acceptable. That judgement stays with
the operator, recorded with `--quality-note`, and comparing hashes is only
meaningful for determinism across identical configurations.

## Attaching completed results

Attach these files together to the issue or pull request:

1. the JSONL result file;
2. each case's `validation.log`;
3. generated outputs and referenced baseline outputs;
4. the exact matrix file used, if it differs from the checked-in commit.

Report the validated commit SHA and whether the worktree was dirty. Do not edit
the checked-in `validation_status: "NOT RUN"` after a partial external run.
Result JSONL records are the source of truth. State skipped cases and missing
metrics explicitly. Passing unit tests or dry-runs do not constitute GPU
end-to-end validation.
