# External GPU Validation Handoff

## Current status: partially run

The checked-in `validation_status` stays **NOT RUN**, because most of the matrix
has not been executed anywhere and a result is only evidence once an operator
attaches its JSONL record, log, and generated output.

What has run: a hundred and eighty-one of the matrix's two hundred and three
cases, on 8× MI355X (`gfx950`), covering the memory-efficient load paths in bf16
and FP8 across twenty models, twelve image and eight video, which is every
model that node can load through those paths. A hundred and sixty-four passed,
every still image scored against an unquantized render and every clip checked
frame by frame, and fifteen more passed by asserting the rejection they
expected: ROCm INT8, seven refusals from models that withhold the memory-efficient
load or the sharding flag outright, and seven limits found by running combinations
that had been waiting on hardware they did not need. Two report an
environment mismatch and were correctly not run, both wanting `gfx942` alone for
the FP4 refusal they assert. The remaining twenty-two need hardware
this node is not, and [what needs other hardware](#what-needs-other-hardware)
says which of them are worth a booking. Nothing recorded is failing. The results
file also holds records for fourteen case IDs the matrix no longer carries, among
them the `gen-mi3xx-z-image-turbo-bf16-fsdp-w4` failure that predates the
port-collision fix; that configuration was re-run by hand at four ranks and
loads, shards and renders, so the failure was the collision and not the model.
Four more are cases the hardware reassessment renamed or retired, and the section
below says what each of them found first. The last six are Ideogram-4's: one
withheld-refusal case retired when the model stopped withholding the path, and
five emitted at eight ranks before the generator learned that eighteen attention
heads cannot be split eight ways, which they failed on rather than on anything
they meant to measure; they are recorded at six ranks now.
[Memory-efficient load results](meta_load_results.md) reports all three sweeps,
including the defects they found, each of them in a combination nothing had run
before. HunyuanVideo's six cases were re-run once more after its text encoder was
brought into the memory-efficient path, which also renamed its two FP8 cases to
the `fp8-te` form the other quantizing models use. The two eager text-encoder FP8
cases were re-run again after the streaming probe was corrected, since their
earlier records measured the fallback rather than the path the flag asks for.
The screening plan gained an unquantized `replicated` case per model, twenty in
all, because the broadcast fill had only ever been generated with FP8 on top of
it: reading its cost meant reading the broadcast and the quantization together,
and the only unquantized measurement of it was a curated pair at four ranks.
Those records live on that node and are not checked in.

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
`sampling` alongside the source it was read from. A model no benchmark config
covers may cite its own runner's `DefaultInputValues` instead, which is a
declaration in this repository and is checked against the runner rather than
taken on trust. There is no global default to fall back on: a model with neither
raises `UnknownSampling` and reports as a case that cannot run, rather than being
rendered at someone else's step count. This matters beyond fairness — four steps
at 512×512 left Z-Image an unconverged blob, and a model's step count and
resolution also decide how much a numeric change moves its output, which is what
the quality gate below measures.

A sampling entry may also carry what a model needs beyond a prompt: `num_frames`
for a video model, `input_images` for one that edits a picture, and
`runtime_args` for an argument the model requires and nothing else does, such as
the task Wan2.2-TI2V must be given because it serves both `i2v` and `t2v`. Paths
are written as environment placeholders, `${XDIT_INPUT_IMAGE}` and
`${XDIT_WAN_INPUT_IMAGE}`, which resolve to the images the models' own benchmark
configs pass. Unset, the affected cases record an `environment_mismatch` rather
than running, which is the harness saying it was not given what the model needs.

`guidance_scale` may be an explicit `null`, which passes no guidance flag at all.
The key is still required, so silence stays a statement rather than an omission.
Ideogram-4 needs it: its runner builds a guidance schedule and only when it is
given no value, so any number, the CLI default included, would replace the
schedule the model ships with. The MiniMax-H3 and HunyuanVideo 1.5 runners
forward no guidance to their pipelines at all, where a number would be inert but
would still read as an operating point somebody measured.

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
and which models need an input image. A model on that list is generated once its
sampling entry names an image, and skipped while none does.

`tests/gpu_validation/curated_cases.json` holds cases whose expected outcome is a
claim about the world rather than about the code, which in practice means every
expected rejection: that AITER ships no FP4 kernels for gfx942, that FSDP2 cannot
shard uint8 MXFP4 before torch 2.12. Deriving those from the same probe functions
the runner calls would leave the suite asserting only that the code agrees with
itself. A curated case wins any collision with a generated one, so its ID, and
therefore its result history, survives regeneration.

The withheld models are curated for the same reason turned around: a model that
declares eager loading only is generated no cases at all, so without one, nothing
checks that its refusal happens. Each such case names the refusal in its
`error_pattern`, and a test compares that pattern against the message the
runner's own declaration produces, so a pattern loose enough to be satisfied by
any failure does not pass. These cases are cheap wherever they run, because the
load contract is selected before allocation: MiniMax-H3's refusal costs process
startup rather than a read of a 330G checkpoint. Two refusals can apply to one
model, and then both are asserted: the HunyuanVideo 1.5 runners declare no
`fully_shard_degree` capability, which is refused while the config is validated
and before any load contract exists, so the sparse runner has one case for that
and one on the replicated path for the withheld reason itself.

Generated cases only ever expect success, which is not circular: the declaration
decides what is worth attempting and the GPU decides whether it works. Generation
is also blind to the local HF cache on purpose. The matrix is a plan, and which
weights a machine happens to hold is an execution-time fact; filtering on it would
make the file differ per node and make case IDs come and go. Use
`tools/cache_inventory.py` to see what the current machine can actually reach.

## What needs other hardware

A case names an accelerator, and it is easy to read that as a hardware
requirement. Often it is not one. Every case that had never run was re-read
against what its claim is actually gated on in the code, and most of them were
waiting on a machine for no reason the code supports.

Three things genuinely need hardware this node is not:

- **RDNA4, for the AITER FP8 backend.** `_use_aiter_fp8_rdna4` is true only on
  `gfx1200` or `gfx1201` with AITER present, and when it is, FP8 goes through
  AITER's quantizer, its FP8 layer layout and its Transformers streaming adapter
  instead of torchao. No amount of gfx950 FP8 reaches any of it. Reviewing this
  found a hole: every RDNA4 FP8 case was eager or quantized the transformer
  alone, so the densest part of that path, per-block quantization during the fill
  with a quantized text encoder, was covered by nothing.
  `rdna4-flux2-fp8-fsdp-te-tf5` was added for it.
- **Blackwell, for NVFP4.** The probe is `capability >= (10, 0)`; below that the
  format reports `NVFP4 requires CUDA capability >= 10.0` and there is nothing to
  measure.
- **Any CUDA device, for INT8 that works.** `TorchAO INT8 is supported only on
  CUDA`, so the supported path cannot be seen from ROCm at all.

Everything else that was waiting either belonged here or was already covered:

- **The offload modes are not architectural.** `--enable_model_cpu_offload`,
  `--enable_sequential_cpu_offload` and the group variants are Diffusers and
  accelerate hooks with no platform gate, yet all four were pinned to RDNA4,
  Hopper, Ada or Blackwell, which left offload with no coverage anywhere. They
  are gfx950 cases now, and running them found three defects described below.
- **Transformers 4 is a library, not a machine.** The split is feature-detected
  on whether `transformers.core_model_loading` imports, so both cases are pinned
  here, and an environment satisfying them was built on this node rather than
  booked. Transformers 4 caps `huggingface-hub` below 1.0 while the image ships
  1.20, and Diffusers accepts anything under 2.0, so a directory holding just
  those two, ahead of the image on `PYTHONPATH`, satisfies both without touching
  the ROCm torch build:

  ```bash
  pip install --no-deps --target /opt/tf4-libs "transformers==4.57.1" "huggingface_hub<1.0"
  PYTHONPATH=/opt/tf4-libs:$PWD python tools/gpu_validation.py --case ... --execute
  ```

  The harness reads the version through `importlib.metadata`, which sees the
  shadowing copy, so the gate opens and each record carries
  `packages.transformers: 4.57.1`. `rocm-flux2-fsdp-te-tf4-rejected` refuses
  before allocation there, as it claims: sharding cannot apply the post-load
  text-encoder fallback that Transformers 4 forces.
- **The mixed FP8/FP4 schedule runs here.** It was withheld from generation only
  because the `gfx942_or_gfx950` token spans two archs that differ on FP4,
  exactly as FP4 itself is, and FP4 was then curated and pinned to gfx950 where
  it passes. Its three cases had sat unrun on RDNA4 and Blackwell, and none of
  them passed `--num_hybrid_gemm_high_precision_steps`, which the schedule
  requires, so each would have failed on a missing argument wherever it ran.
- **The ROCm INT8 refusal is platform-level.** `rocm-zimage-int8-rejected`
  already asserts it here, so the RDNA4 twin was retired rather than booked.
- **Three rejections cannot run anywhere as written.** SD3.5, CausalWan and
  Wan2.2-Distilled-I2V have no sampling entry, and an entry is read while the
  command is built, so on an RDNA4 node each reports that it cannot run and
  asserts nothing. No node this has run on holds their weights, so an operating
  point cannot be chosen without guessing one. Their notes say so, so the
  refusals are not mistaken for work a booking would complete.

Running the re-pinned cases is what made the reassessment worth doing, because
four of the six failed and none of the failures was about hardware:

- Sequential offload at two ranks died in NCCL with `Duplicate GPU detected`.
  Accelerate places the offloaded modules on the default device for every rank,
  so both ranks landed on one device and the first all-to-all failed. The case
  had inherited two ranks from the RDNA4 case it replaced and would have failed
  the same way there. It is a one-rank case now, which is where offload belongs.
- Group offload with AITER FP4 aborted the rank with SIGABRT and no traceback:
  AITER's quant module binds a device from the parameter it is handed, and a
  parameter on the host resolves to an invalid ordinal.
- The same offload with `--group_offload_low_cpu_mem` raised from inside the
  hook, which pins each tensor before onloading it, because torch has no
  `pin_memory` for `Float4_e2m1fn_x2`.
- The mixed schedule under the blockwise fill was refused, because this torch
  cannot shard a non-floating-point parameter under FSDP2 and the FP4 half
  targets the blocks the strategy wraps. That is the gate `rdna4-wan22-hybrid-fsdp`
  predicted in its own notes; it is now measured, and both cases expect the
  refusal until torch 2.12.

The two FP4 offload failures are now refused before allocation, by
`assert_offload_is_compatible_with_format`, and asserted as rejections. The
refusal is scoped to the AITER backend because that is where both were measured;
CUDA FP4 packs through TorchAO tensor subclasses, whose offload behaviour nothing
here has tested. `rocm-krea2-fp8-eager-group-offload` keeps a passing result for
the same offload mode, so the two rejections read as a limit of AITER FP4 rather
than of group offloading.

Once offload had cases at all, two further questions could be asked of it, and
both were worth asking:

**Across ranks.** The sequential failure above was read as offload being a
single-rank feature, and it is not. Group offloading had always computed its
onload device from `get_world_group().local_rank`; the two whole-pipeline modes
passed nothing, so Diffusers defaulted them to `cuda:0` and every rank took it.
Both now name the local device, and `rocm-flux1-fp8-replicated-sequential-offload`
and `rocm-zimage-turbo-bf16-replicated-model-offload` pass at two ranks, which
also gives offload its first results on top of the replicated meta load.

**On top of the blockwise fill.** This is the combination the branch is for, and
nothing had run it. Whole-model offload works, because it moves components rather
than reaching into them:
`rocm-zimage-turbo-bf16-fsdp-model-offload` fills block by block, shards, and
then moves the sharded model between host and device around each call. The other
two modes cannot, because sharding replaces each parameter with a DTensor. Group
offloading asks every parameter whether it is pinned and torch registers no
sharding strategy for `aten.is_pinned`; sequential offloading rebuilds each
parameter as it moves it, and `DTensor.__new__` needs a spec a plain tensor does
not carry. Both failed mid-denoise, after a full sharded load had been paid for,
so `assert_offload_is_compatible_with_sharding` refuses them before allocation
and the two rejection cases assert that.

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
- which materialization each quantized component took, under
  `quantization_paths`, read back from the descriptor line every one of them
  logs. Whether a text encoder quantizes each parameter as it is read or
  materializes first and converts afterwards is chosen at runtime from what the
  installed libraries expose, so it is not derivable from the case, and it moves
  the memory figures below by gigabytes. A record without the field predates its
  being captured, which is not the same as a run that quantized nothing;
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
peak device and host memory per case, says for a quantized text encoder which
materialization it took, and then lists which combinations for each
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
floor sits two orders of magnitude below the flattest real render measured.

A video clip is read the same way, per frame as well as whole: twelve evenly
spaced frames are decoded and the flattest of them is measured, so a clip that
renders one good frame and then collapses to black cannot average its way past
the floor. Twelve keeps the check cheap on a 129-frame clip while still sampling
the whole timeline. Across every video case run so far the flattest sampled frame
measured 0.21 or above against a floor of 0.01. An artifact nothing can decode
records as unmeasured rather than being credited with having drawn something.

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
- **A video case records as `unscored` rather than being compared.** SSIM over a
  clip would only decide identity-stable models, and no video model has been
  measured to be one; the blank-output gate above still reads every clip, so a
  collapsed render is still caught.

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
