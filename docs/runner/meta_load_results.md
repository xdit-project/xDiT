# Memory-efficient load: measured results on MI355X

First GPU end-to-end evidence for the memory-efficient load paths, taken on 8× AMD Instinct MI355X
(`gfx950`) across six image models this node's cache holds. Thirty-six cases, all passing, every image
scored against an unquantized render of the same prompt.

Read this together with [the handoff](meta_load_handoff.md), which covers the node's setup and the
threads still open, and [the validation handoff](gpu_validation_handoff.md), which covers how the
harness plans, runs and scores a case.

## What ran

Six cases per model, at the rank count these models are served at:

| Case | What it is |
| --- | --- |
| `eager` at 1 rank | the reference every image is scored against |
| `eager` at 8 ranks | the honest baseline: what the same node does without this branch, at the same parallelism |
| `fsdp_eager_fill` at 8 ranks | the control: materialize the transformer, then shard it, which is what a naive FSDP load does |
| `fsdp_blockwise` at 8 ranks | this branch: build on meta, fill and shard block by block |
| `fsdp_blockwise` + fp8 at 8 ranks | the same, quantizing each block as it materializes, text encoder included |
| `replicated` + fp8 at 8 ranks | rank 0 loads and broadcasts, quantizing before placement |

Every multi-rank case uses Ulysses 8, `torch.compile` is on throughout, the attention backend is
AITER, and each model is sampled the way its own benchmark config samples it:

| Model | Steps | Size (w×h) | Guidance | Seed |
| --- | --- | --- | --- | --- |
| FLUX.2-dev | 50 | 1024×1024 | 4.0 | 42 |
| Qwen-Image | 50 | 2048×2048 | 0.0 | 42 |
| FLUX.2-klein-9B | 4 | 2048×2048 | 1.0 | 42 |
| FLUX.1-dev | 25 | 1024×1024 | 0.0 | 42 |
| Z-Image | 50 | 1920×1088 | 4.0 | 42 |
| Z-Image-Turbo | 4 | 320×512 | 0.0 | 42 |

Environment: torch `2.9.1+gitff65f5b`, torchao `0.18.0+git92dcc9616`, transformers `5.5.4`, diffusers
`0.38.0.dev0`, AITER present, ROCm 6.16.2, Python 3.12.3, 8 × `gfx950` with 288 GB each on a
64-core EPYC 9575F. Page cache was dropped before every timed run, so the load times are cold-cache
and comparable to each other.

## Load time and load-time VRAM

Blockwise fill beats the eager baseline on both, on every model, and the margin grows with the model.

| Model | eager, 8 ranks | blockwise, 8 ranks | speedup | eager load VRAM | blockwise load VRAM | reduction |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.2-dev | 153.5s | 82.8s | 1.85x | 109.2G | 21.0G | 5.2x |
| Qwen-Image | 79.4s | 48.0s | 1.65x | 58.2G | 18.8G | 3.1x |
| FLUX.2-klein-9B | 46.2s | 33.2s | 1.39x | 36.7G | 16.1G | 2.3x |
| FLUX.1-dev | 47.8s | 32.3s | 1.48x | 35.8G | 15.6G | 2.3x |
| Z-Image | 29.5s | 25.5s | 1.16x | 23.9G | 13.9G | 1.7x |
| Z-Image-Turbo | 37.0s | 34.2s | 1.08x | 23.9G | 13.9G | 1.7x |

Load VRAM is the peak on the busiest single device while the load is in flight, not a sum over
devices, so the figures are comparable across rank counts. It is the number a memory-efficient load
exists to move: loading FLUX.2-dev eagerly at 8 ranks peaks at 109.2G, because every rank builds the
whole pipeline before FSDP takes the transformer apart, and 109.2G of a card's 288G is most of the
headroom inference then wants for activations.

## Host memory

| Model | eager anon | control anon | blockwise anon | control load VRAM |
| --- | --- | --- | --- | --- |
| FLUX.2-dev | 102.7G | 173.3G | 76.0G | 23.6G |
| Qwen-Image | 67.6G | 110.6G | 75.0G | 16.6G |
| FLUX.2-klein-9B | 99.2G | 170.0G | 82.7G | 15.0G |
| FLUX.1-dev | 102.1G | 187.6G | 86.2G | 15.2G |
| Z-Image | 118.9G | 126.6G | 77.1G | 11.3G |
| Z-Image-Turbo | 148.0G | 180.1G | 67.9G | 11.1G |

The control is the point of that table. Sharding after materialization gets the device saving —
15.2G against eager's 35.8G on FLUX.1-dev — and pays for it on the host, 187.6G against 102.1G,
because every rank holds a full copy while it waits to be cut down. Blockwise gets the same device
saving and *reduces* the host figure instead, because no rank ever holds a whole component.

Blockwise's host cost is also the flat one: 68–86G on all six models, where eager ranges 68–148G.
Qwen-Image is the one model where eager is already as cheap on the host as blockwise, so the 75.0G
there is not blockwise behaving oddly — it is Qwen's eager path being unusually cheap. Why eager
varies that much between models is a question about the stock loader, not about this branch.

Host anon is the container's anonymous pages, which is what the OOM killer watches. It is not summed
over ranks. Page-cache figures are excluded here on purpose: the kernel reclaims that under pressure,
so it is not a cost.

## The image is the same image

Every case's render, scored against the same model's single-rank eager render with `torch.compile`
disabled on both sides so the comparison is deterministic:

| Model | eager w8 | eager-fill w8 | blockwise w8 | fp8 blockwise | fp8 replicated |
| --- | --- | --- | --- | --- | --- |
| FLUX.2-dev | 0.9982 | 0.9982 | 0.9982 | 0.9006 | 0.9006 |
| Qwen-Image | 0.9920 | 0.9920 | 0.9920 | 0.5748 | 0.5748 |
| FLUX.2-klein-9B | 0.9735 | 0.9735 | 0.9735 | 0.8882 | 0.8821 |
| FLUX.1-dev | 0.9949 | 0.9949 | 0.9949 | 0.9741 | 0.9741 |
| Z-Image | 0.9364 | 0.9364 | 0.9364 | 0.6353 | 0.6353 |
| Z-Image-Turbo | 0.9935 | 0.9935 | 0.9935 | 0.7328 | 0.7244 |

SSIM, so 1.0 is identical. The three bf16 columns agree to four decimals on every model, which is the
claim this branch has to support: whatever separates an 8-rank render from a 1-rank one is the
parallelism, and the loading strategy contributes nothing on top of it. Only Z-Image-Turbo's row is
a verdict; the other models record the number as an observation, because a model that redraws the
prompt as a different sample under any numeric change cannot be judged by this metric. See
`IDENTITY_STABLE_MODELS` in `tools/gpu_validation.py` for that argument in full.

The fp8 column is quantization error, and how much of it lands in the image depends on the sampling
trajectory rather than on the load. Against its own bf16 row, fp8 costs FLUX.1-dev 0.021 over
twenty-five steps and Qwen-Image 0.417 over fifty at 2048² — and the Qwen render is a converged,
legible picture of the prompt, just a different sample of it. The two fp8 placements land within
0.009 of each other on every model, so replicated and blockwise quantization agree with each other far
more closely than either agrees with bf16, which is what you want from two routes to the same
quantized weights.

Wall time, for completeness, since a faster load is not worth much if the run gets slower:

| Model | eager w8 | blockwise w8 |
| --- | --- | --- |
| FLUX.2-dev | 235.8s | 165.5s |
| Qwen-Image | 138.4s | 103.3s |
| FLUX.2-klein-9B | 94.6s | 87.1s |
| FLUX.1-dev | 144.7s | 118.9s |
| Z-Image | 80.2s | 74.3s |
| Z-Image-Turbo | 77.5s | 72.3s |

## Two bugs this sweep found

Neither is in the load path. Both need a combination that nothing had run before: quantization at a
rank count high enough to leave a rank with nothing but padding, and sharding together with a
CUDA-graph compile mode, which CI only pairs on RDNA4 where the mode is `default`.

**Every FP8 render of Qwen-Image at eight ranks was pure black.** A dynamic per-tensor FP8 scale is
`max_abs / 448`, which is zero for a tensor of zeros, and quantizing divides by that scale, so the
layer returned NaN where it should have returned its bias. Zeros are not a corner case: a model that
zero-pads its text sequence up to a multiple of the sequence-parallel world size hands whole
padding-only chunks to late ranks, and the NaN then reached every rank through the attention
all-to-all. Fixed with torchao's `activation_value_lb`, set to the EPS torchao uses on its own
training path, at every site that builds the config. It only binds when the largest value in the
tensor is below it, so ordinary activations are measured bit-identical. In the four-step bisect that
isolated it, the case measured 0.0006 against bf16 at the same rank count before the fix and 0.9463
after.

**Every FLUX case with sharding and compile failed to run.** Sharding turns one compiled transformer
into one compiled block per layer, each its own CUDA graph segment; recording a later block reads the
previous block's output buffer, and the graph system refuses the read because it still holds the
previous step's outputs live. The FLUX runners ask for `reduce-overhead` off RDNA4, so this was the
whole FLUX column. Fixed with a forward pre-hook that announces the step boundary, which is the
remedy the error message names and the one the FLUX.2 cache adapter already applied for the same
reason.

The black frames also exposed a hole in the harness: nothing had looked at the image, because
reference comparison only gates models that reproduce their sample. Whether a picture exists needs no
reference and no per-model judgement, so a spread measurement now gates every case, and a run that
writes a uniform frame is `failed_blank_output` no matter what it exited with.

## One gate defect the scoring found

Scoring failed Z-Image-Turbo's two fp8 cases at 0.7244 against a floor of 0.80, and the images are
the same cat in the same pose with slightly different fur. That floor had been calibrated when every
model rendered 512×512 at seed 1234, where the same case measured 0.9227; sampling each model its own
way moved it and the floor was left behind. A floor only means something next to the sampling it was
measured at, which the calibration in `tools/image_quality.py` now records.

Recalibrating to a single looser floor would have given up the check that matters most here, since
sharding alone measures 0.9935 against the same reference: a floor loose enough for fp8 would pass a
shard that rendered a different picture. So a case that only moves weights is held to 0.90 and one
that quantizes them to 0.60, both clearing their measured result by a wide margin and both failing
the 0.0006 a collapsed render scores.

## What this does not show

- **Only bf16 and FP8.** No FP4, NVFP4 or INT8 case ran. On `gfx950` FP4 remains blocked with FSDP
  until torch 2.12, for the reason in the handoff.
- **Only 8 ranks, plus one spot check.** Z-Image-Turbo also loads, shards and renders correctly with
  `fsdp_blockwise` at 4 ranks. Nothing else was run at another rank count.
- **Only six models,** all image models with a standard transformer. No video model, no image-input
  model, no offload combination, and nothing that needs weights this node's cache does not hold.
- **Timings are one run each,** on a shared node with device-global VRAM sampling, so treat the VRAM
  figures as upper bounds and the times as indicative rather than as a benchmark.
- **The comparison is gross correctness,** not a quality assessment. It answers whether the model
  still drew the picture; it does not say fp8 output is as good as bf16 output.

## Reproducing it

The sweep runs one case per invocation with a shared run id, so the whole thing lands in one output
directory, with a page-cache drop between cases so the load times mean something:

```bash
python tools/gpu_validation.py --case <case-id> --execute --continue-on-error \
  --run-id my-sweep --results gpu-validation-results/results.jsonl
```

Scoring is a second pass over the same cases. It disables `torch.compile`, so its timings are marked
as scoring runs and kept out of the performance columns, and each case reuses the recorded
compile-free render of its model's `eager`/1-rank case, so run those first:

```bash
python tools/gpu_validation.py --case <case-id> --execute --continue-on-error --score-quality \
  --run-id my-scoring --results gpu-validation-results/results.jsonl
```

Then read both passes together, which is what the tables above came from:

```bash
python tools/validation_report.py --results gpu-validation-results/results.jsonl
```

The records were taken at `86c4113` with the working tree that became the six commits from
`d2a8745` to `8287cc9`. The eleven cases that first failed were re-run after the two fixes above, and
every scoring run is from after them. Result JSONL, logs and images are node-local and not checked
in; the performance pass is run id `20260808T081101Z-feature-evidence` and the scoring pass
`20260808T154216Z-feature-evidence-scoring`.
