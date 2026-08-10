# Memory-efficient load: measured results on MI355X

GPU end-to-end evidence for the memory-efficient load paths, taken on 8× AMD Instinct MI355X
(`gfx950`) across all nineteen models this node can load through them. Three sweeps: thirty-six
cases over six image models with a standard transformer, then seventy-three over the ten that had
never run — two that edit a picture, two that arrived upstream untested, and six video models — then
twenty over the three whose weights had to be fetched before they could run at all. Every case
passing, every still image scored against an unquantized render of the same prompt, every clip
checked frame by frame for having rendered anything. A final batch of eleven covers the other claim
these paths make: the models that refuse them, refusing before they allocate.

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
64-core EPYC 9575F with 3T of host memory. Page cache was dropped before every timed run, so the load times are cold-cache
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

## The second sweep: ten models that had never run

The first sweep took the six image models with a standard transformer. This one takes the rest of what
this node's cache holds: two models that edit a picture rather than drawing one, two that arrived
upstream and had never been run at all, and six video models. Seventy-three cases, of which
seventy-one ran and passed; the two that did not are curated FP4 cases
asserting a rejection that only holds on `gfx942`, and they correctly reported
`environment_mismatch` on this `gfx950` node.

Same node, same harness, one environment change: the container's from-source Diffusers was 248 commits
behind and had no `krea2` package at all, so it moved from `0.38.0.dev0` (`447e571ad`) to
`0.39.0.dev0` (`21ba39457`). Z-Image-Turbo was re-run on the new version and still passes, so treat
the first sweep's figures as taken on the older one rather than assuming both sweeps are one
experiment.

Each model is sampled its own way, as before. Five of these ten have no benchmark config at all, so
their entries cite their runner's own `DefaultInputValues`, and a test checks each citation against
what the runner declares so it cannot go stale:

| Model | Steps | Size (w×h) | Guidance | Frames | Sampling from |
| --- | --- | --- | --- | --- | --- |
| FLUX.1-Kontext-dev | 30 | 1024×1024 | 2.5 | — | benchmark config, plus an input image |
| Qwen-Image-Edit | 50 | 2048×2048 | 4.0 | — | benchmark config, plus an input image |
| Krea-2-Raw | 52 | 2048×2048 | 3.5 | — | runner declaration |
| Krea-2-Turbo | 8 | 2048×2048 | 0.0 | — | runner declaration, fixed schedule |
| HunyuanVideo | 50 | 1280×720 | 6.0 | 129 | benchmark config, tiled and sliced VAE |
| Wan2.1-T2V | 40 | 1280×720 | 3.5 | 81 | runner declaration |
| Wan2.1-I2V | 40 | 1280×720 | 3.0 | 81 | benchmark config, plus an input image |
| Wan2.2-T2V | 40 | 1280×720 | 3.5 | 81 | runner declaration |
| Wan2.2-I2V | 40 | 1280×720 | 3.5 | 81 | benchmark config, plus an input image; guidance from the runner, which is the one value that config leaves unset |
| Wan2.2-TI2V | 50 | 1280×736 | 5.0 | 121 | runner declaration, and `--task i2v`, which it requires |

The two input images are the ones these models' benchmark configs pass, and they ship in the
container; the matrix names them as environment placeholders so a case reports an environment
mismatch when they are absent rather than failing as if the load were broken.

The case counts are six per model, fifty-nine generated cases in all, plus fourteen curated ones — the
three FP4 Krea-2 cases and eleven Wan2.2-I2V cases that predate this sweep. Wan2.2-I2V is the one
model with five generated cases rather than six, because a curated single-rank eager case already
occupies that slot; it renders a 17-frame clip rather than 81, which matters only in that it is not a
like-for-like reference, and no video case is scored against a reference anyway.

### Load time and load-time VRAM

| Model | eager, 8 ranks | blockwise, 8 ranks | speedup | eager load VRAM | blockwise load VRAM | reduction |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.1-Kontext-dev | 47.7s | 29.7s | 1.61x | 35.8G | 15.6G | 2.3x |
| Qwen-Image-Edit | 80.7s | 45.4s | 1.78x | 58.2G | 18.8G | 3.1x |
| Krea-2-Raw | 48.6s | 38.7s | 1.26x | 36.7G | 15.4G | 2.4x |
| Krea-2-Turbo | 48.6s | 39.2s | 1.24x | 36.7G | 15.4G | 2.4x |
| HunyuanVideo | 54.7s | 41.3s | 1.32x | 42.6G | 29.2G | 1.5x |
| Wan2.1-T2V | 93.5s | 55.9s | 1.67x | 42.1G | 21.1G | 2.0x |
| Wan2.1-I2V | 102.3s | 50.8s | 2.01x | 47.3G | 21.1G | 2.2x |
| Wan2.2-T2V | 151.7s | 80.4s | 1.89x | 68.9G | 19.0G | 3.6x |
| Wan2.2-I2V | 152.1s | 80.0s | 1.90x | 69.0G | 19.1G | 3.6x |
| Wan2.2-TI2V | 41.2s | 33.3s | 1.24x | 25.6G | 16.3G | 1.6x |

The direction holds on every one of them, and the two Wan2.2 A14B models give up the largest absolute
saving measured in either sweep: 72 seconds of load time and 50G of load-time VRAM. They are also the
models that most need it, being the only ones here that carry two transformers.

### Host memory

| Model | eager anon | control anon | blockwise anon | reduction against control |
| --- | --- | --- | --- | --- |
| FLUX.1-Kontext-dev | 101.7G | 184.3G | 76.3G | 2.4x |
| Qwen-Image-Edit | 84.0G | 110.0G | 82.8G | 1.3x |
| Krea-2-Raw | 102.4G | 178.6G | 103.8G | 1.7x |
| Krea-2-Turbo | 103.6G | 177.1G | 103.0G | 1.7x |
| HunyuanVideo | 145.9G | 146.9G | 155.3G | 0.9x |
| Wan2.1-T2V | 328.4G | 343.0G | 88.4G | 3.9x |
| Wan2.1-I2V | 368.4G | 372.4G | 84.4G | 4.4x |
| Wan2.2-T2V | 493.5G | 522.9G | 84.0G | 6.2x |
| Wan2.2-I2V | 492.1G | 522.0G | 88.1G | 5.9x |
| Wan2.2-TI2V | 151.3G | 173.4G | 89.9G | 1.9x |

The video models are where the host figure starts to constrain what is possible. Loading Wan2.2-I2V
eagerly at eight ranks peaks at 492.1G of anonymous host memory. This node has 3T, so it fits here;
the same load does not fit on a 512G host, and the figure grows with rank count because every rank
holds its own copy. Blockwise holds it to 88.1G, and holds every model in this sweep except
HunyuanVideo between 76G and 104G, close to the 68–86G the first sweep measured across models a
quarter the size. That flatness is the point: the host cost is a property of the strategy rather than
of the model.

HunyuanVideo is the exception and worth stating plainly: 145.9G eager against 155.3G blockwise, no
improvement. Its load declaration covers the transformer alone, and the fill logs show that part
working, holding host anon flat at 51.9G for the whole transformer fill. The peak is elsewhere — its
Llama text encoder, which every rank loads in full in all three cases. That is the next component
worth bringing inside this path, not a defect in it, and it is why load-time VRAM still improves 1.5x
while the host figure does not. It was brought inside afterwards, and
[the section below](#hunyuanvideos-text-encoder-brought-inside-the-path) has the figures; the row
above is what the model measured before that change.

### Did it draw anything

Every video case's clip is read frame by frame, twelve evenly spaced samples of it, so a clip that
renders one good frame and then collapses cannot average its way past the floor:

| Model | Frames | Flattest sampled frame |
| --- | --- | --- |
| HunyuanVideo | 129 | 0.2149 |
| Wan2.1-T2V | 81 | 0.2852 |
| Wan2.1-I2V | 81 | 0.2502 |
| Wan2.2-T2V | 81 | 0.2868 |
| Wan2.2-I2V | 81 | 0.2487 |
| Wan2.2-TI2V | 121 | 0.2560 |

Blockwise cases shown; the floor is 0.01, so the closest any real clip came to it was twenty times
above. That margin is the useful part: it says the gate has room to catch a collapse without
threatening a legitimately flat render.

### The image is the same image

The four image models, scored against their own single-rank eager render with `torch.compile`
disabled on both sides:

| Model | eager w8 | eager-fill w8 | blockwise w8 | fp8 blockwise | fp8 replicated |
| --- | --- | --- | --- | --- | --- |
| FLUX.1-Kontext-dev | 0.9931 | 0.9931 | 0.9931 | 0.8099 | 0.8099 |
| Qwen-Image-Edit | 0.9873 | 0.9873 | 0.9873 | 0.9495 | 0.9495 |
| Krea-2-Raw | 0.9344 | 0.9344 | 0.9344 | 0.9403 | 0.9403 |
| Krea-2-Turbo | 0.8576 | 0.8576 | 0.8576 | 0.6596 | 0.6597 |

Every number is an observation rather than a verdict, since none of these four is in
`IDENTITY_STABLE_MODELS`. The claim they support is the same as before and it holds again: the three
bf16 columns agree to four decimals on every model, so the loading strategy contributes nothing on
top of whatever separates an 8-rank render from a 1-rank one, and the two fp8 placements agree with
each other to the fourth decimal or better. Krea-2-Raw scoring marginally *higher* under fp8 (0.9403)
than in bf16 (0.9344) is the clearest available reminder of what these numbers are on a model that
redraws its sample: noise around a different picture of the same prompt, not a measurement of
quantization error.

No video case is scored. SSIM over a clip would only decide identity-stable models and no video model
has been measured to be one, so those cases record as `unscored` and rely on the frame gate above.

### Wall time

| Model | eager w8 | blockwise w8 |
| --- | --- | --- |
| FLUX.1-Kontext-dev | 102.2s | 78.1s |
| Qwen-Image-Edit | 361.5s | 118.3s |
| Krea-2-Raw | 161.2s | 100.0s |
| Krea-2-Turbo | 103.5s | 84.8s |
| HunyuanVideo | 279.7s | 218.0s |
| Wan2.1-T2V | 279.5s | 188.4s |
| Wan2.1-I2V | 235.1s | 185.8s |
| Wan2.2-T2V | 348.9s | 285.1s |
| Wan2.2-I2V | 392.5s | 289.6s |
| Wan2.2-TI2V | 153.2s | 115.4s |

## HunyuanVideo's text encoder, brought inside the path

The anomaly above turned out to be a gap in what the model declared rather than in the path. Both
meta paths pick their components out of `fsdp_strategy`, and HunyuanVideo named only its transformer,
so its 14G Llama encoder was loaded whole by every rank in all five cases — 112G of the peak at eight
ranks, against a transformer that was already filling one block at a time. Declaring the encoder, and
handing the pipeline the meta module the declaration produces, is the whole change; the path itself
needed nothing. Its 0.2G CLIP encoder stays out, as FLUX's does, because a collective per prompt to
save a fraction of a gigabyte is not worth having.

The same six cases, before the change and after:

| Case | host anon | load VRAM | load time |
| --- | --- | --- | --- |
| eager, 8 ranks | 145.9G → 145.8G | 42.6G → 42.4G | 54.7s → 54.1s |
| shard-after-materialize control | 146.9G → 159.2G | 26.2G → 13.9G | 55.0s → 55.4s |
| blockwise | 155.3G → 102.2G | 29.2G → 15.4G | 41.3s → 35.9s |
| fp8 blockwise | 155.3G → 104.6G | 28.0G → 14.6G | 41.6s → 37.0s |
| fp8 replicated | 156.4G → 105.5G | 35.6G → 34.9G | 40.4s → 29.2s |

Blockwise now holds host anon to 102.2G, inside the 76–104G band every other model in this sweep sits
in, and 1.6x under its own control. Load-time VRAM improves twice over: the encoder is never whole on
any rank, so 42.4G eager becomes 15.4G, where before the change it was 29.2G. The model renders the
same clip, and the frame gate reads it the same way, 0.2149 on the flattest of twelve sampled frames
against a 0.01 floor.

Two rows deserve reading rather than skimming. The eager row is unchanged, which is the point of it:
declaring a component tells the meta paths and the sharding step about it and leaves an ordinary load
alone. The control row got 12G *worse*, and that is correct — it now materializes the encoder per rank
and then FSDP-wraps it, paying a transient the old control never paid, which is exactly the cost
blockwise avoids. A control that covered fewer components than the case it is a control for was
flattering the comparison.

The two fp8 cases changed identity as well as numbers. A model with declared text-encoder FP8 targets
gets a `te_fp8` case rather than a transformer-only one, so `hunyuanvideo-fp8-fsdp-w8` and
`-fp8-replicated-w8` are now `-fp8-te-fsdp-w8` and `-fp8-te-replicated-w8`, and the encoder is
quantized per block as the fill materializes it. That is the fourteenth model on that path and the
first Llama encoder on it. Its clip is a different sample of the same scene rather than a collapse,
which is all a video case is checked for here.

## Three models that had never run at all

Sixteen models had run on these paths and three had not, for one reason that looked like three: the
weights were not on this node. Each of those three declares the memory-efficient load and each had
generated cases sitting in the matrix, but no sampling entry, because an entry is copied from the
model's own configuration and recording an operating point for a model nobody can run is recording a
claim nobody can check. The harness refuses a case with no entry rather than borrowing another
model's, so the cases could not run even as failures. Fetching 192G of weights, 24G for
FLUX.2-klein-4B and 35G and 133G for the two Cosmos3 models, turned that from a permanent gap into
three sampling entries and twenty runnable cases. All twenty pass: eight on klein-4B, six on
Cosmos3-Nano and six on Cosmos3-Super.

None of the three is named by a benchmark config, so all three take their operating point from their
runner's own `default_input_values`, which the entries say. For Cosmos3 the prompt comes from the
checkpoint: it ships `assets/example_t2v_prompt.json` and a matching negative, and the runner reads a
`.json` prompt path and serializes it because the model asks for structured prompts rather than
prose. A sentence would have been the wrong input rather than a different one.

**Cosmos3 has no text encoder, and needed none of the work HunyuanVideo needed.** Reading its
`model_index.json` before its weights arrived settled a question worth settling early: the pipeline
names a `text_tokenizer`, a `vision_encoder`, a transformer, a VAE and a sound tokenizer, and its
`__init__` takes a tokenizer where every other model here takes an encoder. Text is tokenized through
a chat template and consumed by the transformer directly. Its non-transformer components come to
4.6G against a 128G transformer, so there is no encoder-shaped saving to declare, and the transformer
the path already covers is the whole of it.

| Case, 8 ranks | host anon | load VRAM | load time |
| --- | --- | --- | --- |
| Cosmos3-Super eager | 110.2G | 137.7G | 41.9s |
| Cosmos3-Super shard-after-materialize control | 114.0G | 32.6G | 84.1s |
| Cosmos3-Super blockwise | 95.6G | 31.0G | 74.7s |
| Cosmos3-Super fp8 blockwise | 102.5G | 24.2G | 67.2s |
| Cosmos3-Super fp8 replicated | 123.5G | 82.7G | 72.5s |
| Cosmos3-Nano eager | 110.6G | 40.1G | 11.4s |
| Cosmos3-Nano shard-after-materialize control | 119.6G | 17.8G | 69.7s |
| Cosmos3-Nano blockwise | 91.1G | 19.0G | 24.1s |
| Cosmos3-Nano fp8 blockwise | 102.7G | 17.7G | 21.9s |
| Cosmos3-Nano fp8 replicated | 115.3G | 28.0G | 25.4s |
| klein-4B eager | 93.0G | 20.6G | 13.7s |
| klein-4B shard-after-materialize control | 109.1G | 11.4G | 16.6s |
| klein-4B blockwise | 83.3G | 15.1G | 19.1s |
| klein-4B fp8-te blockwise | 99.0G | 14.7G | 21.1s |
| klein-4B fp8-te replicated | 91.7G | 21.5G | 24.0s |

Cosmos3-Super is where the device-memory saving is largest of anything in this document: 137.7G of
load-time VRAM eager against 31.0G blockwise, 4.4x, and 24.2G with FP8, 5.7x, which is what a 128G
transformer filled one block at a time buys. Its host anon improves by much less, 110.2G to 95.6G,
and that is the same story as the paragraph above — there is no encoder to bring inside, so the
transformer is the only component the fill touches. Cosmos3-Nano halves its load VRAM, 40.1G to
19.0G, and holds host anon 1.21x under eager and 1.31x under its own control. klein-4B's margins are
smaller again because the model is smaller, a 7.8G transformer beside its encoder. All three hold the
shape the rest of the sweep has: eager is cheapest to load and dearest in host memory, the control is
dearest of all, and blockwise is the only row better than eager on both counts.

Super's replicated row is the one to read carefully: 82.7G of load-time VRAM against 31.0G blockwise,
and the highest host anon of its six cases at 123.5G. That is the shape of the path rather than a
defect — rank 0 loads the component whole and broadcasts it, so a 128G transformer is 128G on one rank
before anything is scattered, and the saving is against eager's per-rank copy rather than against a
blockwise fill. On a transformer this size the difference between the two memory-efficient paths
stops being a detail.

**One bug this found, in the offline path rather than in a model.** klein-4B ships its transformer as
a single `diffusion_pytorch_model.safetensors` with no shard index, and discovery asks for the index
first and falls through to the single file when the hub answers that no such file exists. With no
network there is no such answer: a name the repo does not carry and a name that was never cached
raise the same `LocalEntryNotFoundError`, and discovery treated both as a real error, so all four
memory-efficient cases died before loading while eager and the control passed. The component's own
cached `config.json` tells the two apart — if that is on disk then this snapshot is on disk, and the
missing name is one this checkpoint does not use. This would have hit any single-file component
loaded offline, and every model run before this one happens to ship a sharded index.

## The refusals, made executable

Everything above is a model loading. This last batch is the opposite claim: the models that withhold
the memory-efficient load, refusing it before they allocate. Thirteen runners declare eager loading
only, each with a written reason, and until now the reasons were only declared. Nothing ran them, so
nothing would have noticed a change that quietly started loading one of them through a path its own
declaration says has never been verified for it — and the failure mode there is not a crash but wrong
weights in a render nobody would question.

Nine cases now assert those refusals, on the seven withheld models whose weights are on this node.
They cost seconds, not loads: the load contract is selected immediately after distributed init, so
MiniMax-H3 refuses without reading any of its 330G. Each case matches on the reason, not merely on
failing, and a test compares the pattern against the message the runner's declaration produces, so a
pattern that would accept any failure does not pass.

| Case | Refused with |
| --- | --- |
| `rocm-ideogram4-fsdp-withheld` | a single-file state dict applied outside `_build_transformer`, and a second denoiser in its own subfolder |
| `rocm-lingbot-dense-fsdp-withheld` | the composed `_build_pipe` construction it shares with the MoE runner |
| `rocm-lingbot-moe-fsdp-withheld` | LingBot's own per-block wrapping, which leaves fp32 norm and router parameters where xDiT's path would shard them |
| `rocm-minimax-h3-fsdp-withheld` | `fuse_qkv_projections` renaming attention weights, so live names stop matching checkpoint keys |
| `rocm-minimax-h3-ref2va-fsdp-withheld` | the same, plus a denoiser loaded as `transformer_ref` |
| `rocm-ltx23-fsdp-withheld` | stage 2 distilled LoRA applied before a meta transformer would get its base checkpoint |
| `rocm-hunyuan15-sparse-replicated-withheld` | base non-block weights composed with remapped Tencent sparse blocks outside `_build_transformer` |
| `rocm-hunyuan15-sparse-fsdp-flag-refused` | no `fully_shard_degree` capability at all, refused while the config is validated |
| `rocm-zimage-int8-rejected` | INT8 on ROCm, which was already curated and had not been run |

Writing them turned up three things worth keeping. The harness could not run a rejection case for a
model with no sampling entry: the entry is read while the command is built, before anything hardware
knows about, so the four rejection cases that already existed for withheld models would have reported
"cannot run" and asserted nothing had anyone put them on the right hardware. The withheld models with
weights here now carry entries from their runners' own declarations.

Those declarations then exposed a gap in what an entry can say. Ideogram-4 builds its own guidance
schedule and only when it is given no guidance value, so every number, the CLI default included, is
wrong for it; MiniMax-H3 is guidance-distilled and its runner forwards no guidance at all. An entry
may now say `null`, which passes no flag, while still being required to say something.

And two of the models refuse earlier than their declarations do, which changed what the cases assert.
The HunyuanVideo 1.5 runners declare no `fully_shard_degree` capability, so the sharding flag is
refused while the config is validated, before a load contract exists; that refusal is stronger than
the withheld reason, since it holds however the declaration later changes, so it is asserted on its
own and the withheld reason is asserted on the replicated path instead. The same runner also refuses
to run without a sparse attention backend, which is why its case names one.

Two cases that had been curated but never run also ran here: FLUX.2-dev FP8 eager, and Qwen-Image FP8
on the replicated path with an FP8 text encoder. Both passed. That leaves twenty-six cases in the
matrix unrun, every one of them naming an accelerator this node is not.

The one unexplained failure in the results file is explained too. `gen-mi3xx-z-image-turbo-bf16-fsdp-w4`
was recorded as `failed_inference` before the port-collision fix, and the generator no longer emits a
four-rank case for that model, so it could not be re-run through the harness. Run by hand at four
ranks, it fills and shards block by block and renders in 6.9s. The failure was the socket.

## What stood between these models and a first run

Five things had to change before these models would run, on top of the Diffusers upgrade above. None of
them could have been found by a case that had already run: each needed a model, a placement or an
environment nothing had combined before. Four blocked a load outright; the fifth made working loads
look broken.

**Krea-2 could not be loaded from weights it already had.** Both variants are cached in full, 34G
each, but the repo refuses file downloads to this node's token, and Diffusers skips its
"does the repo have these shards" metadata call only when `local_files_only` is set. Exporting
`HF_HUB_OFFLINE` was not enough, because that variable does not reach the decision. A checkpoint
request now takes its default from it, which is the standard way to say the network is not there.

**Every sharded Krea-2 case died before reaching a load.** The FSDP strategy named `model.layers` for
its text encoder, which is where a text-only encoder keeps its decoder layers; Krea-2's encoder is a
`Qwen3VLModel`, which holds them under `language_model` beside the vision tower. Eager and replicated
cases never walk that path, so twelve passing cases said nothing about it. The test now checks the
declared path against the class Transformers actually builds, because a wrap path is a claim about
another library's module layout.

**HunyuanVideo could not load at all.** Transformers v5 parses a model's root `config.json` for an
unrelated Mistral regex fix, and once offline mode is set it does that for every model rather than
only official Mistral repos. HunyuanVideo ships a root `config.json` with a trailing comma, which is
not JSON, so the tokenizer reload raised `JSONDecodeError` — on a file no loader here needs and which
is not ours to fix. The reload now reads the tokenizer's own directory, which has no `config.json` to
trip over.

**Wan2.2-TI2V rejected all six of its cases.** It is the one runner serving both `i2v` and `t2v`, so
it requires the task to be named, and the sampling entry gave it an input image but no task.

**Four cases were recorded as `failed_inference` when a socket was the problem.** Back-to-back runs
collided on torchrun's default port 29500, and because the affected placements differed between two
runs of the same set, the pattern read as a nondeterministic bug in the sharded load. Each case now
asks the OS for a free rendezvous port. This one is worth remembering as a reading error as much as a
bug: three of those four cases had already been fixed by the change above them, and the port
collision hid it.

## What this node needed, which is not in the repository

Two facts about this machine's cache, recorded because the runs depended on them and the next operator
will not guess them:

Every case in the second sweep ran with `HF_HUB_OFFLINE=1`. All the weights are on disk, two of the
repos refuse downloads to this token, and a metadata round trip is variance in a load-time measurement
for nothing. It also means the offline path above is what these figures were taken through.

`krea/krea-2-turbo`'s cache held a complete snapshot of one revision while `refs/main` named a
different one, so nothing could resolve it offline; the ref was pointed at the snapshot that is
actually there. All fifty repos on this node were checked for that inconsistency and one other has it,
`tencent/HunyuanVideo`, which needs no repair because its runner pins `refs/pr/18` and that is the
revision on disk. Nothing else in the cache was changed.

## What this does not show

- **Only bf16 and FP8, plus three FP4 cases on Krea-2.** Those three are curated cases that assert
  what `gfx950` can do: FP4 eager and FP4 replicated load and render, FP4 with FSDP is rejected in
  preflight, which is the torch 2.12 limit described in the handoff.
- **Only 8 ranks, plus spot checks.** Z-Image-Turbo loads, shards and renders with `fsdp_blockwise` at
  4 ranks, klein-4B covers `fsdp_blockwise` and `replicated` in FP8 at 4 ranks, and the curated
  Wan2.2-I2V cases cover 2 and 4 ranks. Nothing else was run at another rank count.
- **Nineteen models,** which is every model this node has usable weights for that these paths can
  load. Thirteen registered runners are withheld from the memory-efficient load by their own
  declarations, each with its reason recorded, and six have no usable weights on this node at all. No
  offload combination was run. `tools/cache_inventory.py` and `tools/load_support_matrix.py` produce
  those three lists.
- **Video is gated, not compared.** A video case is checked for having rendered something, frame by
  frame; nothing checks that a sharded clip matches a single-rank clip, because SSIM over a clip only
  decides identity-stable models and no video model has been shown to be one.
- **Timings are one run each,** on a shared node with device-global VRAM sampling, so treat the VRAM
  figures as upper bounds and the times as indicative rather than as a benchmark. The two sweeps also
  ran on different Diffusers commits, so compare within a sweep rather than across them.
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

The first sweep's records were taken at `86c4113` with the working tree that became the six commits
from `d2a8745` to `8287cc9`. The eleven cases that first failed were re-run after the two fixes above,
and every scoring run is from after them. Its performance pass is run id
`20260808T081101Z-feature-evidence` and its scoring pass `20260808T154216Z-feature-evidence-scoring`.

The second sweep's records were taken with the working tree that became the seven commits from
`bef8aab` onwards, on Diffusers `21ba39457`, with `HF_HUB_OFFLINE=1` and the two input image paths
exported. Every case that failed was re-run after the change that fixed it, so no figure above comes
from a run predating the change its own model needed. Its scoring pass covers the four image models;
the six video models have no scored column by design. The passes are run ids
`20260809T085227Z-coverage-images`, `20260809T100205Z-coverage-krea-fixed`,
`20260809T101323Z-coverage-krea-fixed2`, `20260809T102509Z-coverage-video`,
`20260810T062730Z-coverage-video-rerun` and `20260810T071614Z-coverage-score`. Result JSONL, logs,
images and clips are node-local and not checked in.

HunyuanVideo's six cases were then re-run once more, on the same node and Diffusers commit, with the
text-encoder declaration in place; that pass is run id `20260810T085256Z-hunyuan-te` and is where the
before-and-after table in its own section comes from. Its "before" column is the second sweep's
figures above, so those two columns are one comparison rather than two sweeps.

The three models that had never run were fetched and run last, on the same node and commit, as run ids
`klein4b-coverage`, `klein4b-offline-fix`, `cosmos3-nano-coverage`, `cosmos3-super-coverage` and
`cosmos3-super-finish`. Four of klein-4B's eight cases failed on the offline single-file bug above and
were re-run after the fix, which is the second run id. Cosmos3-Super's sweep was interrupted to free
the node and its last three cases finished under the fifth; the record the interruption produced was
removed from the results file rather than left to read as a failure, since it is evidence about the
interruption and not about the load. The Cosmos3 cases need the checkpoint's own prompt assets
exported, and the two Cosmos3 repos are byte-identical in those two files:

```bash
export XDIT_COSMOS3_PROMPT_JSON=<snapshot>/assets/example_t2v_prompt.json
export XDIT_COSMOS3_NEGATIVE_PROMPT_JSON=<snapshot>/assets/negative_prompt.json
```
