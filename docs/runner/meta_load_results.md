# Memory-efficient load: measured results

Two load paths are measured here against the two ways of loading that already existed. `fsdp_blockwise`
builds on meta and fills and shards one block at a time; `replicated` has rank 0 read each block and
broadcast it, quantizing before placement. They are measured against `eager`, which builds every
component in full on every rank, and `fsdp_eager_fill`, which does the same and then shards it — what a
naive FSDP load does, and the control. The claim under test is that the first two hold a load inside a
device and a host that the other two do not, and draw the same picture doing it.

They do, on every one of the twenty models this node can load through them. Blockwise beats eager on
load-time VRAM everywhere, by 1.4x to 5.2x, and the margin grows with the model: FLUX.2-dev peaks at
109 GB loaded eagerly at eight ranks and 21 GB filled block by block, and Cosmos3-Super, whose
transformer is 128 GB, needs 128 GB eagerly against 29 filled. It also beats eager on host memory, which
the control does not — sharding after materializing buys the device saving and pays for it on the host, up
to 522 GB on Wan2.2-I2V where blockwise holds 88 GB, because every rank holds a whole copy while it waits
to be cut down. Blockwise's host cost is flat across models, 68–104 GB on nineteen of the twenty, which is the
useful property: it is a cost of the strategy rather than of the model. It is also faster than eager on
seventeen of the twenty, and faster than the control on nineteen, so none of this is bought with load time.
On the images, eager, the control
and blockwise are indistinguishable: scored against the same single-rank render they agree to every digit
the comparison reports, on all ten models scored, so the loading strategy costs nothing in quality.

Read this with [the validation handoff](gpu_validation_handoff.md) for how the harness plans, runs and
scores a case, and [the meta-load handoff](meta_load_handoff.md) for the node's setup.

## MI355X (`gfx950`), 8 devices

161 of the matrix's 183 cases have run, all of them on this node: 144 passed, 15 asserted a refusal, 2
report an environment mismatch for wanting `gfx942` alone. torch `2.9.1+gitff65f5b`, torchao `0.18.0+git92dcc9616`,
transformers `5.5.4`, diffusers `0.39.0.dev0` (`21ba39457`, the first sweep on `447e571ad`), AITER
present, ROCm 6.16.2, Python 3.12.3, 288 GB per device, 3 TB host, `HF_HUB_OFFLINE=1`.

Each table below reports one measurement across every model, with the same five columns: one per
placement. The placements, in the order the columns run:

| Column (placement) | How the weights get to the device | Flags |
| --- | --- | --- |
| `eager` | Every component built in full on every rank. The baseline. | none |
| `repl fp8` | Rank 0 reads each block, quantizes it and broadcasts it. Streamed, but every rank still ends up with a full copy. | `--memory_efficient_replicated_load --use_fp8_gemms` |
| `fsdp eager`<br>(`fsdp_eager_fill`) | Built in full on every rank, then sharded — what a naive FSDP load does, and the control for the two columns beside it. | `--fully_shard_degree N` |
| `fsdp block`<br>(`fsdp_blockwise`) | Built on meta, then filled and sharded one block at a time, so no rank ever holds the component whole. The streamed load. | `--fully_shard_degree N --memory_efficient_sharding` |
| `block fp8` | `fsdp block`, quantizing each block on the way in. | the above plus `--use_fp8_gemms` |

The first two columns leave every rank holding a full copy and the last three shard through FSDP, which is
the order the columns run in. Ulysses across all ranks, `torch.compile` on, AITER attention. Every model at
eight ranks except Ideogram-4 at six, whose 18 attention heads do not divide eight ways. There is no
quality table: every placement of a model scores the same, so a per-model number would report on the model
rather than on the load, as [below](#how-to-read-the-numbers).

**Bold marks the lowest figure in the row**, and it is worth knowing what a win costs before reading it as
one: `fsdp eager` takes the VRAM row on thirteen models by keeping the full copy on the host instead, which
the second table charges it for, and the two `fp8` columns buy their figures with quantization error the
bf16 columns do not carry. Only `fsdp block` against `fsdp eager` is a like-for-like comparison.

**Load-time VRAM**, the peak on the busiest device while the load is in flight, in GB. The figure these
paths exist to move:

| Model | eager | repl fp8 | fsdp eager | fsdp block | block fp8 |
| --- | --- | --- | --- | --- | --- |
| Cosmos3-Nano | 37 | 26 | 17 | 18 | **16** |
| Cosmos3-Super | 128 | 77 | 30 | 29 | **23** |
| FLUX.1-dev | 36 | 29 | **15** | 16 | 19 |
| FLUX.1-Kontext-dev | 36 | 29 | **13** | 16 | 15 |
| FLUX.2-dev | 109 | 87 | 24 | **21** | 34 |
| FLUX.2-klein-4B | 19 | 20 | **11** | 14 | 14 |
| FLUX.2-klein-9B | 37 | 33 | **15** | 16 | 18 |
| HunyuanVideo | 42 | 35 | **14** | 15 | 15 |
| Ideogram-4 | 55 | 41 | 28 | 30 | **26** |
| Krea-2-Raw | 37 | 30 | **14** | 15 | 15 |
| Krea-2-Turbo | 37 | 30 | **15** | **15** | **15** |
| Qwen-Image | 58 | 43 | 17 | 19 | **16** |
| Qwen-Image-Edit | 58 | 43 | **16** | 19 | **16** |
| Wan2.1-I2V | 47 | 36 | **16** | 21 | 21 |
| Wan2.1-T2V | 42 | 33 | **14** | 21 | 21 |
| Wan2.2-I2V | 69 | 48 | 18 | 19 | **17** |
| Wan2.2-T2V | 69 | 48 | 18 | 19 | **17** |
| Wan2.2-TI2V | 26 | 26 | **13** | 16 | 16 |
| Z-Image | 24 | 22 | **11** | 14 | 13 |
| Z-Image-Turbo | 24 | 23 | **11** | 14 | 13 |

**Host anonymous memory**, the peak over the same window, in GB. Where `fsdp eager` pays for the device
figures it just won:

| Model | eager | repl fp8 | fsdp eager | fsdp block | block fp8 |
| --- | --- | --- | --- | --- | --- |
| Cosmos3-Nano | 103 | 107 | 111 | **85** | 96 |
| Cosmos3-Super | 103 | 115 | 106 | **89** | 95 |
| FLUX.1-dev | 102 | 91 | 188 | **86** | 87 |
| FLUX.1-Kontext-dev | 102 | 80 | 184 | **76** | 81 |
| FLUX.2-dev | 103 | 93 | 173 | **76** | 88 |
| FLUX.2-klein-4B | 87 | 85 | 102 | **78** | 82 |
| FLUX.2-klein-9B | 99 | 100 | 170 | **83** | 92 |
| HunyuanVideo | 146 | 106 | 159 | **102** | 105 |
| Ideogram-4 | 424 | **182** | 433 | 186 | 194 |
| Krea-2-Raw | **102** | 106 | 179 | 104 | 104 |
| Krea-2-Turbo | 104 | 106 | 177 | **103** | 104 |
| Qwen-Image | **68** | 84 | 111 | 75 | 78 |
| Qwen-Image-Edit | 84 | 107 | 110 | **83** | 88 |
| Wan2.1-I2V | 368 | 99 | 372 | **84** | 99 |
| Wan2.1-T2V | 328 | 99 | 343 | **88** | 105 |
| Wan2.2-I2V | 492 | 93 | 522 | **88** | 96 |
| Wan2.2-T2V | 493 | **84** | 523 | **84** | 94 |
| Wan2.2-TI2V | 151 | 105 | 173 | **90** | 100 |
| Z-Image | 119 | 79 | 127 | **77** | **77** |
| Z-Image-Turbo | 148 | 72 | 180 | **68** | 74 |

**Load duration**, in seconds. One run each on a shared node, so read these as indicative:

| Model | eager | repl fp8 | fsdp eager | fsdp block | block fp8 |
| --- | --- | --- | --- | --- | --- |
| Cosmos3-Nano | **11** | 25 | 70 | 24 | 22 |
| Cosmos3-Super | **42** | 73 | 84 | 75 | 67 |
| FLUX.1-dev | 48 | 36 | 53 | **32** | 35 |
| FLUX.1-Kontext-dev | 48 | 35 | 51 | **30** | 32 |
| FLUX.2-dev | 154 | 93 | 152 | 83 | **81** |
| FLUX.2-klein-4B | **14** | 24 | 17 | 19 | 22 |
| FLUX.2-klein-9B | 46 | 37 | 51 | **33** | 34 |
| HunyuanVideo | 54 | **29** | 55 | 36 | 37 |
| Ideogram-4 | 234 | 177 | 229 | 172 | **170** |
| Krea-2-Raw | 49 | **38** | 53 | 39 | 39 |
| Krea-2-Turbo | 49 | **38** | 51 | 39 | 39 |
| Qwen-Image | 79 | **36** | 83 | 48 | 50 |
| Qwen-Image-Edit | 81 | 50 | 84 | **45** | **45** |
| Wan2.1-I2V | 102 | **43** | 105 | 51 | 50 |
| Wan2.1-T2V | 94 | **45** | 98 | 56 | 57 |
| Wan2.2-I2V | 152 | 84 | 158 | **80** | 82 |
| Wan2.2-T2V | 152 | 85 | 154 | **80** | 82 |
| Wan2.2-TI2V | 41 | 35 | 46 | **33** | **33** |
| Z-Image | 30 | 27 | 34 | **26** | **26** |
| Z-Image-Turbo | 37 | 32 | 43 | 34 | **31** |

Five things in those tables need a note rather than a second look. **Ideogram-4** is the only model with a component
outside the path — its text encoder is built through `AutoModel` with `trust_remote_code`, so no
manifest can know its names ahead of the load — and its 424 GB eager host figure is that encoder on
every rank; blockwise still cuts it to 186. **The control is the slowest placement**, not just the
hungriest: it is behind eager on eighteen of the twenty models, because it does eager's work and then a
sharding pass, and on Cosmos3-Nano that is 70s against 11s. So the sharding it adds to eager buys device
memory at a cost in host memory and in time, which is the whole reason to fill blockwise instead.
**Three models load slower** blockwise than eager. Cosmos3-Nano at 24s against 11s and klein-4B at 19s
against 14s are the fill's per-block collective costing more than it saves on a transformer small enough
to materialize, and both still cut their load VRAM, Nano by half and klein-4B, the smallest model here, by
a third. Cosmos3-Super is the third, 75s against 42s, which that explanation does not cover on the largest
transformer in the table; with one run per cell it is a thread to pull rather than a result. **The `block fp8`
column also fills and quantizes the text encoder** wherever the model declares targets for it, so it is
not a pure quantization delta, and on FLUX.2-dev, klein-9B and FLUX.1-dev that makes the quantized fill
dearer than the bf16 one. **`repl fp8` is consistently the dearest of the three** memory-efficient
columns, and on Cosmos3-Super dramatically so at 77 GB, because rank 0 holds the component whole before
broadcasting it. On a 128 GB transformer the choice between the two paths stops being a detail.

The other 43 cases cover what the grid does not: quantization formats, offload modes, rank counts, a
library version, and the refusals.

| Case | Model | Combination | load VRAM | load s | Outcome |
| --- | --- | --- | --- | --- | --- |
| `rocm-flux1-fp8-eager-sequential-offload` | FLUX.1-dev | eager/fp8/w1+offload=sequential | 15 | 27 | pass |
| `rocm-flux1-fp8-replicated-sequential-offload` | FLUX.1-dev | replicated/fp8/w2+offload=sequential | 24 | 18 | pass |
| `rocm-flux2-fp8-eager` | FLUX.2-dev | eager/fp8/w1 | 79 | 113 | pass |
| `rocm-flux2-fp8-eager-te-tf4` | FLUX.2-dev | eager/fp8/w1+te, Transformers 4 | 80 | 30 | pass |
| `rocm-flux2-fp8-eager-te-tf5` | FLUX.2-dev | eager/fp8/w1+te, Transformers 5 | 58 | 62 | pass |
| `rocm-flux2-fsdp-te-tf4-rejected` | FLUX.2-dev | fsdp_blockwise/fp8/w2+te, Transformers 4 | - | - | refused |
| `rocm-flux2-hybrid-eager` | FLUX.2-dev | eager/hybrid FP8+FP4/w1 | 93 | 33 | pass |
| `rocm-klein4b-bf16-eager` | FLUX.2-klein-4B | eager/bf16/w1 | 16 | 5 | pass |
| `rocm-klein4b-fp8-fsdp4` | FLUX.2-klein-4B | fsdp_blockwise/fp8/w4 | 11 | 16 | pass |
| `rocm-klein4b-fp8-replicated4` | FLUX.2-klein-4B | replicated/fp8/w4 | 17 | 19 | pass |
| `rocm-hunyuan15-sparse-fsdp-flag-refused` | Hunyuanvideo-1.5-Sparse | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-hunyuan15-sparse-replicated-withheld` | Hunyuanvideo-1.5-Sparse | replicated/bf16/w2 | - | - | refused |
| `rocm-ideogram4-text-encoder-excluded` | Ideogram-4 | fsdp_blockwise/bf16/w2 | 45 | 163 | pass |
| `rocm-gfx950-krea2-fp4-eager` | Krea-2-Turbo | eager/fp4/w1 | 17 | 23 | pass |
| `rocm-gfx950-krea2-fp4-fsdp4` | Krea-2-Turbo | fsdp_blockwise/fp4/w4 | - | - | refused |
| `rocm-gfx950-krea2-fp4-replicated4` | Krea-2-Turbo | replicated/fp4/w4 | 21 | 34 | pass |
| `rocm-krea2-fp4-eager-group-offload-rejected` | Krea-2-Turbo | eager/fp4/w1+offload=group | - | - | refused |
| `rocm-krea2-fp4-replicated-group-lowcpu-rejected` | Krea-2-Turbo | replicated/fp4/w2+offload=group low-cpu | - | - | refused |
| `rocm-krea2-fp8-eager-group-offload` | Krea-2-Turbo | eager/fp8/w1+offload=group | 16 | 39 | pass |
| `rocm-ltx23-fsdp-withheld` | LTX-2.3 | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-lingbot-dense-fsdp-withheld` | LingBot-Video-Dense | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-lingbot-moe-fsdp-withheld` | LingBot-Video-MoE | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-minimax-h3-fsdp-withheld` | MiniMax-H3 | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-minimax-h3-ref2va-fsdp-withheld` | MiniMax-H3-Ref2VA | fsdp_blockwise/bf16/w2 | - | - | refused |
| `rocm-qwen-fp8-replicated-te` | Qwen-Image | replicated/fp8/w2+te | 39 | 39 | pass |
| `rocm-qwen-edit-fp8-eager-model-offload` | Qwen-Image-Edit | eager/fp8/w1+te+offload=model | 22 | 27 | pass |
| `rocm-wan22-i2v-bf16-eager` | Wan2.2-I2V | eager/bf16/w1 | 66 | 60 | pass |
| `rocm-wan22-i2v-bf16-replicated4` | Wan2.2-I2V | replicated/bf16/w4 | 69 | 69 | pass |
| `rocm-wan22-i2v-bf16-ulysses4-control` | Wan2.2-I2V | eager/bf16/w4 | 67 | 110 | pass |
| `rocm-wan22-i2v-fp4-eager` | Wan2.2-I2V | eager/fp4/w1 | - | - | wants `gfx942` |
| `rocm-wan22-i2v-fp4-fsdp4` | Wan2.2-I2V | fsdp_blockwise/fp4/w4 | - | - | wants `gfx942` |
| `rocm-wan22-i2v-fp8-eager` | Wan2.2-I2V | eager/fp8/w1 | 42 | 50 | pass |
| `rocm-wan22-i2v-fp8-fsdp4` | Wan2.2-I2V | fsdp_blockwise/fp8/w4 | 16 | 69 | pass |
| `rocm-wan22-i2v-fp8-replicated2` | Wan2.2-I2V | replicated/fp8/w2 | 43 | 62 | pass |
| `rocm-wan22-i2v-fp8-replicated4` | Wan2.2-I2V | replicated/fp8/w4 | 45 | 71 | pass |
| `rocm-wan22-i2v-fp8-te-eager` | Wan2.2-I2V | eager/fp8/w1+te | 38 | 23 | pass |
| `rocm-wan22-i2v-fp8-ulysses4-control` | Wan2.2-I2V | eager/fp8/w4 | 43 | 92 | pass |
| `rocm-wan22-t2v-hybrid-fsdp4` | Wan2.2-T2V | fsdp_blockwise/hybrid FP8+FP4/w4 | - | - | refused |
| `rocm-zimage-int8-rejected` | Z-Image-Turbo | eager/int8/w1 | - | - | refused |
| `rocm-zimage-turbo-bf16-fsdp-group-offload-rejected` | Z-Image-Turbo | fsdp_blockwise/bf16/w2+offload=group | - | - | refused |
| `rocm-zimage-turbo-bf16-fsdp-model-offload` | Z-Image-Turbo | fsdp_blockwise/bf16/w2+offload=model | 15 | 22 | pass |
| `rocm-zimage-turbo-bf16-fsdp-sequential-offload-rejected` | Z-Image-Turbo | fsdp_blockwise/bf16/w2+offload=sequential | - | - | refused |
| `rocm-zimage-turbo-bf16-replicated-model-offload` | Z-Image-Turbo | replicated/bf16/w2+offload=model | 23 | 26 | pass |

Every refusal fires before the load allocates and is matched on its reason rather than on merely failing,
and a test compares each pattern against the message the runner's declaration produces. Seven come from
models that withhold the memory-efficient load or the sharding flag outright — MiniMax-H3 refuses without
reading any of its 330 GB — and the other eight are limits these runs found, below.

## Other GPUs

Nothing has run anywhere else, so nothing here has a second data point. Twenty-two cases are pinned to
hardware this node is not, and only three of their claims are genuinely architectural: FP8 through
AITER, which is selected on `gfx1200` and `gfx1201` alone; NVFP4, gated on CUDA capability 10.0; and
INT8 that works rather than refuses, which TorchAO supports on CUDA only.

| Accelerator | Cases | What they would add |
| --- | --- | --- |
| `gfx1200`/`gfx1201` (RDNA4) | 8 | the AITER FP8 quantizer, which is also the only backend whose text encoder can stream under a replicated load |
| `sm100`+ (Blackwell) | 6 | NVFP4 across all three placements, INT8 with group offload, and a hybrid-schedule refusal |
| `sm90` (Hopper) | 4 | FP8 and INT8 on CUDA, INT8 with sequential offload, and LTX-2, which nothing here covers |
| `sm89` (Ada) | 3 | FP8 with a text encoder and INT8 replicated on an older CUDA arch, plus an NVFP4 refusal |
| `gfx942` | 1 | the FP4 refusal that holds where AITER ships no FP4 kernels |

Three of the twenty-two cannot run anywhere as written: SD3.5, CausalWan and Wan2.2-Distilled-I2V have
no sampling entry and no node has their weights, so an operating point cannot be chosen without
inventing one. [The handoff](gpu_validation_handoff.md#what-needs-other-hardware) carries the full
accounting of which are worth a booking.

## What these runs found

Nineteen things, none of them findable by a case that had already run: each needed a model, a placement
or an environment nothing had combined before. Some are defects that are fixed, some are limits that are
now refused before they can waste a load.

| Where | What it was |
| --- | --- |
| FP8 quantization | A dynamic per-tensor scale is `max_abs / 448`, which is zero for a tensor of zeros, so quantizing returned NaN. Padding-only sequence chunks at 8 ranks made every FP8 Qwen-Image render pure black. Fixed with torchao's `activation_value_lb`, which only binds below it, so ordinary activations measure bit-identical. |
| Sharding with compile | Sharding turns one compiled transformer into one CUDA graph per block, and recording a later block reads a buffer the graph system still holds live. This was the whole FLUX column. Fixed with the step-boundary pre-hook the error message names. |
| HunyuanVideo declaration | Its 14 GB Llama encoder was outside `fsdp_strategy`, so every rank loaded it whole and host memory did not improve. Declaring it took blockwise from 155 to 102 GB and load VRAM from 29 to 15 GB. |
| Ideogram-4 checkpoint | Its FP8 checkpoint fuses the three attention projections and stores each weight beside its scale, so one live tensor needs two stored ones. Manifests carry derived tensors now, and the model came off the withheld list. |
| Replicated denoiser routing | The path chose per-block fill against whole broadcast by name, so Ideogram-4's `unconditional_transformer` took the text-encoder branch and was broadcast from a rank-0 copy still on meta. Every render was black at `std=0.00`, and nothing raised. Decided from the declaration now. |
| Text-encoder FP8 | The streaming probe looked for a method pair Transformers 5 had renamed, so every eager text encoder took the post-load fallback and materialized in bf16 first. Accepting either surface holds FLUX.2-dev at 58 GB where the fallback needs 79, for a load twice as long. |
| Offload placement | Model and sequential offload passed no device, so Diffusers defaulted every rank to `cuda:0` and multi-rank offload died in NCCL with `Duplicate GPU detected`. Both name the local rank's device now. |
| Offload with sharding | Group offload asks each parameter whether it is pinned and torch has no sharding strategy for `aten.is_pinned`; sequential offload rebuilds each parameter, which needs a spec a DTensor's replacement does not carry. Both failed mid-denoise after a full sharded load; both are refused up front. Whole-model offload works, because it moves components rather than reaching into them. |
| Offload with AITER FP4 | Group offload aborted the rank with SIGABRT, AITER having bound a device from a host parameter, and with `--group_offload_low_cpu_mem` raised inside the hook, torch having no `pin_memory` for `Float4_e2m1fn_x2`. Both refused, scoped to AITER. |
| Mixed FP8/FP4 with sharding | Refused: this torch cannot shard a non-floating-point parameter under FSDP2 while the FP4 half targets the wrapped blocks. Exactly the gate an unrun RDNA4 case had predicted in its notes. |
| Quality gate | Nothing looked at the image, because reference scoring only gates models that reproduce their sample. A spread measurement gates every case now, and a uniform frame is `failed_blank_output` whatever it exited with. |
| Quality floor | A single floor calibrated at 512×512 and seed 1234 failed two FP8 cases whose images are the same cat. Floors are recorded next to the sampling they were measured at, and split: 0.90 for a case that only moves weights, 0.60 for one that quantizes them. |
| Offline Krea-2 | Diffusers skips its "does the repo have these shards" call only under `local_files_only`, which `HF_HUB_OFFLINE` does not reach, and the repo refuses downloads to this token. A checkpoint request takes its default from the variable now. |
| Krea-2 wrap path | The FSDP strategy named `model.layers`, where a text-only encoder keeps its decoder layers; Krea-2's `Qwen3VLModel` keeps them under `language_model`. Twelve passing eager and replicated cases had said nothing about it. |
| Offline single-file discovery | klein-4B ships one unsharded transformer file, and offline a name the repo lacks and a name never cached raise the same error, so discovery read "no index" as failure. The component's cached `config.json` tells them apart. |
| HunyuanVideo tokenizer | Transformers 5 parses every model's root `config.json` offline for an unrelated Mistral fix, and HunyuanVideo's has a trailing comma. The reload reads the tokenizer's own directory now. |
| Wan2.2-TI2V | The one runner serving both tasks requires `--task`, which its sampling entry did not pass. |
| Rendezvous ports | Back-to-back runs collided on torchrun's default 29500, and because the affected placements moved between runs it read as a nondeterministic sharded-load bug. Each case asks the OS for a free port. |
| Ulysses degree | The generator emitted 8-rank cases for a model with 18 attention heads, which cannot be split 8 ways. Models declare their head count and the generator picks the largest admissible rank count. |

## How to read the numbers

- **Load-time VRAM** is the peak on the busiest single device while the load is in flight, not a sum
  over devices, so it is comparable across rank counts. It is the figure these paths exist to move.
- **Host anon** is the container's anonymous pages, which is what the OOM killer watches, and is not
  summed over ranks. Page cache is excluded: the kernel reclaims it under pressure, so it is not a cost.
- **Quality is a property of the model here, not of the load.** SSIM is measured against the same model's
  single-rank eager render with `torch.compile` disabled on both sides, in a separate scoring pass whose
  timings are kept out of the columns above. On every model scored, eager, the control and blockwise
  return the same score to every digit reported, so no score in that pass is attributable to how the
  weights were loaded: what is left is the parallelism, and whether the model redraws its sample under any
  numeric change. Only Z-Image-Turbo's score is a verdict for that reason; see `IDENTITY_STABLE_MODELS` in
  `tools/gpu_validation.py`. Quantizing does move the score, by construction and by a model-dependent
  amount, which is why the fp8 numbers are not reported beside the memory a quantized load saves —
  `tools/validation_report.py` prints the per-case scores.
- **Video is gated, not compared.** Twelve evenly spaced frames of each clip are read for having
  rendered anything, so a clip that collapses after one good frame cannot average past the floor. The
  closest any real clip came to the 0.01 floor was 0.2149, twenty times above it.
- **Timings are one run each** on a shared node, and ROCm samples VRAM device-globally, so treat the
  memory figures as upper bounds and the times as indicative. The first sweep ran on an older Diffusers
  commit than the rest, so compare within a sweep rather than across.
- **Each model is sampled the way its own benchmark config or runner declaration samples it**, at seed
  42, which is why the load times are not comparable between models. A test checks each citation against
  what the runner declares, so an entry cannot go stale. Cosmos3 takes its prompt from the checkpoint's
  own `assets/example_t2v_prompt.json`, because the model wants structured prompts rather than prose.

## Reproducing it

One case per invocation with a shared run id, and a page-cache drop between cases so the load times mean
something:

```bash
python tools/gpu_validation.py --case <case-id> --execute --continue-on-error \
  --run-id my-sweep --results gpu-validation-results/results.jsonl
```

Scoring is a second pass with `--score-quality`, which disables `torch.compile` and reuses each model's
recorded compile-free single-rank render, so run those first. Then read both passes together, which is
where the tables above come from:

```bash
python tools/validation_report.py --results gpu-validation-results/results.jsonl
```

Records, logs, images and clips are node-local and not checked in. The run ids behind the tables, in the
order the work happened:

| Run ids | What they cover |
| --- | --- |
| `20260808T081101Z-feature-evidence`, `20260808T154216Z-feature-evidence-scoring` | the six image models with a standard transformer, performance then scoring |
| `20260809T085227Z-coverage-images`, `20260809T100205Z-coverage-krea-fixed`, `20260809T101323Z-coverage-krea-fixed2`, `20260809T102509Z-coverage-video`, `20260810T062730Z-coverage-video-rerun`, `20260810T071614Z-coverage-score` | the ten models that had never run |
| `20260810T085256Z-hunyuan-te` | HunyuanVideo again, with its text encoder declared |
| `withheld-rejections`, `withheld-rejections-sparse` | the models that refuse the path |
| `klein4b-coverage`, `klein4b-offline-fix`, `cosmos3-nano-coverage`, `cosmos3-super-coverage`, `cosmos3-super-finish` | the three models whose weights had to be fetched |
| `curated-backfill`, `repin-gfx950`, `repin-gfx950-fixed`, `offload-multirank`, `offload-sharded` | the curated cases and the offload work |
| `ideogram-meta`, `ideogram-w6`, `ideogram-replicated-fix` | Ideogram-4 coming onto the path |
| `te-streaming`, `te-streaming-eager`, `te-path-recorded`, `te-path-recorded-tf4` | the text-encoder streaming fix and its measurement |

Every case that failed was re-run after the change that fixed it, so no figure here predates the change
its own model needed. Cosmos3-Super's sweep was interrupted to free the node and its last three cases
finished under a second run id; the record the interruption produced was removed from the results file
rather than left to read as a failure, since it is evidence about the interruption and not the load.

Two facts about this node's cache the next operator will not guess: `krea/krea-2-turbo` held a complete
snapshot of one revision while `refs/main` named another, so nothing could resolve it offline and the
ref was pointed at the snapshot that is there; and of the fifty repos checked for that inconsistency the
only other one is `tencent/HunyuanVideo`, which needs no repair because its runner pins the revision on
disk.
