# xDiT Unified Runner

The xDiT Unified Runner provides a single entry point for running all supported diffusion models with proper benchmarking and profiling support.

## Overview

The unified runner provides:

- **Single CLI interface** for all supported models
- **Programmatic API** for integration into custom code
- **Built-in benchmarking** with timing measurements
- **Profiling support** via PyTorch profiler
- **Automatic validation** of model capabilities and arguments
- **Parallelization** across all supported models

## Quick Start

### Basic Usage

Run any supported model using `xdit`:

```bash
xdit --model FLUX.1-dev \
    --prompt "A cat running in a garden" \
    --ulysses_degree 8
```

This will generate an image with Flux.1-dev and uses the model-specific values for any parameters that were not provided.


## Architecture

The unified runner consists of three main components:

### 1. Runner (`xfuser/runner.py`)

The main entry point that users interact with. It handles:

- Argument parsing and validation
- Model selection from the registry
- Execution flow (initialization → run/profile → save → cleanup)

```python
# api_example.py
# Usage: torchrun --nproc_per_node=4 api_example.py

from xfuser.runner import xFuserModelRunner

# Programmatic usage
config = {
    "model": "FLUX.1-dev",
    "prompt": "A cat running",
    "ulysses_degree": 4,
}
runner = xFuserModelRunner(config)
input_args = runner.preprocess_args(config)
runner.initialize(input_args)
output, timings = runner.run(input_args)
runner.save(output=output, timings=timings)
runner.cleanup()
```

### 2. Base Model (`xfuser/model_executor/models/runner_models/base_model.py`)

Contains all shared logic for model operations, e.g:

- Model loading and initialization
- Benchmarking and timing
- Profiling with PyTorch profiler
- Output saving
- Torch compilation
- Warmup calls
- All other generic features

### 3. Model Implementations

Individual model classes that inherit from `xFuserModel`:

- Define model-specific loading logic
- Implement the inference pipeline
- Specify default values and capabilities
- Override base methods when needed for custom features

## Supported Models

| Model | Valid Model Name(s) |
|-------|-----------------|
| FLUX.1-dev | `FLUX.1-dev`, `black-forest-labs/FLUX.1-dev` |
| FLUX.1-Kontext | `FLUX.1-Kontext-dev`, `black-forest-labs/FLUX.1-Kontext-dev` |
| FLUX.2 | `FLUX.2-dev`, `black-forest-labs/FLUX.2-dev` |
| FLUX.2-klein | `FLUX.2-klein-9B`, `black-forest-labs/FLUX.2-klein-9B`, `FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-4B` |
| HunyuanVideo | `HunyuanVideo`, `tencent/HunyuanVideo` |
| HunyuanVideo-1.5 | `HunyuanVideo-1.5`, `tencent/HunyuanVideo-1.5` |
| Wan 2.1/2.2 I2V | `Wan2.1-I2V`, `Wan2.2-I2V`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| Wan 2.2 Distilled I2V (LightX2V 4-step) | `Wan2.2-Distilled-I2V` |
| Wan 2.1/2.2 T2V | `Wan2.1-T2V`, `Wan2.2-T2V`, `Wan-AI/Wan2.1-T2V-14B-720P-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Wan 2.1 VACE | `Wan2.1-VACE-14B`, `Wan2.1-VACE-1.3B`, `Wan-AI/Wan2.1-VACE-14B`, `Wan-AI/Wan2.1-VACE-1.3B` |
| Stable Diffusion 3 | `SD3.5`, `stabilityai/stable-diffusion-3.5-large` |
| Z-Image-Turbo | `Z-Image-Turbo`, `Tongyi-MAI/Z-Image-Turbo` |
| LTX-2 | `LTX-2`, `Lightricks/LTX-2` |
| LTX-2.3 | `LTX-2.3`, `dg845/LTX-2.3-Diffusers` |
| Cosmos3-Super | `Cosmos3-Super`, `nvidia/Cosmos3-Super` |
| Cosmos3-Nano | `Cosmos3-Nano`, `nvidia/Cosmos3-Nano` |
| CausalWan | `CausalWan` |
| Qwen-Image | `Qwen-Image`, `Qwen/Qwen-Image`, `Qwen-Image-2512`, `Qwen/Qwen-Image-2512` |
| Qwen-Image-Edit | `Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit`, `Qwen-Image-Edit-2509`, `Qwen/Qwen-Image-Edit-2509`, `Qwen-Image-Edit-2511`, `Qwen/Qwen-Image-Edit-2511` |
| Krea2-Raw | `krea/krea-2-raw`, `krea/Krea-2-Raw`, `Krea-2-Raw` |
| Krea2-Turbo | `krea/krea-2-turbo`, `krea/Krea-2-Turbo`, `Krea-2-Turbo` |
| Ideogram 4 | `Ideogram-4`, `ideogram-ai/ideogram-v4`, `ideogram-ai/ideogram-4-nf4`, `ideogram-ai/ideogram-4-fp8` |
| MiniMax-H3 | `MiniMaxAI/MiniMax-H3`, `MiniMax-H3`, `MiniMax-H3-Ref2VA` |

## CLI Arguments

#### Note: not all models support all of the features.

### Model Selection

| Argument | Description |
|----------|-------------|
| `--model` | Model name or HuggingFace path (required) |
| `--task` | Task type for multi-task models |

### Parallelization

| Argument | Description | Default |
|----------|-------------|---------|
| `--ulysses_degree` | Ulysses sequence parallel degree | 1 |
| `--ring_degree` | Ring sequence parallel degree | 1 |
| `--pipefusion_parallel_degree` | PipeFusion pipeline stages | 1 |
| `--tensor_parallel_degree` | Tensor parallel degree | 1 |
| `--data_parallel_degree` | Data parallel degree | 1 |
| `--use_cfg_parallel` | Enable CFG parallel | False |
| `--use_parallel_vae` | Enable parallel VAE | False |
| `--fully_shard_degree` | FSDP sharding degree; set to number of GPUs to shard across. 1 disables sharding. | 1 |
| `--no_reshard_after_forward` | Keep parameters gathered after each block forward; trades memory for latency | False |
| `--memory_efficient_sharding` | Reduce peak VRAM during load: shard transformer blocks one at a time on GPU during init, so the full unsharded component never materializes on device. Slightly slower to load. Use if the model OOMs on GPU during init. Requires `--fully_shard_degree > 1` | False |
| `--memory_efficient_replicated_load` | Reduce host RAM used by replicated model weights when a model that fits one GPU is replicated across ranks (pure sequence/CFG/data parallelism): rank 0 loads the real weights and peers receive them over a GPU→GPU broadcast, so the replicated-weight contribution to host peak is approximately 1× the model instead of N×. Tokenizers, framework state, checkpoint page cache, and other process-local allocations remain per rank. | False |

### Input Parameters

| Argument | Description | Default |
|----------|-------------|---------------|
| `--prompt` | Text prompt(s) for generation | - |
| `--negative_prompt` | Negative prompt(s) | - |
| `--height` | Output height | Model-specific |
| `--width` | Output width | Model-specific |
| `--num_frames` | Number of frames for video models | Model-specific |
| `--num_inference_steps` | Denoising steps | Model-specific |
| `--guidance_scale` | Classifier-free guidance scale | Model-specific |
| `--max_sequence_length` | Maximum sequence length | Model-specific |
| `--seed` | Random seed for reproducibility | 42 |
| `--input_images` | Input image paths for image-to-image/video | [] |

### Optimization Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_torch_compile` | Enable torch.compile acceleration | False |
| `--use_fp8_gemms` | Enable FP8 GEMM quantization for the transformer | False |
| `--use_fp8_text_encoder` | Extend FP8 quantization to the text encoder as well (requires `--use_fp8_gemms`). Frees several GB for models with large bf16 text encoders. | False |
| `--use_fp4_gemms` | Enable FP4 GEMM quantization for declared transformer targets (ROCm MXFP4 or CUDA NVFP4) | False |
| `--use_hybrid_gemm_schedule` | Enable the explicit FP8/FP4 hybrid schedule. Requires `--use_fp4_gemms`; also required when `--use_fp8_gemms` and `--use_fp4_gemms` are both set. | False |
| `--use_int8_gemms` | Enable torchao W8A8 INT8 quantization for declared transformer targets. Cannot be combined with FP8, FP4, or hybrid FP8/FP4 mode. | False |
| `--enable_tiling` | Enable VAE tiling | False |
| `--enable_slicing` | Enable VAE slicing | False |
| `--enable_model_cpu_offload` | Enable model CPU offload | False |
| `--enable_sequential_cpu_offload` | Enable sequential CPU offload | False |
| `--enable_group_cpu_offload` | Group CPU offload: parameters live on the host and are streamed to the GPU a group at a time, overlapping transfer with compute | False |
| `--group_offload_low_cpu_mem` | With `--enable_group_cpu_offload`, pin each tensor as it is offloaded rather than pre-pinning whole components. Keeps host RAM flat at the cost of some of the streaming speedup | False |
| `--attention_backend` | Attention backend selection | None |

### Loading and Quantization Contract

The quantization flags select model-declared linear-layer targets; they do not quantize every pipeline component. Unless noted below, the VAE and untargeted text encoders retain the pipeline dtype.

#### Backend and Format Semantics

This matrix records the runner's implemented dispatch and rejection paths. GPU
end-to-end validation is still pending for every listed hardware/model
combination; see [Validation Status](#validation-status). Model capability
checks and target declarations also apply.

| Hardware / available backend | FP8 (`--use_fp8_gemms`) | FP4 (`--use_fp4_gemms`) | INT8 (`--use_int8_gemms`) |
|------------------------------|--------------------------|--------------------------|-----------------------------|
| ROCm RDNA4 (`gfx1200`/`gfx1201`) + AITER | AITER block-scale W8A8 FP8 (block size 128); streaming load where the runner is wired for it, otherwise layer-by-layer post-load | AITER MXFP4; ordinary loads convert after load, replicated/FSDP meta loads convert blockwise; runtime hybrid FP8/FP4 is supported | Excluded: INT8 is rejected on ROCm |
| ROCm RDNA4 without AITER | torchao per-tensor dynamic-activation/FP8-weight; native Diffusers/Transformers per-weight streaming where API, exact target mapping, and placement permit, otherwise explicit post-load fallback | Excluded: ROCm FP4 requires AITER | Excluded: INT8 is rejected on ROCm |
| Other ROCm + AITER | torchao per-tensor dynamic-activation/FP8-weight; native Diffusers/Transformers per-weight streaming where API, exact target mapping, and placement permit, otherwise explicit post-load fallback; AITER FP8 block-scale is RDNA4-only | AITER MXFP4 | Excluded: INT8 is rejected on ROCm |
| Other ROCm without AITER | torchao per-tensor dynamic-activation/FP8-weight; native Diffusers/Transformers per-weight streaming where API, exact target mapping, and placement permit, otherwise explicit post-load fallback | Excluded: ROCm FP4 requires AITER | Excluded: INT8 is rejected on ROCm |
| CUDA capability 10.0+ (Blackwell) | torchao per-tensor dynamic-activation/FP8-weight; native Diffusers/Transformers per-weight streaming where API, exact target mapping, and placement permit, otherwise explicit post-load fallback | torchao NVFP4 with dynamic per-tensor activation scaling; native Diffusers streams NVFP4 leaves while explicit FP8 overrides remain full precision for post-load FP8 conversion; hybrid ownership is excluded | torchao dynamic-activation/dynamic-weight W8A8 INT8; native Diffusers per-weight streaming preserves target and minimum-size exclusions |
| CUDA capability 8.9 through 9.x | torchao per-tensor dynamic-activation/FP8-weight; native Diffusers/Transformers per-weight streaming where API, exact target mapping, and placement permit, otherwise explicit post-load fallback | Excluded: NVFP4 requires capability 10.0+ | torchao dynamic-activation/dynamic-weight W8A8 INT8; native Diffusers per-weight streaming where accepted |
| CUDA capability below 8.9 | Excluded: TorchAO FP8 is rejected during backend preflight before model allocation | Excluded: NVFP4 requires capability 10.0+ | torchao dynamic-activation/dynamic-weight W8A8 INT8; runtime kernel support remains hardware-dependent |

INT8 uses per-row symmetric scaling and skips linear layers smaller than 512 in
either dimension. Ordinary loading uses Diffusers `TorchAoConfig` per-weight
streaming when it accepts the exact TorchAO config. xDiT derives
`modules_to_not_convert` from a config-built meta structure, preserving the
declared targets and 512 minimum. The descriptor records a post-load fallback
when streaming is unavailable.

Replicated meta-load applies INT8 blockwise. Memory-efficient FSDP requires the
installed INT8 tensor subclass to expose composable-FSDP gather hooks; xDiT
rejects the mode before allocation when those hooks are absent. With sequence
parallelism, Z-Image leaves `context_refiner` unquantized because its local
sequence can be too short for `torch._int_mm`.

CUDA NVFP4 uses native Diffusers `TorchAoConfig` per-weight streaming. FP8
prefix/suffix overrides are excluded from the native config; the FP4 adapter
converts those residual leaves to FP8 after loading. CUDA NVFP4 lacks the ROCm
runtime high/low precision wrapper. Therefore, xDiT rejects
`--use_hybrid_gemm_schedule` before model allocation with guidance to disable
it or use ROCm+AITER.

ROCm MXFP4 has no native per-weight streaming path. AITER needs the full source
weight to create and shuffle `xFuserMXFP4Linear` state. Ordinary loads convert
post-load; replicated and FSDP meta-loads convert each block before placement.
Preflight checks `aiter.get_hip_quant`, `aiter.QuantType.per_1x32`,
`aiter.gemm_a4w4`, and `aiter.ops.shuffle.shuffle_weight`. ROCm plus those
symbols is the static capability contract; kernel execution still requires GPU
validation. The packed MXFP4 weight is a shardable, non-trainable `Parameter`.
Its scale is a persistent replicated buffer.

All quantized GEMM modes are inference-only. TorchAO serialization requires
compatible torchao, Diffusers, huggingface-hub, and tensor-subclass support.
MXFP4 packed state can round-trip in xDiT, but its AITER-specific layout is not
a portable training checkpoint. Backend descriptors log the requested format,
selected backend, storage, materialization, trainability, serialization, and
fallback reason. FP8+FP4 requires explicit hybrid mode, while INT8 is mutually
exclusive. FP8 precision prefix/suffix overrides remain FP8-owned inside FP4
targets.

#### Model Support Matrix

“Streaming” means native transformer quantization during ordinary loading:
AITER for supported FP8 paths, or torchao through supported Diffusers APIs for
FP8, NVFP4, and INT8. NVFP4 can stream around explicitly owned FP8 residual
leaves. Torchao records an explicit post-load fallback when its API, exact
target mapping, hybrid ownership, or placement prevents streaming.

On a single rank, a standard runner with declared meta construction and
block-owned targets promotes that post-load fallback to local blockwise
loading. The loader materializes one checkpoint block, quantizes it, and
releases the source block before reading the next. Multi-rank and offloaded
runs retain their existing placement-specific paths.

“Memory-efficient FSDP” means `--memory_efficient_sharding` can avoid
materializing the full transformer before FSDP wraps it. “Replicated meta-load”
means `--memory_efficient_replicated_load` can build the transformer on meta and
fill it from rank 0.

| Model family | Transformer loading | Text-encoder FP8 target | Memory-efficient FSDP | Replicated meta-load |
|--------------|---------------------|-------------------------|-----------------------|----------------------|
| FLUX.1-dev, FLUX.1-Kontext | Streaming; transformer targets declared | `text_encoder_2` | Transformer + targeted text encoder | Yes |
| FLUX.2, FLUX.2-klein (4B/9B) | Streaming; transformer targets declared | `text_encoder` | Transformer + targeted text encoder | Yes |
| Wan 2.1/2.2 I2V and T2V, Wan 2.2 TI2V | Streaming; both Wan 2.2 transformers are covered | `text_encoder` | Transformer(s) + targeted text encoder | Yes |
| Wan 2.2 Distilled I2V | Explicit `distilled_wan_remap` adapter exclusion; external LightX2V state dicts replace both transformers | Declared, post-load only | Rejected before allocation: strict remapping is not collective-safe | No |
| Wan 2.1 VACE | Streaming; main and VACE blocks covered | `text_encoder` | Not exposed by this runner | Yes |
| Qwen-Image and Qwen-Image-Edit variants | Streaming; transformer targets declared | `text_encoder` | Transformer + targeted text encoder | Yes |
| Z-Image and Z-Image-Turbo | Streaming; transformer, noise refiner, and context refiner covered | `text_encoder` | Transformer + targeted text encoder | Yes |
| Krea2-Raw and Krea2-Turbo | Streaming; transformer targets declared | Explicitly excluded: the Qwen3VL ROCm float32-Linear workaround has no exact quantization target/API contract | Transformer only; text encoder loads normally | Transformer only; text encoder loads per rank |
| Stable Diffusion 3.5 | Explicit `sd35_composition` adapter exclusion; direct/post-load only | `text_encoder_3`, post-load only | Rejected before allocation: the composition wrapper has no config-only transformer seam | No |
| HunyuanVideo | Streaming from pinned revision `refs/pr/18`; both transformer block lists declared | None | Transformer | Transformer |
| HunyuanVideo-1.5, distilled, and sparse/remapped variants | Explicit `hunyuan_video_15_variants` adapter exclusion; direct/remapped loading only | None | Rejected before allocation: separate wrapper/config and remapped sparse composition are not verified against the standard seam | No |
| LTX-2 | Eager/native streaming through the shared transformer seam; transformer targets declared | None | Rejected before allocation: stage-2 distilled LoRA currently precedes base checkpoint fill on a meta transformer | No |
| LTX-2.3 | Eager load through the shared transformer seam; no quantized GEMM capability declared | None | Rejected before allocation: stage-2 distilled LoRA currently precedes base checkpoint fill on a meta transformer | No |
| Cosmos3-Super and Cosmos3-Nano | Streaming; transformer targets declared | None | Transformer only | Transformer only |
| CausalWan | Explicit `causal_wan_custom` adapter exclusion; direct load with manual single-file fallback | None | Rejected before allocation: fallback discovery is not collective-safe | No |

The FLUX PipeFusion loading branches construct their complete pipelines directly. They therefore do not use transformer streaming, and replicated meta-load is excluded whenever PipeFusion is active. Named custom adapters are declarative exclusions: they preserve eager behavior, but a requested meta mode fails before model allocation and cannot enter the standard transformer collective path.

#### Per-model FP4 and INT8 Coverage

An entry of “No” means the runner capability check rejects the flag, even if the hardware matrix would otherwise allow the format.

| Model family | FP4 targets | INT8 targets |
|--------------|-------------|--------------|
| FLUX.2 | Transformer blocks and single-transformer blocks; model quality overrides remain FP8 | No |
| Wan 2.1/2.2 I2V and T2V, Wan 2.2 TI2V | Declared transformer blocks; on dual-transformer Wan 2.2, the FP4 list covers `transformer` while `transformer_2` remains FP8 | No |
| Wan 2.2 Distilled I2V | Same declared dual-transformer split as Wan 2.2, applied post-load | No |
| Krea2-Raw and Krea2-Turbo | Transformer blocks | No |
| Cosmos3-Super and Cosmos3-Nano | Transformer layers; Cosmos3-Super keeps its declared first/last layer ranges in FP8 | No |
| Z-Image and Z-Image-Turbo | No | Transformer layers, noise refiner, and context refiner; `context_refiner` is excluded under sequence parallelism |
| FLUX.1, FLUX.1-Kontext, FLUX.2-klein, Wan VACE, Qwen-Image variants, Stable Diffusion 3.5, HunyuanVideo variants, LTX variants, CausalWan | No | No |

#### Flag Combinations and Exclusions

- `--use_fp8_gemms` quantizes transformer targets only. Text-encoder FP8 used to be implicit on supported RDNA4 runs; it is now opt-in everywhere. Add `--use_fp8_text_encoder` explicitly when wanted.
- `--use_fp8_text_encoder` requires `--use_fp8_gemms`. On an FP8-capable runner that declares no text-encoder target, the text-encoder flag has no effect; a runner without FP8 capability rejects the FP8 request during capability validation. Quantizing a supported text encoder may reduce text-conditioning quality.
- RDNA4+AITER streaming FP8 for a text encoder requires `transformers>=5.0` with `transformers.core_model_loading`. On Transformers 4.x, xDiT logs the reason and uses AITER post-load conversion where placement permits it; memory-efficient FSDP rejects the fallback before allocation because its sharded meta layout cannot be changed safely. The general `transformers>=4.39.1` package floor remains valid.
- Native torchao text-encoder loading requires `torchao>=0.15.0`, Diffusers `PipelineQuantizationConfig`, and Transformers `TorchAoConfig` quantize-on-load APIs. Native torchao transformer loading separately requires Diffusers `TorchAoConfig` accepting the exact `AOBaseConfig`; this includes NVFP4 and INT8 when installed APIs support them. These APIs are feature-probed lazily and unavailable paths fall back explicitly where placement permits it.
- `--use_int8_gemms` cannot be combined with `--use_fp8_gemms` or `--use_fp4_gemms`. This exclusion includes explicit hybrid FP8/FP4 mode. INT8 is also rejected on ROCm.
- Setting `--use_fp8_gemms` and `--use_fp4_gemms` together requires `--use_hybrid_gemm_schedule`; the generic combination is rejected. The hybrid FP4 path owns its FP8 high-precision conversion, so the generic FP8 traversal does not run afterward. Model-selected quality overrides and FP8-only components remain FP8. Use the FP8 precision-override flags only with FP4.
- `--memory_efficient_sharding` requires `--fully_shard_degree > 1`. It is a sharded load: rank 0 reads one block at a time and broadcasts it within the FSDP group before each rank receives its shard.
- `--fully_shard_degree` is orthogonal to the parallel degree and does not contribute to it. A multi-rank run must still declare a parallel degree whose product (`data × cfg × sequence × tensor × pipefusion`) equals the DiT parallel size, so pair `--fully_shard_degree N` with, for example, `--ulysses_degree N`. Setting only `--fully_shard_degree` fails config validation before the model is built.
- `--memory_efficient_replicated_load` is opt-in, requires multiple ranks, and applies only when weights are replicated. It is ignored with FSDP, PipeFusion, or tensor parallelism, and for runners marked “No” above. Pure Ulysses, ring, CFG, and data parallelism remain eligible.
- The two memory-efficient load flags represent different layouts and are not used together: FSDP splits weights, while replicated meta-load gives every rank the same weights.
- CPU/model offload can be combined with AITER FP8; converted leaves are evicted as they are processed. Other quantization backends first require their block or component on the GPU.

#### Practical Examples

RDNA4 transformer-only streaming FP8:

```bash
xdit --model FLUX.2-dev \
    --prompt "A lighthouse in a winter storm" \
    --use_fp8_gemms
```

RDNA4 transformer and text-encoder streaming FP8 (Transformers 5 required):

```bash
xdit --model FLUX.2-dev \
    --prompt "A lighthouse in a winter storm" \
    --use_fp8_gemms \
    --use_fp8_text_encoder
```

Memory-efficient FP8 FSDP load:

```bash
torchrun --nproc_per_node=4 -m xfuser.runner \
    --model Wan2.1-T2V \
    --prompt "Clouds moving over a mountain lake" \
    --ulysses_degree 4 \
    --fully_shard_degree 4 \
    --memory_efficient_sharding \
    --use_fp8_gemms
```

Replicated load with sequence parallelism:

```bash
torchrun --nproc_per_node=4 -m xfuser.runner \
    --model Qwen-Image \
    --prompt "An isometric botanical library" \
    --ulysses_degree 4 \
    --memory_efficient_replicated_load \
    --use_fp8_gemms
```

CUDA Blackwell NVFP4:

```bash
xdit --model FLUX.2-dev \
    --prompt "A studio photograph of a glass sculpture" \
    --use_fp4_gemms
```

These examples show the loading contract, not universal performance recommendations. Quantized output quality, peak memory, kernel availability, and the best override patterns depend on the checkpoint, GPU, torch/torchao/AITER versions, and parallel layout.

#### Validation Status

The reproducible external execution matrix, recorder, result schema, and
operator workflow are in the
[GPU Validation Handoff](gpu_validation_handoff.md). Its checked-in status is
**NOT RUN**; dry-runs and repository tests are not GPU end-to-end evidence.

Part of the matrix has now been executed. Thirty-six cases on 8× MI355X
(`gfx950`) cover the memory-efficient load paths in bf16 and FP8 across six image
models, reported in
[Memory-efficient load results](meta_load_results.md); the rows below are marked
against that. Everything outside those paths, models and quantizations remains
GPU-unvalidated here.

| Contract area | Implementation status | Static / unit-test evidence in the repository | GPU end-to-end status |
|---------------|-----------------------|----------------------------------------------|-----------------------|
| FP8 target selection and explicit `--use_fp8_text_encoder` opt-in | Implemented | Unit tests directly exercise target inclusion/exclusion and component-prefix routing. CLI validation and each runner's exact target declarations are verified by static inspection; the registry test only guards against text-encoder targets leaking into the always-on transformer list. | Validated with the text encoder quantized alongside the transformer on six image models at 8 ranks on gfx950, on both the sharded and replicated placements. Unvalidated on the other documented GPU/model combinations |
| AITER diffusers/Transformers streaming adapters and FP8 layer layout | Implemented | Unit tests directly cover quantizer registration/packaging and sentinel FP8 parameter/buffer layouts without invoking kernels. Streaming adapter execution is verified only by static inspection. | Streaming checkpoint load and AITER kernels are GPU-unvalidated here |
| Replicated meta-load policy | Implemented | Unit tests directly exercise the pure opt-in decision and exclusions for single rank, weight-splitting parallelism, and unwired runners. Collective fill behavior is verified only by static inspection. | Validated at 8 ranks with FP8 on six image models on gfx950: broadcast fill and complete inference, scored against an unquantized render. bf16 replicated, offload combinations, and other quantizations remain unvalidated |
| Memory-efficient FSDP policy and meta construction | Implemented | Unit tests directly exercise the FSDP gate and bf16 meta-transformer construction. Per-block fill and quantize routing are verified only by static inspection. | Validated at 8 ranks on six image models on gfx950, bf16 and FP8, against both an eager load at the same rank count and a shard-after-materialize control: same image to four decimals, at 1.1-1.9x the load speed and 1.7-5.2x less load-time device memory. One 4-rank spot check. Offload combinations, video models and FP4/INT8 remain unvalidated |
| FP4/INT8 adapters, hardware gates, targets, and placement | Implemented | Dependency-light tests cover injected hardware routing, exact native-config acceptance, target/minimum-size exclusions, hybrid fallback, descriptors, FSDP preflight, and MXFP4 parameter/buffer layout. Guarded integration tests require installed Diffusers/torchao and skip unsupported accelerators. | MXFP4, NVFP4, INT8, mixed FP4/FP8, and model-specific quality combinations are GPU-unvalidated here |
| Registry/model construction declarations | Implemented | Dependency-light AST tests require every registered runner to declare its load contract, extract actual `ModelSettings.fsdp_strategy` and instance strategy assignments for meta declarations, and keep named custom adapters out of standard collective modes. Guarded model tests check the Hunyuan/LTX wrapper config APIs when Diffusers is installed. | HunyuanVideo meta loading and LTX-2/2.3 eager/native loading remain GPU-unvalidated here |

Where a row claims no GPU result, the label describes code and repository test coverage only. Where a row names one, it points at the recorded sweep above, and its scope is exactly what that row states: another model, rank count, quantization or accelerator is not covered by it.

### Model-specific Arguments

| Argument | Description | Required for |
|----------|-------------|--------------|
| `--distilled_transformer_path` | Path to the **high-noise** distilled transformer safetensors | `Wan2.2-Distilled-I2V` |
| `--distilled_transformer_2_path` | Path to the **low-noise** distilled transformer safetensors | `Wan2.2-Distilled-I2V` |

### Benchmarking

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_iterations` | Number of benchmark iterations | 1 |
| `--warmup_calls` | Warmup iterations before timing | 0 |
| `--batch_size` | Batch size for dataset inference | None |
| `--dataset_path` | Path to prompt dataset csv | None |
| `--output_directory` | Output save directory | `.` |

### Profiling

| Argument | Description | Default |
|----------|-------------|---------|
| `--profile` | Enable PyTorch profiler | False |
| `--profile_wait` | Profiler wait steps | 2 |
| `--profile_warmup` | Profiler warmup steps | 2 |
| `--profile_active` | Profiler active steps | 1 |

## Examples

### Multi-GPU Image Generation

```bash
xdit --model FLUX.1-dev \
    --prompt "A majestic mountain landscape at sunset" \
    --height 1024 \
    --width 1024 \
    --ulysses_degree 4 \
    --num_inference_steps 50
```

### Video Generation

```bash
xdit --model HunyuanVideo \
    --prompt "A cat playing with a ball" \
    --height 720 \
    --width 1280 \
    --num_frames 49 \
    --ulysses_degree 8
```

### Distilled Video Generation

```bash
xdit --model Wan2.2-Distilled-I2V \
    --distilled_transformer_path   /path/to/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors \
    --distilled_transformer_2_path /path/to/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors \
    --input_images /path/to/image.jpg \
    --prompt "A cat walking in a garden" \
    --ulysses_degree 8
```

### Benchmarking with Multiple Iterations

```bash
xdit --model FLUX.1-dev \
    --prompt "Benchmark test image" \
    --ulysses_degree 8 \
    --num_iterations 5 \
    --output_directory ./benchmark_results
```

### Profiling

```bash
xdit --model FLUX.1-dev \
    --prompt "Profile test" \
    --ulysses_degree 8 \
    --profile \
    --output_directory ./profile_results
```

### With Torch Compile

```bash
xdit --model FLUX.1-dev \
    --prompt "Compiled inference test" \
    --ulysses_degree 4 \
    --use_torch_compile
```

### Dataset Inference

```bash
xdit --model FLUX.1-dev \
    --dataset_path ./prompts.csv \  # CSV file with at least column "prompt"
    --batch_size 4 \
    --ulysses_degree 8 \
    --output_directory ./dataset_outputs
```

## Programmatic Usage

The runner can be imported and used programmatically:

```python
from xfuser.runner import xFuserModelRunner

# Configuration dictionary
config = {
    "model": "FLUX.1-dev",
    "prompt": "A beautiful garden with flowers",
    "height": 1024,
    "width": 1024,
    "ulysses_degree": 4,
    "num_inference_steps": 50,
    "seed": 42,
    "output_directory": "./outputs",
}

# Create runner
runner = xFuserModelRunner(config)

# Preprocess arguments (applies model defaults)
input_args = runner.preprocess_args(config)

# Initialize model
runner.initialize(input_args)

# Run inference
output, timings = runner.run(input_args)

# Save outputs
runner.save(output=output, timings=timings)

# Cleanup
runner.cleanup()
```

### Profiling Programmatically

```python
runner = xFuserModelRunner(config)
input_args = runner.preprocess_args(config)
runner.initialize(input_args)

# Profile instead of run
output, timings, profile = runner.profile(input_args)
runner.save(profile=profile)

runner.cleanup()
```

## Output Files

The runner saves outputs to the specified `--output_directory`:

| File | Description |
|------|-------------|
| `{model}_u{ulysses}r{ring}_tc_{compile}_{height}x{width}_{index}.png` | Generated images |
| `{model}_u{ulysses}r{ring}_tc_{compile}_{height}x{width}_{index}.mp4` | Generated videos |
| `timings.json` | Timing measurements for each iteration |
| `profile_trace_rank_{rank}.json` | Chrome trace file for profiling |

Saved outputs depend on the input arguments used.
