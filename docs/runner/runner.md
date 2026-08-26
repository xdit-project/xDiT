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
| CausalWan | `CausalWan` |
| Cosmos3-Nano | `Cosmos3-Nano`, `nvidia/Cosmos3-Nano` |
| Cosmos3-Super | `Cosmos3-Super`, `nvidia/Cosmos3-Super` |
| FLUX.1-dev | `FLUX.1-dev`, `black-forest-labs/FLUX.1-dev` |
| FLUX.1-Kontext | `FLUX.1-Kontext-dev`, `black-forest-labs/FLUX.1-Kontext-dev` |
| FLUX.2 | `FLUX.2-dev`, `black-forest-labs/FLUX.2-dev` |
| FLUX.2-klein | `FLUX.2-klein-9B`, `black-forest-labs/FLUX.2-klein-9B`, `FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-4B` |
| HunyuanVideo | `HunyuanVideo`, `tencent/HunyuanVideo` |
| HunyuanVideo-1.5 | `Hunyuanvideo-1.5`, `tencent/HunyuanVideo-1.5`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` |
| HunyuanVideo-1.5 Distilled | `Hunyuanvideo-1.5-Distilled`, `tencent/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled` |
| HunyuanVideo-1.5 Sparse | `Hunyuanvideo-1.5-Sparse`, `tencent/HunyuanVideo-1.5-Sparse`, `tencent/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled_sparse` |
| Ideogram 4 | `Ideogram-4`, `ideogram-ai/ideogram-v4`, `ideogram-ai/ideogram-4-nf4`, `ideogram-ai/ideogram-4-fp8` |
| Ideogram 4 Diffusers | `ideogram-ai/ideogram-4-nf4-diffusers`, `CalamitousFelicitousness/Ideogram-4-bf16-Diffusers` |
| Krea2-Raw | `krea/krea-2-raw`, `krea/Krea-2-Raw`, `Krea-2-Raw` |
| Krea2-Turbo | `krea/krea-2-turbo`, `krea/Krea-2-Turbo`, `Krea-2-Turbo` |
| LingBot-Video-Dense | `LingBot-Video-Dense`, `robbyant/lingbot-video-dense-1.3b` |
| LingBot-Video-MoE | `LingBot-Video-MoE`, `robbyant/lingbot-video-moe-30b-a3b` |
| LTX-2 | `LTX-2`, `Lightricks/LTX-2` |
| LTX-2.3 | `LTX-2.3`, `dg845/LTX-2.3-Diffusers` |
| LTX-2.5 Distilled | `LTX-2.5`, `LTX-2.5-distilled`, `Lightricks/LTX-2.5-Diffusers` |
| LTX-2.5 Full | `LTX-2.5-full` |
| MiniMax-H3 | `MiniMaxAI/MiniMax-H3`, `MiniMax-H3`, `MiniMax-H3-Ref2VA` |
| Qwen-Image | `Qwen-Image`, `Qwen/Qwen-Image`, `Qwen-Image-2512`, `Qwen/Qwen-Image-2512` |
| Qwen-Image-Edit | `Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit`, `Qwen-Image-Edit-2509`, `Qwen/Qwen-Image-Edit-2509`, `Qwen-Image-Edit-2511`, `Qwen/Qwen-Image-Edit-2511` |
| Stable Diffusion 3.5 | `SD3.5`, `stable-diffusion-3.5-large`, `stabilityai/stable-diffusion-3.5-large` |
| Wan 2.1 VACE | `Wan2.1-VACE-14B`, `Wan2.1-VACE-1.3B`, `Wan-AI/Wan2.1-VACE-14B-diffusers`, `Wan-AI/Wan2.1-VACE-1.3B-diffusers` |
| Wan 2.1/2.2 I2V | `Wan2.1-I2V`, `Wan2.2-I2V`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| Wan 2.1/2.2 T2V | `Wan2.1-T2V`, `Wan2.2-T2V`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Wan 2.2 Distilled I2V (LightX2V 4-step) | `Wan2.2-Distilled-I2V` |
| Wan 2.2 TI2V | `Wan2.2-TI2V`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| Z-Image | `Z-Image`, `Tongyi-MAI/Z-Image` |
| Z-Image-Turbo | `Z-Image-Turbo`, `Tongyi-MAI/Z-Image-Turbo` |

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

#### Backend Selection by Hardware

Two defaults keep the table below short. “torchao FP8” means per-tensor
dynamic-activation with FP8 weights, streamed through the native
Diffusers/Transformers loaders where the API, the exact target mapping and the
placement all permit it, and converted after load where they do not. INT8 is
rejected on ROCm throughout. Model capability checks and target declarations
apply on top of whatever a row allows.

| Hardware / available backend | FP8 (`--use_fp8_gemms`) | FP4 (`--use_fp4_gemms`) | INT8 (`--use_int8_gemms`) |
|------------------------------|--------------------------|--------------------------|-----------------------------|
| ROCm RDNA4 (`gfx1200`/`gfx1201`) + AITER | AITER block-scale W8A8 FP8, block size 128; streams where the runner is wired for it, otherwise converts layer by layer after load | Excluded: AITER ships no FP4 kernels for RDNA4, refused in preflight | Excluded |
| ROCm RDNA4 without AITER | torchao FP8 | Excluded: ROCm FP4 requires AITER, which has no RDNA4 kernels either way | Excluded |
| Other ROCm + AITER | torchao FP8; AITER block-scale FP8 is RDNA4-only | AITER MXFP4 on `gfx950` and `gfx1250`, the architectures AITER has FP4 kernels for; ordinary loads convert after load, replicated and FSDP meta loads convert blockwise; runtime hybrid FP8/FP4 is supported. Any other architecture, `gfx942` included, is refused in preflight | Excluded |
| Other ROCm without AITER | torchao FP8 | Excluded: ROCm FP4 requires AITER | Excluded |
| CUDA capability 10.0+ (Blackwell) | torchao FP8 | torchao NVFP4 with dynamic per-tensor activation scaling; native Diffusers streams the NVFP4 leaves while explicit FP8 overrides stay full precision for post-load FP8 conversion; hybrid ownership is excluded | torchao W8A8 INT8; native streaming preserves the target and minimum-size exclusions |
| CUDA capability 8.9 through 9.x | torchao FP8 | Excluded: NVFP4 requires capability 10.0+ | torchao W8A8 INT8; native streaming where accepted |
| CUDA capability below 8.9 | Excluded: rejected in backend preflight before allocation | Excluded: NVFP4 requires capability 10.0+ | torchao W8A8 INT8; whether the kernels run stays hardware-dependent |

INT8 uses per-row symmetric scaling and skips linear layers smaller than 512 in
either dimension. To keep the declared targets and that 512 minimum while
streaming, xDiT derives `modules_to_not_convert` from a config-built meta
structure.

Replicated meta-load applies INT8 blockwise. Memory-efficient FSDP requires the
installed INT8 tensor subclass to expose composable-FSDP gather hooks; xDiT
rejects the mode before allocation when those hooks are absent. With sequence
parallelism, Z-Image leaves `context_refiner` unquantized because its local
sequence can be too short for `torch._int_mm`.

FP8 prefix/suffix overrides are excluded from the NVFP4 native config, and the
FP4 adapter converts those residual leaves to FP8 after loading. CUDA has no
equivalent of the ROCm runtime high/low precision wrapper, so xDiT rejects
`--use_hybrid_gemm_schedule` before allocation and points at ROCm+AITER.

ROCm MXFP4 converts rather than streams because AITER needs the full source
weight to create and shuffle `xFuserMXFP4Linear` state. Preflight checks
`aiter.get_hip_quant`, `aiter.QuantType.per_1x32`, `aiter.gemm_a4w4`, and
`aiter.ops.shuffle.shuffle_weight`, then the architecture: AITER exports those
symbols everywhere but builds FP4 kernels only for `gfx950` and `gfx1250`, and
calling them elsewhere aborts the process rather than raising. `AITER_FP4x2=0`
disables them too, and preflight honors that. The packed weight is a shardable,
non-trainable `Parameter` whose scale is a persistent replicated buffer.

All quantized GEMM modes are inference-only. TorchAO serialization requires
compatible torchao, Diffusers, huggingface-hub, and tensor-subclass support.
MXFP4 packed state can round-trip in xDiT, but its AITER-specific layout is not
a portable training checkpoint. Backend descriptors log the requested format,
selected backend, storage, materialization, trainability, serialization, and any
fallback reason.

#### Per-model Load Paths

“Streaming” means the transformer is quantized during ordinary loading rather
than after it, by AITER or by torchao through the Diffusers APIs.

On a single rank, a standard runner with declared meta construction and
block-owned targets promotes a post-load fallback to local blockwise loading:
the loader materializes one checkpoint block, quantizes it, and releases the
source block before reading the next. Multi-rank and offloaded runs keep their
placement-specific paths.

“Memory-efficient FSDP” means `--memory_efficient_sharding` can avoid
materializing the full transformer before FSDP wraps it. “Replicated meta-load”
means `--memory_efficient_replicated_load` can build the transformer on meta and
fill it from rank 0.

| Model family | Transformer loading | Text-encoder FP8 target | Memory-efficient FSDP | Replicated meta-load |
|--------------|---------------------|-------------------------|-----------------------|----------------------|
| FLUX.1-dev, FLUX.1-Kontext | Streaming; transformer targets declared | `text_encoder_2` | Transformer + targeted text encoder | Yes |
| FLUX.2, FLUX.2-klein (4B/9B) | Streaming; transformer targets declared | `text_encoder` | Transformer + targeted text encoder | Yes |
| Wan 2.1/2.2 I2V and T2V, Wan 2.2 TI2V | Streaming; both Wan 2.2 transformers are covered | `text_encoder` | Transformer(s) + targeted text encoder | Yes |
| Wan 2.2 Distilled I2V | Local blockwise loading is declared for external LightX2V state dicts; standard collectives are not | Declared, post-load only | Rejected before allocation: strict remapping is not collective-safe | No |
| Wan 2.1 VACE | Streaming; main and VACE blocks covered | `text_encoder` | Not exposed by this runner | Yes |
| Qwen-Image and Qwen-Image-Edit variants | Streaming; transformer targets declared | `text_encoder` | Transformer + targeted text encoder | Yes |
| Z-Image and Z-Image-Turbo | Streaming; transformer, noise refiner, and context refiner covered | `text_encoder` | Transformer + targeted text encoder | Yes |
| Krea2-Raw and Krea2-Turbo | Streaming; transformer targets declared | Not declared for shared loading: the Qwen3VL ROCm float32-Linear workaround has no exact quantization target/API contract | Transformer only; text encoder loads normally | Transformer only; text encoder loads per rank |
| Stable Diffusion 3.5 | No shared load route declared; direct/post-load only | `text_encoder_3`, post-load only | Rejected before allocation: the composition wrapper has no config-only transformer seam | No |
| HunyuanVideo | Streaming from pinned revision `refs/pr/18`; both transformer block lists declared | `text_encoder`, the Llama encoder; the CLIP encoder is left at pipeline dtype | Transformer + targeted text encoder | Yes |
| HunyuanVideo-1.5, distilled, and sparse/remapped variants | No shared load route declared; direct/remapped loading only | None | Rejected before allocation: separate wrapper/config and remapped sparse composition are not verified against the standard seam | No |
| LTX-2 | Eager/native streaming through the shared transformer seam; transformer targets declared | None | Rejected before allocation: stage-2 distilled LoRA currently precedes base checkpoint fill on a meta transformer | No |
| LTX-2.3 | Eager load through the shared transformer seam; no quantized GEMM capability declared | None | Rejected before allocation: stage-2 distilled LoRA currently precedes base checkpoint fill on a meta transformer | No |
| Cosmos3-Super and Cosmos3-Nano | Streaming; transformer targets declared | None | Transformer only | Transformer only |
| CausalWan | No shared load route declared; direct load with manual single-file fallback | None | Rejected before allocation: fallback discovery is not collective-safe | No |
| Ideogram 4 | Streaming; both the conditional and unconditional transformer declared | None | Both transformers; text encoder loads normally | Yes, for both transformers; the `trust_remote_code` text encoder is filled eagerly because the manifest cannot read its parameter names ahead of the load |
| MiniMax-H3 and MiniMax-H3-Ref2VA | Modular `ModularPipeline` construction with fused QKV projections; direct load only | None | Rejected before allocation: fusion rewrites attention into `attn.to_qkv`, so live tensor names stop matching checkpoint keys | No |

The FLUX PipeFusion loading branches construct their complete pipelines directly, so they do not use transformer streaming, and replicated meta-load is excluded whenever PipeFusion is active. Each runner keeps class-level `load_support` beside its capabilities and settings. The declaration positively names eligible meta transformers and text encoders and records standard-collective and local-blockwise routes independently; a requested unsupported meta mode fails before model allocation.

#### Per-model FP4 and INT8 Targets

A listed target is only half the answer: the format still has to survive the hardware table above, so a declared FP4 target is refused in preflight on RDNA4 or on pre-Blackwell CUDA no matter what this table says. “No” is the stronger statement, because the runner's capability check rejects the flag on every device.

| Model family | FP4 targets | INT8 targets |
|--------------|-------------|--------------|
| FLUX.2 | `transformer.transformer_blocks` and `transformer.single_transformer_blocks` | No |
| Wan 2.1/2.2 I2V and T2V, Wan 2.2 TI2V | `transformer.blocks`; on dual-transformer Wan 2.2 that covers `transformer` while all of `transformer_2` stays FP8 | No |
| Wan 2.2 Distilled I2V | The same split, inherited from Wan 2.2 I2V and applied post-load | No |
| Krea2-Raw and Krea2-Turbo | `transformer.transformer_blocks` | No |
| Ideogram 4 | `transformer.layers` and `unconditional_transformer.layers` | No |
| MiniMax-H3 | `transformer.transformer_blocks`; the Ref2VA variant declares `transformer_ref.transformer_blocks` | No |
| Cosmos3-Super and Cosmos3-Nano | `transformer.layers` | No |
| Z-Image and Z-Image-Turbo | No | `transformer.layers`, `transformer.noise_refiner`, and `transformer.context_refiner`; `context_refiner` drops out under sequence parallelism |
| FLUX.1, FLUX.1-Kontext, FLUX.2-klein, Wan 2.1 VACE, Qwen-Image variants, Stable Diffusion 3.5, HunyuanVideo variants, LTX variants, CausalWan | No | No |

Several runners hold part of an FP4 target at FP8 where full FP4 costs too much
quality: Wan 2.1 I2V and T2V keep blocks 0-9 and 30-39, Wan 2.2 TI2V keeps
blocks 0, 1, 28, and 29 plus the `.net.0.proj` and `.net.2` leaves, Cosmos3-Super
keeps layers 0-9 and 54-63, and both MiniMax-H3 runners keep the `attn.to_out.0`,
`ff.net.2`, and `adaln_proj.linear` leaves. Wan 2.2 I2V and T2V declare no block
overrides at all. `--fp8_precision_override_prefix_patterns` and
`--fp8_precision_override_suffix_patterns` replace the declared prefix or suffix
list rather than adding to it.

#### Flag Combinations and Exclusions

- `--use_fp8_gemms` quantizes transformer targets only. Text-encoder FP8 is opt-in everywhere: add `--use_fp8_text_encoder` when you want it.
- `--use_fp8_text_encoder` requires `--use_fp8_gemms` and a runner that explicitly declares text-encoder FP8 capability and targets. Other runners reject the text-encoder flag during capability validation. Quantizing a supported text encoder may reduce text-conditioning quality.
- RDNA4+AITER streaming FP8 for a text encoder requires `transformers>=5.0` with `transformers.core_model_loading`. On Transformers 4.x, xDiT logs the reason and uses AITER post-load conversion where placement permits it; memory-efficient FSDP rejects the fallback before allocation because its sharded meta layout cannot be changed safely. The general `transformers>=4.39.1` package floor remains valid.
- Native torchao text-encoder loading requires `torchao>=0.15.0`, Diffusers `PipelineQuantizationConfig`, and Transformers `TorchAoConfig` quantize-on-load APIs. Native torchao transformer loading separately requires Diffusers `TorchAoConfig` accepting the exact `AOBaseConfig`; this includes NVFP4 and INT8 when installed APIs support them. These APIs are feature-probed lazily and unavailable paths fall back explicitly where placement permits it.
- `--use_int8_gemms` cannot be combined with `--use_fp8_gemms` or `--use_fp4_gemms`, and that exclusion includes explicit hybrid FP8/FP4 mode.
- Setting `--use_fp8_gemms` and `--use_fp4_gemms` together requires `--use_hybrid_gemm_schedule`; the generic combination is rejected. The hybrid FP4 path owns its FP8 high-precision conversion, so the generic FP8 traversal does not run afterward. Model-selected quality overrides and FP8-only components remain FP8. Use the FP8 precision-override flags only with FP4.
- `--memory_efficient_sharding` requires `--fully_shard_degree > 1`. It is a sharded load: rank 0 reads one block at a time and broadcasts it within the FSDP group before each rank receives its shard.
- `--fully_shard_degree` is orthogonal to the parallel degree and does not contribute to it. A multi-rank run must still declare a parallel degree whose product (`data × cfg × sequence × tensor × pipefusion`) equals the DiT parallel size, so pair `--fully_shard_degree N` with, for example, `--ulysses_degree N`. Setting only `--fully_shard_degree` fails config validation before the model is built.
- `--memory_efficient_replicated_load` is opt-in, requires multiple ranks, and applies only when weights are replicated. It is ignored with FSDP, PipeFusion, or tensor parallelism, and for runners marked “No” above. Pure Ulysses, ring, CFG, and data parallelism remain eligible.
- The two memory-efficient load flags represent different layouts and are not used together: FSDP splits weights, while replicated meta-load gives every rank the same weights.
- CPU/model offload can be combined with AITER FP8; converted leaves are evicted as they are processed. Other quantization backends first require their block or component on the GPU.
- `--enable_group_cpu_offload` is rejected with FP4 on the AITER backend, including the hybrid FP8/FP4 mode, because packed FP4 weights survive neither leg of the offload: with `--group_offload_low_cpu_mem` the hook pins each tensor and torch has no `pin_memory` for `Float4_e2m1fn_x2`, and without it AITER binds a device from the parameter it is handed, so a host parameter resolves to an invalid ordinal and aborts the process. Offload at FP8 or bf16, or run FP4 without offload.
- Offload works across ranks: each rank onloads to its own local device, so it combines with a parallel degree and with the replicated meta load.
- `--enable_group_cpu_offload` and `--enable_sequential_cpu_offload` are rejected with `--fully_shard_degree > 1`, because both reach inside a parameter that sharding has replaced with a DTensor: the group hook asks each parameter whether it is pinned, for which torch registers no sharding strategy, and sequential offload rebuilds each parameter as it moves it, which needs a spec a plain tensor does not carry. `--enable_model_cpu_offload` moves whole components and does work on top of a sharded, blockwise-filled load. Combining the two does mean a component's weights are driven by two mechanisms at once — FSDP2 all-gathers the parameters, the offload hook moves the module — so anything a kernel needs that is not a parameter has to be checked against the launch device rather than assumed to have travelled: the AITER block-scale FP8 layer keeps its weight scale beside the weight as a replicated buffer and co-locates it in `forward` for that reason.

#### Load and Quantization Examples

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

These examples show how the flags are wired, not tuned recommendations: output quality, peak memory, and kernel availability depend on the checkpoint, the GPU, the torch/torchao/AITER versions, and the parallel layout.

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
