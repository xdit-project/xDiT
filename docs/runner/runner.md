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

Run any supported model using `torchrun`:

```bash
torchrun --nproc_per_node=8 xfuser/runner.py \
    --model FLUX.1-dev \
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
from xfuser.runner import xFuserModelRunner

# Programmatic usage, still requires torchrun
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
| HunyuanVideo | `HunyuanVideo`, `tencent/HunyuanVideo` |
| HunyuanVideo-1.5 | `HunyuanVideo-1.5`, `tencent/HunyuanVideo-1.5` |
| Wan 2.1/2.2 I2V | `Wan2.1-I2V`, `Wan2.2-I2V`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| Wan 2.1/2.2 T2V | `Wan2.1-T2V`, `Wan2.2-T2V`, `Wan-AI/Wan2.1-T2V-14B-720P-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Stable Diffusion 3 | `SD3.5`, `stabilityai/stable-diffusion-3-medium-diffusers` |

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
| `--use_fsdp` | Enable FSDP | False |

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
| `--use_fp8_gemms` | Enable FP8 GEMM quantization | False |
| `--enable_tiling` | Enable VAE tiling | False |
| `--enable_slicing` | Enable VAE slicing | False |
| `--enable_model_cpu_offload` | Enable model CPU offload | False |
| `--enable_sequential_cpu_offload` | Enable sequential CPU offload | False |
| `--attention_backend` | Attention backend selection | None |

### Benchmarking

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_iterations` | Number of benchmark iterations | 1 |
| `--warmup_calls` | Warmup iterations before timing | 0 |
| `--batch_size` | Batch size for dataset inference | None |
| `--dataset_path` | Path to prompt dataset | None |
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
torchrun --nproc_per_node=4 xfuser/runner.py \
    --model FLUX.1-dev \
    --prompt "A majestic mountain landscape at sunset" \
    --height 1024 \
    --width 1024 \
    --ulysses_degree 4 \
    --num_inference_steps 50
```

### Video Generation

```bash
torchrun --nproc_per_node=8 xfuser/runner.py \
    --model HunyuanVideo \
    --prompt "A cat playing with a ball" \
    --height 720 \
    --width 1280 \
    --num_frames 49 \
    --ulysses_degree 8
```

### Benchmarking with Multiple Iterations

```bash
torchrun --nproc_per_node=8 xfuser/runner.py \
    --model FLUX.1-dev \
    --prompt "Benchmark test image" \
    --ulysses_degree 8 \
    --num_iterations 5 \
    --output_directory ./benchmark_results
```

### Profiling

```bash
torchrun --nproc_per_node=8 xfuser/runner.py \
    --model FLUX.1-dev \
    --prompt "Profile test" \
    --ulysses_degree 8 \
    --profile \
    --output_directory ./profile_results
```

### With Torch Compile

```bash
torchrun --nproc_per_node=4 xfuser/runner.py \
    --model FLUX.1-dev \
    --prompt "Compiled inference test" \
    --ulysses_degree 4 \
    --use_torch_compile
```

### Dataset Inference

```bash
torchrun --nproc_per_node=8 xfuser/runner.py \
    --model FLUX.1-dev \
    --dataset_path ./prompts.txt \
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