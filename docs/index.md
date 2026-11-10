# xDiT documentation

Run image, video, and audiovisual diffusion models on one device or across
multiple devices and nodes. xDiT supports CUDA, ROCm, and Ascend environments;
Model and feature availability depends on the device and installed software.

**[Supported models](runner/models.md)** · **[Quick start](runner/runner.md)** · **[CLI reference](runner/cli.md)**

## Run a model

1. **[Choose a model](runner/models.md).** Check its accepted names, parallelism,
   memory, precision, and caching support in the feature tables.
2. **[Set up your environment](runner/installation.md).** Install the packages or
   choose a container for your device and model.
3. **[Generate an output](runner/runner.md).** Start on one device, then use
   [launching](runner/launching.md) to add devices or nodes. The
   [examples](runner/examples.md) cover images, video, audio, and hardware recipes.

The [CLI reference](runner/cli.md) groups options by purpose and explains their
values and defaults. Individual guides describe compatibility and practical use.

## Scale and optimize

Start with the model's defaults and measure a baseline. Ulysses is a useful first
layout for distributing attention. Choose additional optimizations for your
workload, available memory, and device connections.

| Goal | Guides |
| --- | --- |
| Distribute inference | [Ulysses and Ring](methods/usp.md), [hybrid layouts](methods/hybrid.md), [CFG](methods/cfg_parallel.md), [PipeFusion](methods/pipefusion.md) |
| Fit the workload in memory | [Loading, sharding, and offload](runner/memory.md), [parallel VAE](methods/parallel_vae.md) |
| Tune kernels and precision | [Attention backends](runner/attention.md), [GEMM quantization](runner/quantization.md), [FP8 communication and Attention A2A](runner/communication.md) |
| Reuse denoising work | [Step caching](runner/caching.md) |
| Adapt and measure | [LoRA](runner/lora.md), [benchmarking and outputs](runner/benchmarking.md) |

Check model support before combining features. Reduced precision and caching can
change output quality; compare results and timing on your workload.

## Integrate and contribute

Use [xDiT ComfyUI](https://github.com/xdit-project/xdit-comfyui), explore
[cloud, device, and container guides](resources.md), or use the separate
[HTTP service example](developer/Http_Service.md).

For development, start with [framework design](developer/The_implement_design_of_xdit_framework.md)
and [adding a model](developer/adding_models/readme.md). Contributors can
[build and update the docs](maintaining.md). Look for **中文** on guides with a
Chinese edition.
