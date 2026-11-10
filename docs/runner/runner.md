# Run a model

[Supported models and features](models.md) lists every runner family, accepted
checkpoint name, and declared parallelism and optimization support.

## Quick start

[Install xDiT](installation.md) in an environment matching your device and model.
Start with one device and enough memory for the checkpoint:

```bash
xdit --model FLUX.1-dev --prompt "A cat running in a garden"
```

Accept the FLUX.1-dev license and authenticate with Hugging Face before running
the example. The first run downloads the weights. To use four devices, add `--ulysses_degree 4`;
[Ulysses sequence parallelism](../methods/usp.md) is a useful starting layout.

The model supplies default dimensions, steps, and guidance when you omit them.
The runner logs the values it uses. A parser option can be valid but unsupported
by the selected model. Initialization checks model support and layout restrictions.

## Choose your next step

| Goal | Guide |
| --- | --- |
| <a id="supported-models"></a> Find a model and compatible features | [Supported models](models.md) |
| <a id="amd-containers-and-configurations"></a><a id="defaults-and-model-support"></a> Install packages or choose a container | [Installation](installation.md) |
| <a id="launch-and-parallel-configuration"></a><a id="launcher-options"></a> Launch on multiple devices or nodes | [Launching](launching.md), [parallel methods](../methods/usp.md) |
| <a id="examples"></a><a id="multi-gpu-image-generation"></a><a id="amd-hardware-examples"></a><a id="flux1-dev-on-one-rdna-4-gpu"></a><a id="wan22-image-to-video-on-eight-instinct-gpus"></a><a id="text-to-video"></a><a id="distilled-video-generation"></a><a id="video-with-audio-from-a-reference-image-prism"></a><a id="video-with-audio-from-text-vdn-h3"></a> Generate video or use hardware-specific settings | [Model and hardware examples](examples.md) |
| <a id="per-model-load-paths"></a> Reduce loading or inference memory | [Memory and loading](memory.md), [parallel VAE](../methods/parallel_vae.md) |
| <a id="loading-and-quantization-contract"></a><a id="backend-selection-by-hardware"></a><a id="per-model-fp4-and-int8-targets"></a><a id="quantization-targets-and-schedules"></a><a id="flag-combinations-and-exclusions"></a><a id="load-and-quantization-examples"></a> <a id="quantized-attention-a2a"></a> Select kernels or precision | [Attention](attention.md), [GEMM quantization](quantization.md), [communication precision](communication.md) |
| <a id="caching-benchmarking-and-profiling"></a><a id="step-caching"></a> Skip reusable denoising work | [Caching](caching.md) |
| Apply an adapter at startup | [LoRA](lora.md) |
| <a id="timing-and-warmup"></a><a id="benchmarking-with-multiple-iterations"></a><a id="profiling"></a><a id="compilation"></a><a id="dataset-inference"></a><a id="programmatic-usage"></a><a id="output-files"></a> Save results, benchmark, profile, or call Python | [Outputs and timing](benchmarking.md) |
| Look up an option | [CLI reference](cli.md) |
| <a id="architecture"></a> Understand model integration | [Framework design](../developer/The_implement_design_of_xdit_framework.md) |
