# Community resources

Follow each resource's requirements for its models, hardware, and xDiT version.

## Projects and deployment guides

These resources have their own instructions and release versions.

| Resource | Source | Scope |
| --- | --- | --- |
| [xDiT ComfyUI](https://github.com/xdit-project/xdit-comfyui) | xdit-project | Runs supported diffusion models through xDiT's unified runner from ComfyUI. |
| [Diffusion model inference](https://github.com/AMD-AGI/diffusion-models-inference) | AMD-AGI | Provides benchmark configurations, profiling tools, tuning data, and container build infrastructure for diffusion model inference. |
| [Serve Diffusion Transformer models using xDiT container on Cloud GPUs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/xdit) | Google Cloud | Documents xDiT container deployment in Model Garden, with FLUX.1-schnell examples. |
| [xDiT diffusion inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/xdit.html?model-group=hunyuan&model=hunyuan_tag) | AMD ROCm AI Ecosystem | Documents container setup and diffusion model benchmarks on AMD Instinct `gfx942` and `gfx950` GPUs. |

For xDiT setup, see [installation and containers](runner/installation.md).
For workload configurations adapted to this CLI, see [hardware recipes](runner/examples.md#hardware-recipes).

## Model projects

StepVideo and Mochi use [external model integrations](runner/examples.md#external-model-integrations)
rather than the unified runner. Follow each project's setup and version requirements.

| Project | Source | Scope |
| --- | --- | --- |
| [Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V) | stepfun-ai | Provides StepVideo weights and inference code; its multi-GPU launcher uses xDiT's transformer TP implementation. |
| [mochi-xdit: Parallel Inference for Mochi-preview Video Generation Model with xDiT](https://github.com/xdit-project/mochi-xdit) | xdit-project | Provides a separate Mochi-preview inference implementation with Ulysses, Ring, and CFG parallelism. |
| [SkyReels V1: Human-Centric Video Foundation Model](https://github.com/SkyworkAI/SkyReels-V1) | SkyworkAI | Provides SkyReels-V1 weights and its own SkyReelsInfer framework. For xDiT's SkyReels-V2 adapters, see the [runner model list](runner/models.md). |

## Articles and case studies

Articles appear by publication date, newest first. Results apply to each article's
hardware, software, and workload.

| Published | Article | Publisher | xDiT coverage |
| --- | --- | --- | --- |
| 2026-11-02 | [How to Scale Diffusion Inference Across AMD RDNA 4 GPUs](https://rocm.blogs.amd.com/artificial-intelligence/diffusion-xdit-rdna4/README.html) | ROCm Blogs | Shows xDiT image and video generation across one, two, or four Radeon AI PRO R9700S GPUs. |
| 2026-09-17 | [Technical Dive into AMD MLPerf Inference v6.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-v6.1/README.html) | ROCm Blogs | Describes xDiT optimizations for Wan2.2-T2V-A14B within AMD's MLPerf Inference v6.1 submission. |
