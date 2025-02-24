<div align="center">
  <!-- <h1>KTransformers</h1> -->
  <p align="center">

  <picture>
    <img alt="xDiT" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/XDiTlogo.png" width="50%">

  </p>
  <h3>A Scalable Inference Engine for Diffusion Transformers (DiTs) on Multiple Computing Devices</h3>
  <a href="#cite-us">📝 Papers</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported DiTs</a> | <a href="#dev-guide">📚 Dev Guide </a> | <a href="https://github.com/xdit-project/xDiT/discussions">📈  Discussion </a> | <a href="https://medium.com/@xditproject">📝 Blogs</a></strong>
  <p></p>

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/YEWzWfCF9S)](https://discord.gg/YEWzWfCF9S)

</div>

<h2 id="agenda">Table of Contents</h2>

- [🔥 Meet xDiT](#meet-xdit)
- [📢 Updates](#updates)
- [🎯 Supported DiTs](#support-dits)
- [📈 Performance](#perf)
  - [HunyuanVideo](#perf_hunyuanvideo)
  - [ConsisID](#perf_consisid)
  - [Mochi-1](#perf_mochi1)
  - [CogVideoX](#perf_cogvideox)
  - [Flux.1](#perf_flux)
  - [HunyuanDiT](#perf_hunyuandit)
  - [SD3](#perf_sd3)
  - [Pixart](#perf_pixart)
  - [Latte](#perf_latte)
- [🚀 QuickStart](#QuickStart)
- [🖼️ ComfyUI with xDiT](#comfyui)
- [✨ xDiT's Arsenal](#secrets)
  - [Parallel Methods](#parallel)
    - [1. PipeFusion](#PipeFusion)
    - [2. Unified Sequence Parallel](#USP)
    - [3. Hybrid Parallel](#hybrid_parallel)
    - [4. CFG Parallel](#cfg_parallel)
    - [5. Parallel VAE](#parallel_vae)
  - [Single GPU Acceleration](#1gpuacc)
    - [Compilation Acceleration](#compilation)
    - [Cache Acceleration](#cache_acceleration)
- [📚  Develop Guide](#dev-guide)
- [🚧  History and Looking for Contributions](#history)
- [📝 Cite Us](#cite-us)


<h2 id="meet-xdit">🔥 Meet xDiT</h2>

Diffusion Transformers (DiTs) are driving advancements in high-quality image and video generation. 
With the escalating input context length in DiTs, the computational demand of the Attention mechanism grows **quadratically**! 
Consequently, multi-GPU and multi-machine deployments are essential to meet the **real-time** requirements in online services.


<h3 id="meet-xdit-parallel">Parallel Inference</h3>

To meet real-time demand for DiTs applications, parallel inference is a must.
xDiT is an inference engine designed for the parallel deployment of DiTs on a large scale. 
xDiT provides a suite of efficient parallel approaches for Diffusion Models, as well as computation accelerations.

The overview of xDiT is shown as follows.

<picture>
  <img alt="xDiT" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/xdit_overview.png">
</picture>


1. Sequence Parallelism, [USP](https://arxiv.org/abs/2405.07719) is a unified sequence parallel approach proposed by us combining DeepSpeed-Ulysses, Ring-Attention.

2. [PipeFusion](https://arxiv.org/abs/2405.14430), a sequence-level pipeline parallelism, similar to [TeraPipe](https://arxiv.org/abs/2102.07988) but takes advantage of the input temporal redundancy characteristics of diffusion models.

3. Data Parallel: Processes multiple prompts or generates multiple images from a single prompt in parallel across images.

4. CFG Parallel, also known as Split Batch: Activates when using classifier-free guidance (CFG) with a constant parallelism of 2.

The four parallel methods in xDiT can be configured in a hybrid manner, optimizing communication patterns to best suit the underlying network hardware.

As shown in the following picture, xDiT offers a set of APIs to adapt DiT models in [huggingface/diffusers](https://github.com/huggingface/diffusers) to hybrid parallel implementation through simple wrappers. 
If the model you require is not available in the model zoo, developing it by yourself is not so difficult; please refer to our [Dev Guide](#dev-guide).

We also have implemented the following parallel strategies for reference:

1. Tensor Parallelism
2. [DistriFusion](https://arxiv.org/abs/2402.19481)

<h3 id="meet-xdit-cache">Cache Acceleration</h3>

Cache method, including [TeaCache](https://github.com/ali-vilab/TeaCache.git), [First-Block-Cache](https://github.com/chengzeyi/ParaAttention.git) and [DiTFastAttn](https://github.com/thu-nics/DiTFastAttn), which exploits computational redundancies between different steps of the Diffusion Model to accelerate inference on a single GPU.

<h3 id="meet-xdit-perf">Computing Acceleration</h3>

Optimization is orthogonal to parallel and focuses on accelerating performance on a single GPU.

First, xDiT employs a series of kernel acceleration methods. In addition to utilizing well-known Attention optimization libraries, we leverage compilation acceleration technologies such as `torch.compile` and `onediff`.


<h2 id="updates">📢 Updates</h2>

* 🎉**December 24, 2024**: xDiT supports [ConsisID-Preview](https://github.com/PKU-YuanGroup/ConsisID) and achieved 3.21x speedup compare to the official implementation! The inference scripts are [examples/consisid_example.py](examples/consisid_example.py) and [examples/consisid_usp_example.py](examples/consisid_usp_example.py).
* 🎉**December 7, 2024**: xDiT is the official parallel inference engine for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), reducing the 5-sec video generation latency from 31 minutes to 5 minutes on 8xH100!
* 🎉**November 28, 2024**: xDiT achieves 1.6 sec end-to-end latency for 28-step [Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) inference on 4xH100!
* 🎉**November 20, 2024**: xDiT supports [CogVideoX-1.5](https://huggingface.co/THUDM/CogVideoX1.5-5B) and achieved 6.12x speedup compare to the implementation in diffusers!
* 🎉**November 11, 2024**: xDiT has been applied to [mochi-1](https://github.com/xdit-project/mochi-xdit) and achieved 3.54x speedup compare to the official open source implementation!
* 🎉**October 10, 2024**: xDiT applied DiTFastAttn to accelerate single GPU inference for Pixart Models!
* 🎉**September 26, 2024**: xDiT has been officially used by [THUDM/CogVideo](https://github.com/THUDM/CogVideo)! The inference scripts are placed in [parallel_inference/](https://github.com/THUDM/CogVideo/blob/main/tools/parallel_inference) at their repository.
* 🎉**September 23, 2024**: Support CogVideoX. The inference scripts are [examples/cogvideox_example.py](examples/cogvideox_example.py).
* 🎉**August 26, 2024**: We apply torch.compile and [onediff](https://github.com/siliconflow/onediff) nexfort backend to accelerate GPU kernels speed.
* 🎉**August 15, 2024**: Support Hunyuan-DiT hybrid parallel version. The inference scripts are [examples/hunyuandit_example.py](examples/hunyuandit_example.py).
* 🎉**August 9, 2024**: Support Latte sequence parallel version. The inference scripts are [examples/latte_example.py](examples/latte_example.py).
* 🎉**August 8, 2024**: Support Flux sequence parallel version. The inference scripts are [examples/flux_example.py](examples/flux_example.py).
* 🎉**August 2, 2024**: Support Stable Diffusion 3 hybrid parallel version. The inference scripts are [examples/sd3_example.py](examples/sd3_example.py).
* 🎉**July 18, 2024**: Support PixArt-Sigma and PixArt-Alpha. The inference scripts are [examples/pixartsigma_example.py](examples/pixartsigma_example.py), [examples/pixartalpha_example.py](examples/pixartalpha_example.py).
* 🎉**July 17, 2024**: Rename the project to xDiT. The project has evolved from a collection of parallel methods into a unified inference framework and supported the hybrid parallel for DiTs.
* 🎉**May 24, 2024**: PipeFusion is public released. It supports PixArt-alpha [scripts/pixart_example.py](./scripts/pixart_example.py), DiT [scripts/ditxl_example.py](./scripts/ditxl_example.py) and SDXL [scripts/sdxl_example.py](./scripts/sdxl_example.py). This version is currently in the `legacy` branch.


<h2 id="support-dits">🎯 Supported DiTs</h2>

<div align="center">

| Model Name | CFG | SP | PipeFusion |
| --- | --- | --- | --- |
| [🎬 HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | NA | ✔️ | ❎ |
| [🎬 ConsisID-Preview](https://github.com/PKU-YuanGroup/ConsisID) | ✔️ | ✔️ | ❎ |
| [🎬 CogVideoX1.5](https://huggingface.co/THUDM/CogVideoX1.5-5B) | ✔️ | ✔️ | ❎ |
| [🎬 Mochi-1](https://github.com/xdit-project/mochi-xdit) | ✔️ | ✔️ | ❎ |
| [🎬 CogVideoX](https://huggingface.co/THUDM/CogVideoX-2b) | ✔️ | ✔️ | ❎ |
| [🎬 Latte](https://huggingface.co/maxin-cn/Latte-1) | ❎ | ✔️ | ❎ |
| [🔵 HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) | ✔️ | ✔️ | ✔️ |
| [🟠 Flux](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | NA | ✔️ | ✔️ |
| [🔴 PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | ✔️ | ✔️ | ✔️ |
| [🟢 PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha) | ✔️ | ✔️ | ✔️ |
| [🟠 Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | ✔️ | ✔️ | ✔️ |

</div>


[🔴 DiT-XL](https://huggingface.co/facebook/DiT-XL-2-256) is supported by legacy version only, including DistriFusion and Tensor Parallel as the standalone parallel strategies:



<h2 id="comfyui">🖼️ TACO-DiT: ComfyUI with xDiT</h2>

ComfyUI, is the most popular web-based Diffusion Model interface optimized for workflow. 
It provides users with a UI platform for image generation, supporting plugins like LoRA, ControlNet, and IPAdaptor. Yet, its design for native single-GPU usage leaves it struggling with the demands of today’s large DiTs, resulting in unacceptably high latency for users like Flux.1. 

Using our commercial project **TACO-DiT**, a close-sourced ComfyUI variant built with xDiT, we’ve successfully implemented a multi-GPU parallel processing workflow within ComfyUI, effectively addressing Flux.1’s performance challenges. Below is an example of using TACO-DiT to accelerate a Flux workflow with LoRA:

![ComfyUI xDiT Demo](https://raw.githubusercontent.com/xdit-project/xdit_assets/main/comfyui/flux-demo.gif)

By using TACO-DiT, you could significantly reduce your ComfyUI workflow inference latency, and  boosting the throughput with Multi-GPUs. Now it is compatible with multiple Plug-ins, including ControlNet and LoRAs.

More features and details can be found in our Intro Video: 
+ [[YouTube] TACO-DiT: Accelerating Your ComfyUI Generation Experience](https://www.youtube.com/watch?v=7DXnGrARqys) 
+ [[Bilibili] TACO-DiT: 加速你的ComfyUI生成体验](https://www.bilibili.com/video/BV18tU7YbEra/?vd_source=59c1f990379162c8f596974f34224e4f)

The blog article is also available: [Supercharge Your AIGC Experience: Leverage xDiT for Multiple GPU Parallel in ComfyUI Flux.1 Workflow](https://medium.com/@xditproject/supercharge-your-aigc-experience-leverage-xdit-for-multiple-gpu-parallel-in-comfyui-flux-1-54b34e4bca05).   

<h2 id="perf">📈 Performance</h2>

<h3 id="perf_hunyuanvideo">HunyuanVideo</h3>

1. [HunyuanVideo Performance Report](./docs/performance/hunyuanvideo.md)

<h3 id="perf_consisid">ConsisID-Preview</h3>

2. [ConsisID Performance Report](./docs/performance/consisid.md)

<h3 id="perf_cogvideox">Mochi1</h3>

3. [mochi1-xdit: Reducing the Inference Latency by 3.54x Compare to the Official Open Souce Implementation!](https://github.com/xdit-project/mochi-xdit)

<h3 id="perf_cogvideox">CogVideo</h3>

4. [CogVideo Performance Report](./docs/performance/cogvideo.md)

<h3 id="perf_flux">Flux.1</h3>

5. [Flux Performance Report](./docs/performance/flux.md)

<h3 id="perf_latte">Latte</h3>

6. [Latte Performance Report](./docs/performance/latte.md)

<h3 id="perf_hunyuandit">HunyuanDiT</h3>

7. [HunyuanDiT Performance Report](./docs/performance/hunyuandit.md)

<h3 id="perf_sd3">SD3</h3>

8. [Stable Diffusion 3 Performance Report](./docs/performance/sd3.md)

<h3 id="perf_pixart">Pixart</h3>

9. [Pixart-Alpha Performance Report (legacy)](./docs/performance/pixart_alpha_legacy.md)


<h2 id="QuickStart">🚀 QuickStart</h2>

### 1. Install from pip

We set `diffusers` and `flash_attn` as two optional installation requirements.

About `diffusers` version: 
- If you only use the USP interface, `diffusers` is not required. Models are typically released as `nn.Module`
 first, before being integrated into diffusers. xDiT sometimes is applied as an USP plugin to existing projects.
- Different models may require different diffusers versions. Model implementations can vary between diffusers versions (e.g., Flux), which affects parallel processing. When encountering model execution errors, you may need to try several recent diffusers versions.
- While we specify a diffusers version in `setup.py`, newer models may require later versions or even installation from main branch.

About `flash_attn` version:
- Without `flash_attn` installed, xDiT falls back to a PyTorch implementation of ring attention, which helps NPU users with compatibility
- However, not using `flash_attn` on GPUs may result in suboptimal performance. For best GPU performance, we recommend installing `flash_attn`.

```
pip install xfuser  # Basic installation
pip install "xfuser[diffusers,flash-attn]"  # With both diffusers and flash attention
```

### 2. Install from source 

```
pip install -e .
# Or optionally, with diffusers
pip install -e ".[diffusers,flash-attn]"
```

Note that we use two self-maintained packages:

1. [yunchang](https://github.com/feifeibear/long-context-attention)
2. [DistVAE](https://github.com/xdit-project/DistVAE)

The [flash_attn](https://github.com/Dao-AILab/flash-attention) used for yunchang should be >= 2.6.0

### 3. Docker

We provide a docker image for developers to develop with xDiT. The docker image is [thufeifeibear/xdit-dev](https://hub.docker.com/r/thufeifeibear/xdit-dev).

### 4. Usage

We provide examples demonstrating how to run models with xDiT in the [./examples/](./examples/) directory. 
You can easily modify the model type, model directory, and parallel options in the [examples/run.sh](examples/run.sh) within the script to run some already supported DiT models.

```bash
bash examples/run.sh
```

Hybridizing multiple parallelism techniques together is essential for efficiently scaling. 
It's important that **the product of all parallel degrees matches the number of devices**. 
Note use_cfg_parallel means cfg_parallel=2. For instance, you can combine CFG, PipeFusion, and sequence parallelism with the command below to generate an image of a cute dog through hybrid parallelism. 
Here ulysses_degree * pipefusion_parallel_degree * cfg_degree(use_cfg_parallel) == number of devices == 8.


```bash
torchrun --nproc_per_node=8 \
examples/pixartalpha_example.py \
--model models/PixArt-XL-2-1024-MS \
--pipefusion_parallel_degree 2 \
--ulysses_degree 2 \
--num_inference_steps 20 \
--warmup_steps 0 \
--prompt "A cute dog" \
--use_cfg_parallel
```

⚠️ Applying PipeFusion requires setting `warmup_steps`, also required in DistriFusion, typically set to a small number compared with `num_inference_steps`.
The warmup step impacts the efficiency of PipeFusion as it cannot be executed in parallel, thus degrading to a serial execution. 
We observed that a warmup of 0 had no effect on the PixArt model.
Users can tune this value according to their specific tasks.

### 5. Launch parallel inference example with ray

We also provide a ray example to launch parallel inference. With ray, we can disaggregate the VAE module and DiT backbone, and allocate different GPU parallelism for them.

[Launch parallel inference example with ray](./examples/ray/README.md)

### 6. Launch an HTTP Service

You can also launch an HTTP service to generate images with xDiT.

[Launching a Text-to-Image Http Service](./docs/developer/Http_Service.md)

<h2 id="dev-guide">📚  Develop Guide</h2>

We provide a step-by-step guide for adding new models, please refer to the following tutorial.

[Apply xDiT to new models](./docs/developer/adding_models/readme.md)

A high-level design of xDiT framework is provided below, which may help you understand the xDiT framework.

[The implement and design of xdit framework](./docs/developer/The_implement_design_of_xdit_framework.md)

<h2 id="secrets">✨ The xDiT's Arsenal</h2>

The remarkable performance of xDiT is attributed to two key facets.
Firstly, it leverages parallelization techniques, pioneering innovations such as USP, PipeFusion, and hybrid parallelism, to scale DiTs inference to unprecedented scales.

Secondly, we employ compilation technologies to enhance execution on GPUs, integrating established solutions like `torch.compile` and `onediff` to optimize xDiT's performance.

<h3 id="parallel">1. Parallel Methods</h3>

As illustrated in the accompanying images, xDiTs offer a comprehensive set of parallelization techniques. For the DiT backbone, the foundational methods—Data, USP, PipeFusion, and CFG parallel—operate in a hybrid fashion. Additionally, the distinct methods, Tensor and DistriFusion parallel, function independently.
For the VAE module, xDiT offers a parallel implementation, [DistVAE](https://github.com/xdit-project/DistVAE), designed to prevent out-of-memory (OOM) issues.
The (<span style="color: red;">xDiT</span>) highlights the methods first proposed by use.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/xdit_method.png" alt="xdit methods">
</div>

The communication and memory costs associated with the aforementioned intra-image parallelism, except for the CFG and DP (they are inter-image parallel), in DiTs are detailed in the table below. (* denotes that communication can be overlapped with computation.)

As we can see, PipeFusion and Sequence Parallel achieve the lowest communication cost on different scales and hardware configurations, making them suitable foundational components for a hybrid approach.

𝒑: Number of pixels;\
𝒉𝒔: Model hidden size;\
𝑳: Number of model layers;\
𝑷: Total model parameters;\
𝑵: Number of parallel devices;\
𝑴: Number of patch splits;\
𝑸𝑶: Query and Output parameter count;\
𝑲𝑽: KV Activation parameter count;\
𝑨 = 𝑸 = 𝑶 = 𝑲 = 𝑽: Equal parameters for Attention, Query, Output, Key, and Value;


|                           | attn-KV | communication cost           | param memory   | activations memory             | extra buff memory                  |
|:-------------------------:|:-------:|:----------------------------:|:--------------:|:------------------------------:|:----------------------------------:|
| Tensor Parallel           | fresh   | $4O(p \times hs)L$           | $\frac{1}{N}P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$     |
| DistriFusion*             | stale   | $2O(p \times hs)L$           | $P$            | $\frac{2}{N}A = \frac{1}{N}QO$ | $2AL = (KV)L$                      |
| Ring Sequence Parallel*   | fresh   | $2O(p \times hs)L$           | $P$            | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$     |
| Ulysses Sequence Parallel | fresh   | $\frac{4}{N}O(p \times hs)L$ | $P$            | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$     |
| PipeFusion*               | stale-  | $2O(p \times hs)$            | $\frac{1}{N}P$ | $\frac{2}{M}A = \frac{1}{M}QO$ | $\frac{2L}{N}A = \frac{1}{N}(KV)L$ |


<h4 id="PipeFusion">1.1. PipeFusion</h4>

[PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models](./docs/methods/pipefusion.md)

<h4 id="USP">1.2. USP: Unified Sequence Parallelism</h4>

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](./docs/methods/usp.md)

<h4 id="hybrid_parallel">1.3. Hybrid Parallel</h4>

[Hybrid Parallelism](./docs/methods/hybrid.md)

<h4 id="cfg_parallel">1.4. CFG Parallel</h4>

[CFG Parallel](./docs/methods/cfg_parallel.md)

<h4 id="parallel_vae">1.5. Parallel VAE</h4>

[Patch Parallel VAE](./docs/methods/parallel_vae.md)

<h3 id="1gpuacc">Single GPU Acceleration</h3>


<h4 id="compilation">Compilation Acceleration</h4>

We utilize two compilation acceleration techniques, [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [onediff](https://github.com/siliconflow/onediff), to enhance runtime speed on GPUs. These compilation accelerations are used in conjunction with parallelization methods.

We employ the nexfort backend of onediff. Please install it before use:

```
pip install onediff
pip install -U nexfort
```

For usage instructions, refer to the [example/run.sh](./examples/run.sh). Simply append `--use_torch_compile` or `--use_onediff` to your command. Note that these options are mutually exclusive, and their performance varies across different scenarios.

<h4 id="cache_acceleration">Cache Acceleration</h4>

You can use `--use_teacache` or `--use_fbcache` in examples/run.sh, which applies TeaCache and First-Block-Cache respectively. 
Note, cache method is only supported for FLUX model with USP. It is currently not applicable for PipeFusion.

xDiT also provides DiTFastAttn for single GPU acceleration. It can reduce the computation cost of attention layers by leveraging redundancies between different steps of the Diffusion Model.

[DiTFastAttn: Attention Compression for Diffusion Transformer Models](./docs/methods/ditfastattn.md)

<h2 id="history">🚧  History and Looking for Contributions</h2>

We conducted a major upgrade of this project in August 2024, introducing a new set of APIs that are now the preferred choice for all users.

The legacy APIs are applied in early stage of xDiT to explore and compare different parallelization methods.
They are located in the [legacy](https://github.com/xdit-project/xDiT/tree/legacy) branch, are now considered outdated and do not support hybrid parallelism. Despite this limitation, they offer a broader range of individual parallelization methods, including PipeFusion, Sequence Parallel, DistriFusion, and Tensor Parallel.

For users working with Pixart models, you can still run the examples in the [scripts/](https://github.com/xdit-project/xDiT/tree/legacy/scripts) directory under the `legacy` branch. However, for all other models, we strongly recommend adopting the formal APIs to ensure optimal performance and compatibility.

We also warmly welcome developers to join us in enhancing the project. If you have ideas for new features or models, please share them in our [issues](https://github.com/xdit-project/xDiT/issues). Your contributions are invaluable in driving the project forward and ensuring it meets the needs of the community.

<h2 id="cite-us">📝 Cite Us</h2>


[xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism](https://arxiv.org/abs/2411.01738)

```
@article{fang2024xdit,
  title={xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism},
  author={Fang, Jiarui and Pan, Jinzhe and Sun, Xibo and Li, Aoyu and Wang, Jiannan},
  journal={arXiv preprint arXiv:2411.01738},
  year={2024}
}

```

[PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference](https://arxiv.org/abs/2405.14430)

```
@article{fang2024pipefusion,
  title={PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference},
  author={Jiarui Fang and Jinzhe Pan and Jiannan Wang and Aoyu Li and Xibo Sun},
  journal={arXiv preprint arXiv:2405.14430},
  year={2024}
}

```

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)


```
@article{fang2024unified,
  title={A Unified Sequence Parallelism Approach for Long Context Generative AI},
  author={Fang, Jiarui and Zhao, Shangchun},
  journal={arXiv preprint arXiv:2405.07719},
  year={2024}
}

```

[Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study](https://arxiv.org/abs/2411.13588)

```
@article{sun2024unveiling,
  title={Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study},
  author={Sun, Xibo and Fang, Jiarui and Li, Aoyu and Pan, Jinzhe},
  journal={arXiv preprint arXiv:2411.13588},
  year={2024}
}

```
