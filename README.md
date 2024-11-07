<div align="center">
  <!-- <h1>KTransformers</h1> -->
  <p align="center">
  
  <picture>
    <img alt="xDiT" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/XDiTlogo.png" width="50%">

  </p>
  <h3>A Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters</h3>
  <strong><a href="https://arxiv.org/abs/2405.14430">📃 Paper</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported DiTs</a> | <a href="#dev-guide">📚 Dev Guide </a> | <a href="https://github.com/xdit-project/xDiT/discussions">📈  Discussion </a> | <a href="https://medium.com/@xditproject">📝 Blogs</a></strong>
  <p></p>

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/YEWzWfCF9S)](https://discord.gg/YEWzWfCF9S)

</div>

<h2 id="agenda">Table of Contents</h2>

- [🔥 Meet xDiT](#meet-xdit)
- [📢 Updates](#updates)
- [🎯 Supported DiTs](#support-dits)
- [📈 Performance](#perf)
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
    - [DiTFastAttn](#dittfastattn)
- [📚  Develop Guide](#dev-guide)
- [🚧  History and Looking for Contributions](#history)
- [📝 Cite Us](#cite-us)


<h2 id="meet-xdit">🔥 Meet xDiT</h2>

Diffusion Transformers (DiTs) are driving advancements in high-quality image and video generation. 
With the escalating input context length in DiTs, the computational demand of the Attention mechanism grows **quadratically**! 
Consequently, multi-GPU and multi-machine deployments are essential to meet the **real-time** requirements in online services.


<h3 id="meet-xdit-parallel">Parallel Inference</h3>

To meet real-time demand for DiTs applications, parallel inference is a must.
xDiT is an inference engine designed for the parallel deployment of DiTs on large scale. 
xDiT provides a suite of efficient parallel approaches for Diffusion Models, as well as computation accelerations.

The overview of xDiT is shown as follows.

<picture>
  <img alt="xDiT" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/xdit_overview.png">
</picture>


1. Sequence Parallelism, [USP](https://arxiv.org/abs/2405.07719) is a unified sequence parallel approach combining DeepSpeed-Ulysses, Ring-Attention proposed by use3.

2. [PipeFusion](https://arxiv.org/abs/2405.14430), a sequence-level pipeline parallelism, similar to [TeraPipe](https://arxiv.org/abs/2102.07988) but takes advantage of the input temporal redundancy characteristics of diffusion models.

3. Data Parallel: Processes multiple prompts or generates multiple images from a single prompt in parallel across images.

4. CFG Parallel, also known as Split Batch: Activates when using classifier-free guidance (CFG) with a constant parallelism of 2.

The four parallel methods in xDiT can be configured in a hybrid manner, optimizing communication patterns to best suit the underlying network hardware.

As shown in the following picture, xDiT offers a set of APIs to adapt DiT models in [huggingface/diffusers](https://github.com/huggingface/diffusers) to hybrid parallel implementation through simple wrappers. 
If the model you require is not available in the model zoo, developing it yourself is straightforward; please refer to our [Dev Guide](#dev-guide).

We also have implemented the following parallel stategies for reference:

1. Tensor Parallelism
2. [DistriFusion](https://arxiv.org/abs/2402.19481)


<h3 id="meet-xdit-perf">Computing Acceleration</h3>

Optimization orthogonal to parallel focuses on accelerating single GPU performance.

First, xDiT employs a series of kernel acceleration methods. In addition to utilizing well-known Attention optimization libraries, we leverage compilation acceleration technologies such as `torch.compile` and `onediff`.

Furthermore, xDiT incorporates optimization techniques from [DiTFastAttn](https://github.com/thu-nics/DiTFastAttn), which exploits computational redundancies between different steps of the Diffusion Model to accelerate inference on a single GPU.

<h2 id="updates">📢 Updates</h2>

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
| [🎬 CogVideoX](https://huggingface.co/THUDM/CogVideoX-2b) | ✔️ | ✔️ | ❎ | 
| [🎬 Latte](https://huggingface.co/maxin-cn/Latte-1) | ❎ | ✔️ | ❎ | 
| [🔵 HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) | ✔️ | ✔️ | ✔️ |
| [🟠 Flux](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | NA | ✔️ | ✔️ |
| [🔴 PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | ✔️ | ✔️ | ✔️ |
| [🟢 PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha) | ✔️ | ✔️ | ✔️ |
| [🟠 Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | ✔️ | ✔️ | ✔️ |

</div>

### Supported by legacy version only, including DistriFusion and Tensor Parallel as the standalong parallel strategies:

<div align="center">

[🔴 DiT-XL](https://huggingface.co/facebook/DiT-XL-2-256)
</div>

<h2 id="perf">📈 Performance</h2>

<h3 id="perf_cogvideox">CogVideo</h3>

1. [CogVideo Performance Report](./docs/performance/cogvideo.md)

<h3 id="perf_flux">Flux.1</h3>

2. [Flux Performance Report](./docs/performance/flux.md)

<h3 id="perf_latte">Latte</h3>

3. [Latte Performance Report](./docs/performance/latte.md)

<h3 id="perf_hunyuandit">HunyuanDiT</h3>

4. [HunyuanDiT Performance Report](./docs/performance/hunyuandit.md)

<h3 id="perf_sd3">SD3</h3>

5. [Stable Diffusion 3 Performance Report](./docs/performance/sd3.md)

<h3 id="perf_pixart">Pixart</h3>

6. [Pixart-Alpha Performance Report (legacy)](./docs/performance/pixart_alpha_legacy.md)


<h2 id="QuickStart">🚀 QuickStart</h2>

### 1. Install from pip

```
pip install xfuser
# Or optionally, with flash_attn
pip install "xfuser[flash_attn]"
```

### 2. Install from source 

```
pip install -e .
# Or optionally, with flash_attn
pip install -e ".[flash_attn]"
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

---

<details>
<summary>Click to see available options for the PixArt-alpha example</summary>

```bash
python ./examples/pixartalpha_example.py -h

...

xFuser Arguments

options:
  -h, --help            show this help message and exit

Model Options:
  --model MODEL         Name or path of the huggingface model to use.
  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights, default to the default cache dir of huggingface.
  --trust-remote-code   Trust remote code from huggingface.

Runtime Options:
  --warmup_steps WARMUP_STEPS
                        Warmup steps in generation.
  --use_parallel_vae
  --use_torch_compile   Enable torch.compile to accelerate inference in a single card
  --seed SEED           Random seed for operations.
  --output_type OUTPUT_TYPE
                        Output type of the pipeline.
  --enable_sequential_cpu_offload
                        Offloading the weights to the CPU.

Parallel Processing Options:
  --use_cfg_parallel    Use split batch in classifier_free_guidance. cfg_degree will be 2 if set
  --data_parallel_degree DATA_PARALLEL_DEGREE
                        Data parallel degree.
  --ulysses_degree ULYSSES_DEGREE
                        Ulysses sequence parallel degree. Used in attention layer.
  --ring_degree RING_DEGREE
                        Ring sequence parallel degree. Used in attention layer.
  --pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE
                        Pipefusion parallel degree. Indicates the number of pipeline stages.
  --num_pipeline_patch NUM_PIPELINE_PATCH
                        Number of patches the feature map should be segmented in pipefusion parallel.
  --attn_layer_num_for_pp [ATTN_LAYER_NUM_FOR_PP ...]
                        List representing the number of layers per stage of the pipeline in pipefusion parallel
  --tensor_parallel_degree TENSOR_PARALLEL_DEGREE
                        Tensor parallel degree.
  --split_scheme SPLIT_SCHEME
                        Split scheme for tensor parallel.

Input Options:
  --height HEIGHT       The height of image
  --width WIDTH         The width of image
  --prompt [PROMPT ...]
                        Prompt for the model.
  --no_use_resolution_binning
  --negative_prompt [NEGATIVE_PROMPT ...]
                        Negative prompt for the model.
  --num_inference_steps NUM_INFERENCE_STEPS
                        Number of inference steps.
```

</details>

---

Hybriding multiple parallelism techniques togather is essential for efficiently scaling. 
It's important that the product of all parallel degrees matches the number of devices. 
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
--prompt "A small dog" \
--use_cfg_parallel
```

⚠️ Applying PipeFusion requires setting `warmup_steps`, also required in DistriFusion, typically set to a small number compared with `num_inference_steps`.
The warmup step impacts the efficiency of PipeFusion as it cannot be executed in parallel, thus degrading to a serial execution. 
We observed that a warmup of 0 had no effect on the PixArt model.
Users can tune this value according to their specific tasks.


<h2 id="comfyui">🖼️ ComfyUI with xDiT</h2>

### 1. Launch ComfyUI

ComfyUI, is the most popular web-based Diffusion Model interface optimized for workflow. 
It provides users with a UI platform for image generation, supporting plugins like LoRA, ControlNet, and IPAdaptor.
Yet, its design for native single-GPU usage leaves it struggling with the demands of today’s large DiTs, resulting in unacceptably high latency for users like Flux.1. Leveraging the power of xDiT, we’ve successfully implemented a multi-GPU parallel processing workflow within ComfyUI, effectively addressing Flux.1’s performance challenges.
Below is an example of using xDiT to accelerate a Flux workflow with LoRA:

![ComfyUI xDiT Demo](https://raw.githubusercontent.com/xdit-project/xdit_assets/main/comfyui/flux-demo.gif)

More details can be found in our Medium article: [Supercharge Your AIGC Experience: Leverage xDiT for Multiple GPU Parallel in ComfyUI Flux.1 Workflow](https://medium.com/@xditproject/supercharge-your-aigc-experience-leverage-xdit-for-multiple-gpu-parallel-in-comfyui-flux-1-54b34e4bca05).

Currently, if you need the parallel version of ComfyUI applying xDiT, please contact us via email [jiaruifang@tencent.com](jiaruifang@tencent.com).

### 2. Launch a Http Service

You can also launch a http service to generate images with xDiT.

[Launching a Text-to-Image Http Service](./docs/developer/Http_Service.md)


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

As we can see, PipeFusion and Sequence Parallel achieve lowest communication cost on different scales and hardware configurations, making them suitable foundational components for a hybrid approach.

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


<h4 id="dittfastattn">DiTFastAttn</h4>

xDiT also provides DiTFastAttn for single GPU acceleration. It can reduce computation cost of attention layer by leveraging redundancies between different steps of the Diffusion Model.

[DiTFastAttn: Attention Compression for Diffusion Transformer Models](./docs/methods/ditfastattn.md)

<h2 id="dev-guide">📚  Develop Guide</h2>

[The implement and design of xdit framework](./docs/developer/The_implement_design_of_xdit_framework.md)

[Manual for adding new models](./docs/developer/Manual_for_Adding_New_Models.md)

<h2 id="history">🚧  History and Looking for Contributions</h2>

We conducted a major upgrade of this project in August 2024.

The latest APIs is located in the [xfuser/](./xfuser/) directory, supports hybrid parallelism. It offers clearer and more structured code but currently supports fewer models.

The legacy APIs is in the [legacy](https://github.com/xdit-project/xDiT/tree/legacy) branch, limited to single parallelism. It supports a richer of parallel methods, including PipeFusion, Sequence Parallel, DistriFusion, and Tensor Parallel. CFG Parallel can be hybrid with PipeFusion but not with other parallel methods.

For models not yet supported by the latest APIs, you can run the examples in the [scripts/](https://github.com/xdit-project/xDiT/tree/legacy/scripts) directory under branch `legacy`. If you wish to develop new features on a model or require hybrid parallelism, stay tuned for further project updates. 

We also welcome developers to join and contribute more features and models to the project. Tell us which model you need in xDiT in [discussions](https://github.com/xdit-project/xDiT/discussions).

<h2 id="cite-us">📝 Cite Us</h2>


[xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism](https://arxiv.org/abs/2411.01738)

```
@misc{fang2024xditinferenceenginediffusion,
      title={xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism}, 
      author={Jiarui Fang and Jinzhe Pan and Xibo Sun and Aoyu Li and Jiannan Wang},
      year={2024},
      eprint={2411.01738},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.01738}, 
}

```

[PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference](https://arxiv.org/abs/2405.14430)

```
@misc{fang2024pipefusionpatchlevelpipelineparallelism,
      title={PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference}, 
      author={Jiarui Fang and Jinzhe Pan and Jiannan Wang and Aoyu Li and Xibo Sun},
      year={2024},
      eprint={2405.14430},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.14430}, 
}

```

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)


```
@misc{fang2024uspunifiedsequenceparallelism,
      title={USP: A Unified Sequence Parallelism Approach for Long Context Generative AI}, 
      author={Jiarui Fang and Shangchun Zhao},
      year={2024},
      eprint={2405.07719},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.07719}, 
}

```
