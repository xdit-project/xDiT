<div align="center">
  <!-- <h1>KTransformers</h1> -->
  <p align="center">
  
  <picture>
    <img alt="xDiT" src="./assets/XDiTlogo.png" width=50%>

  </picture>

  </p>
  <h3>A Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters</h3>
  <strong><a href="https://arxiv.org/abs/2405.14430">📃 Paper</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported DiTs</a> | <a href="#dev-guide">📚 Dev Guide </a> | <a href="https://github.com/xdit-project/xDiT/discussions">📈  Discussion </a> </strong>
</div>

<h2 id="agenda">Table of Contents</h2>

- [🔥 Meet xDiT](#meet-xdit)
- [📢 Updates](#updates)
- [🎯 Supported DiTs](#support-dits)
- [📈 Performance](#perf)
- [🚀 QuickStart](#QuickStart)
- [✨ the xDiT's secret weapons](#secrets)
  - [1. PipeFusion](#PipeFusion)
  - [2. USP](#USP)
  - [3. Hybrid Parallel](#hybrid_parallel)
  - [4. CFG Parallel](#cfg_parallel)
  - [5. Parallel VAE](#parallel_vae)
- [📚  Develop Guide](#dev-guide)
- [🚧  History and Looking for Contributions](#history)
- [📝 Cite Us](#cite-us)


<h2 id="meet-xdit">🔥 Meet xDiT</h2>

Diffusion Transformers (DiTs), pivotal in text-to-image and text-to-video models, are driving advancements in high-quality image and video generation. 
With the escalating input sequence length in DiTs, the computational demand of the Attention mechanism grows **quadratically**! 
Consequently, multi-GPU and multi-machine deployments are essential to maintain real-time performance in online services.

To meet real-time demand for DiTs applications, parallel inference is a must.
xDiT is an inference engine designed for the parallel deployment of DiTs on large scale. 
xDiT provides a suite of efficient parallel inference approaches for Diffusion Models.

1. Sequence Parallelism, [USP](https://arxiv.org/abs/2405.07719) is a unified sequence parallel approach combining DeepSpeed-Ulysses, Ring-Attention.

2. [PipeFusion](https://arxiv.org/abs/2405.14430), a patch level pipeline parallelism using displaced patch by taking advantage of the diffusion model characteristics.

3. Data Parallel: Processes multiple prompts or generates multiple images from a single prompt in parallel across images.

4. CFG Parallel, also known as Split Batch: Activates when using classifier-free guidance (CFG) with a constant parallelism of 2.

The four parallel methods in xDiT can be configured in a hybrid manner, optimizing communication patterns to best suit the underlying network hardware.

xDiT offers a set of APIs to adapt DiT models in [huggingface/diffusers](https://github.com/huggingface/diffusers) to hybrid parallel implementation through simple wrappers. 
If the model you require is not available in the model zoo, developing it yourself is straightforward; please refer to our [Dev Guide](#dev-guide).


We also have implemented the following parallel stategies for reference:

1. Tensor Parallelism
2. [DistriFusion](https://arxiv.org/abs/2402.19481)

The communication and memory costs associated with the aforementioned parallelism, except for the CFG and DP, in DiTs are detailed in the table below. (* denotes that communication can be overlapped with computation.)


As we can see, PipeFusion and Sequence Parallel achieve lowest communication cost on different scales and hardware configurations, making them suitable foundational components for a hybrid approach.

𝒑: Number of pixels;
𝒉𝒔: Model hidden size;
𝑳: Number of model layers;
𝑷: Total model parameters;
𝑵: Number of parallel devices;
𝑴: Number of patch splits;
𝑸𝑶: Query and Output parameter count;
𝑲𝑽: KV Activation parameter count;
𝑨 = 𝑸 = 𝑶 = 𝑲 = 𝑽: Equal parameters for Attention, Query, Output, Key, and Value;

<div align="center">

|          | attn-KV | communication cost | param memory | activations memory | extra buff memory |
|:--------:|:-------:|:-----------------:|:-----:|:-----------:|:----------:|
| Tensor Parallel | fresh | $4O(p \times hs)L$ | $\frac{1}{N}P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| DistriFusion* | stale | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $2AL = (KV)L$ |
| Ring Sequence Parallel* | fresh | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| Ulysses Sequence Parallel | fresh | $\frac{4}{N}O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| PipeFusion* | stale- | $2O(p \times hs)$ | $\frac{1}{N}P$ | $\frac{2}{M}A = \frac{1}{M}QO$ | $\frac{2L}{N}A = \frac{1}{N}(KV)L$ |

</div>

<h2 id="updates">📢 Updates</h2>

* 🎉**August 2, 2024**: Support Stable Diffusion 3 hybrid parallel version. The inference scripts are [examples/sd3_example](examples/sd3_example.py).
* 🎉**July 18, 2024**: Support PixArt-Sigma and PixArt-Alpha. The inference scripts are [examples/pixartsigma_example.py](examples/pixartsigma_example.py), [examples/pixartalpha_example.py](examples/pixartalpha_example.py).
* 🎉**July 17, 2024**: Rename the project to xDiT. The project has evolved from a collection of parallel methods into a unified inference framework and supported the hybrid parallel for DiTs.
* 🎉**July 10, 2024**: Support HunyuanDiT. The inference script is [legacy/scripts/hunyuandit_example.py](./legacy/scripts/hunyuandit_example.py).
* 🎉**June 26, 2024**: Support Stable Diffusion 3. The inference script is [legacy/scripts/sd3_example.py](./legacy/scripts/sd3_example.py).
* 🎉**May 24, 2024**: PipeFusion is public released. It supports PixArt-alpha [legacy/scripts/pixart_example.py](./legacy/scripts/pixart_example.py), DiT [legacy/scripts/ditxl_example.py](./legacy/scripts/ditxl_example.py) and SDXL [legacy/scripts/sdxl_example.py](./legacy/scripts/sdxl_example.py).


<h2 id="support-dits">🎯 Supported DiTs</h2>

-  [🔴 PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
-  [🔵 HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)
-  [🟢 PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha)
-  [🟠 Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
-  [🔴 DiT-XL](https://huggingface.co/facebook/DiT-XL-2-256)


<h2 id="perf">📈 Performance</h2>

Here are the benchmark results for Pixart-Alpha using the 20-step DPM solver as the scheduler across various image resolutions. To replicate these findings, please refer to the script at [./legacy/scripts/benchmark.sh](./legacy/scripts/benchmark.sh).

**TBD**: Updates results on hybrid parallelism.

1. The Latency on 4xA100-80GB (PCIe)

<div align="center">
    <img src="./assets/latency-A100-PCIe.png" alt="A100 PCIe latency">
</div>

2. The Latency on 8xL20-48GB (PCIe)

<div align="center">
    <img src="./assets/latency-L20.png" alt="L20 latency">
</div>

3. The Latency on 8xA100-80GB (NVLink)

<div align="center">
    <img src="./assets/latency-A100-NVLink.png" alt="latency-A100-NVLink">
</div>

4. The Latency on 4xT4-16GB (PCIe)

<div align="center">
    <img src="./assets/latency-T4.png" 
    alt="latency-T4">
</div>


<h2 id="QuickStart">🚀 QuickStart</h2>

1. Install yunchang for sequence parallel.

Install yunchang from [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention).
Please note that it has a dependency on flash attention and specific GPU model requirements. We recommend installing yunchang from the source code rather than using `pip install yunchang==0.2.0`.

2. Install xDiT

```
python setup.py install
```

3. Usage

We provide examples demonstrating how to run models with PipeFusion in the [./examples/](./examples/) directory. To inspect the available options for the PixArt-alpha example, use the following command:

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
  --seed SEED           Random seed for operations.
  --output_type OUTPUT_TYPE
                        Output type of the pipeline.

Parallel Processing Options:
  --do_classifier_free_guidance
  --use_split_batch     Use split batch in classifier_free_guidance. cfg_degree will be 2 if set
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

Hybriding multiple parallelism techniques togather is essential for efficiently scaling. 
It's important that the product of all parallel degrees matches the number of devices. 
For instance, you can combine CFG, PipeFusion, and sequence parallelism with the command below to generate an image of a cute dog through hybrid parallelism. 
Here ulysses_degree * pipefusion_parallel_degree * cfg_degree(use_split_batch) == number of devices == 8.


```bash
torchrun --nproc_per_node=8 \
examples/pixartalpha_example.py \
--model models/PixArt-XL-2-1024-MS \
--pipefusion_parallel_degree 2 \
--ulysses_degree 2 \
--num_inference_steps 20 \
--warmup_steps 0 \
--prompt "A small dog" \
--use_split_batch
```


⚠️ Applying PipeFusion requires setting `warmup_steps`, also required in DistriFusion, typically set to a small number compared with `num_inference_steps`.
The warmup step impacts the efficiency of PipeFusion as it cannot be executed in parallel, thus degrading to a serial execution. 
We observed that a warmup of 0 had no effect on the PixArt model.
Users can tune this value according to their specific tasks.


<h2 id="secrets">✨ The xDiT's Secret Weapons</h2>

The exceptional capabilities of xDiT stem from our innovative technologies.

<h3 id="PipeFusion">1. PipeFusion</h3>

[PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models](./docs/methods/pipefusion.md)

<h3 id="USP">2. USP: Unified Sequence Parallelism</h3>

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](./docs/methods/usp.md)

<h3 id="hybrid_parallel">3. Hybrid Parallel</h3>

[Hybrid Parallelism](./docs/methods/hybrid.md)

<h3 id="cfg_parallel">4. CFG Parallel</h3>

[CFG Parallel](./docs/methods/cfg_parallel.md)

<h3 id="parallel_vae">5. Parallel VAE</h3>

[Patch Parallel VAE](./docs/methods/parallel_vae.md)

<h2 id="dev-guide">📚  Develop Guide</h2>
TBD

<h2 id="history">🚧  History and Looking for Contributions</h2>

We conducted a major upgrade of this project in August 2024.

The latest APIs is located in the [xfuser/](./xfuser/) directory, supports hybrid parallelism. It offers clearer and more structured code but currently supports fewer models.

The legacy APIs is in the [legacy/](./legacy/) directory, limited to single parallelism. It supports a richer of parallel methods, including PipeFusion, Sequence Parallel, DistriFusion, and Tensor Parallel. CFG Parallel can be hybrid with PipeFusion but not with other parallel methods.

For models not yet supported by the latest APIs, you can run the examples in the [legacy/scripts/](./legacy/scripts/) directory. If you wish to develop new features on a model or require hybrid parallelism, stay tuned for further project updates. 

We also welcome developers to join and contribute more features and models to the project. Tell us which model you need in xDiT in [discussions](https://github.com/xdit-project/xDiT/discussions).

<h2 id="cite-us">📝 Cite Us</h2>

```
@article{wang2024pipefusion,
      title={PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models}, 
      author={Jiannan Wang and Jiarui Fang and Jinzhe Pan and Aoyu Li and PengCheng Yang},
      year={2024},
      eprint={2405.07719},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

