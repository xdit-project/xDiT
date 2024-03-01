# DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models

### [Paper](https://drive.google.com/file/d/1mxcPaCYAZ0uTWue1HAqOuEHHlPQqGxzD/view?usp=sharing) | [Project](https://hanlab.mit.edu/projects/distrifusion) | [Blog](https://hanlab.mit.edu/blog/distrifusion)

**[NEW!]** DistriFusion is accepted by CVPR 2024! Our code is publicly available!

![teaser](https://github.com/mit-han-lab/distrifuser/blob/main/assets/teaser.jpg)
*We introduce DistriFusion, a training-free algorithm to harness multiple GPUs to accelerate diffusion model inference without sacrificing image quality. Naïve Patch (Overview (b)) suffers from the fragmentation issue due to the lack of patch interaction. The presented examples are generated with SDXL using a 50-step Euler sampler at 1280×1920 resolution, and latency is measured on A100 GPUs.*

DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models</br>
[Muyang Li](https://lmxyy.me/)\*, [Tianle Cai](https://www.tianle.website/)\*, [Jiaxin Cao](https://www.linkedin.com/in/jiaxin-cao-2166081b3/), [Qinsheng Zhang](https://qsh-zh.github.io), [Han Cai](https://han-cai.github.io), [Junjie Bai](https://www.linkedin.com/in/junjiebai/), [Yangqing Jia](https://daggerfs.com), [Ming-Yu Liu](https://mingyuliu.net), [Kai Li](https://www.cs.princeton.edu/~li/), and [Song Han](https://hanlab.mit.edu/songhan)</br>
MIT, Princeton, Lepton AI, and NVIDIA</br>
In CVPR 2024.

## Overview
![idea](https://github.com/mit-han-lab/distrifuser/blob/main/assets/idea.jpg)
**(a)** Original diffusion model running on a single device. **(b)** Naïvely splitting the image into 2 patches across 2 GPUs has an evident seam at the boundary due to the absence of interaction across patches. **(c)** Our DistriFusion employs synchronous communication for patch interaction at the first step. After that, we reuse the activations from the previous step via asynchronous communication. In this way, the communication overhead can be hidden into the computation pipeline.

## Performance
### Speedups

<p align="center">
  <img src="https://github.com/mit-han-lab/distrifuser/blob/main/assets/speedups.jpg" width="80%"/>
</p>Measured total latency of DistriFusion with SDXL using a 50-step DDIM sampler for generating a single image across on NVIDIA A100 GPUs. When scaling up the resolution, the GPU devices are better utilized. Remarkably, when generating 3840×3840 images, DistriFusion achieves 1.8×, 3.4× and 6.1× speedups with 2, 4, and 8 A100s, respectively.



### Quality

![quality](https://github.com/mit-han-lab/distrifuser/blob/main/assets/quality.jpg)
Qualitative results of SDXL. FID is computed against the ground-truth images. Our DistriFusion can reduce the latency according to the number of used devices while preserving visual fidelity.

References:

* Denoising Diffusion Implicit Model (DDIM), Song *et al.*, ICLR 2021
* Elucidating the Design Space of Diffusion-Based Generative Models, Karras *et al.*, NeurIPS 2022
* Parallel Sampling of Diffusion Models, Shih *et al.*, NeurIPS 2023
* SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, Podell *et al.*, ICLR 2024

## Prerequisites

* Python3
* NVIDIA GPU + CUDA >= 12.0 and corresponding CuDNN
* [PyTorch](https://pytorch.org) >= 2.2.

## Getting Started

### Installation

After installing [PyTorch](https://pytorch.org), you should be able to install `distrifuser` with PyPI

```shell
pip install distrifuser
```

or via GitHub:

```shell
pip install git+https://github.com/mit-han-lab/distrifuser.git
```

or locally for development

```shell
git clone git@github.com:mit-han-lab/distrifuser.git
cd distrifuser
pip install -e .
```

### Usage Example

In  [`scripts/sdxl_example.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/sdxl_example.py), we provide a minimal script for running [SDXL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) with DistriFusion. 

```python
import torch

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config, pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
)
pipeline.prepare()

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if distri_config.rank == 0:
    image.save("astronaut.png")
```

Specifically, our `distrifuser` shares the same APIs as [diffusers](https://github.com/huggingface/diffusers) and can be used in a similar way. You just need to define a `DistriFusion` and use our wrapped `DistriSDXLPipeline` to load the pretrained SDXL model. Then, we can generate the image like the  `StableDiffusionXLPipeline` in [diffusers](https://github.com/huggingface/diffusers). The running command is

```shell
torchrun --nproc_per_node=$N_GPUS scripts/sdxl_example.py
```

where `$N_GPUS` is the number GPUs you want to use.

### Benchmark

Our benchmark results are using [PyTorch](https://pytorch.org) 2.2 and [diffusers](https://github.com/huggingface/diffusers) 0.24.0. First, you may need to install some additional dependencies:

```shell
pip install git+https://github.com/zhijian-liu/torchprofile datasets torchmetrics dominate clean-fid
```

#### COCO Quality

You can use [`scripts/generate_coco.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/generate_coco.py) to generate images with COCO captions. The command is

```
torchrun --nproc_per_node=$N_GPUS scripts/generate_coco.py --no_split_batch
```

where `$N_GPUS` is the number GPUs you want to use. By default, the generated results will be stored in `results/coco`. You can also customize it with `--output_root`. Some additional arguments that you may want to tune:

* `--num_inference_steps`: The number of inference steps. We use 50 by default.
* `--guidance_scale`: The classifier-free guidance scale. We use 5 by default.
* `--scheduler`: The diffusion sampler. We use [DDIM sampler](https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/ddim#ddimscheduler) by default. You can also use `euler` for [Euler sampler](https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/euler#eulerdiscretescheduler) and `dpm-solver` for [DPM solver](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver).
* `--warmup_steps`: The number of additional warmup steps (4 by default). 
* `--sync_mode`: Different GroupNorm synchronization modes. By default, it is using our corrected asynchronous GroupNorm.
* `--parallelism`: The parallelism paradigm you use. By default, it is patch parallelism. You can use `tensor` for tensor parallelism and `naive_patch` for naïve patch.

After you generate all the images, you can use our script [`scripts/compute_metrics.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/compute_metrics.py) to calculate PSNR, LPIPS and FID. The usage is 

```shell
python scripts/compute_metrics.py --input_root0 $IMAGE_ROOT0 --input_root1 $IMAGE_ROOT1
```

where `$IMAGE_ROOT0` and `$IMAGE_ROOT1` are paths to the image folders you are trying to comparing. If `IMAGE_ROOT0` is the ground-truth foler, please add a `--is_gt` flag for resizing. We also provide a script [`scripts/dump_coco.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/dump_coco.py) to dump the ground-truth images.

#### Latency

You can use  [`scripts/run_sdxl.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/run_sdxl.py) to benchmark the latency our different methods. The command is

```shell
torchrun --nproc_per_node=$N_GPUS scripts/run_sdxl.py --mode benchmark --output_type latent
```

where `$N_GPUS` is the number GPUs you want to use. Similar to [`scripts/generate_coco.py`](https://github.com/mit-han-lab/distrifuser/blob/main/scripts/generate_coco.py), you can also change some arguments:

* `--num_inference_steps`: The number of inference steps. We use 50 by default.
* `--image_size`: The generated image size. By default, it is 1024×1024.
* `--no_split_batch`: Disable the batch splitting for classifier-free guidance.
* `--warmup_steps`: The number of additional warmup steps (4 by default). 
* `--sync_mode`: Different GroupNorm synchronization modes. By default, it is using our corrected asynchronous GroupNorm.
* `--parallelism`: The parallelism paradigm you use. By default, it is patch parallelism. You can use `tensor` for tensor parallelism and `naive_patch` for naïve patch.
* `--warmup_times`/`--test_times`: The number of warmup/test runs. By default, they are 5 and 20, respectively.


## Citation

If you use this code for your research, please cite our paper.

```bibtex
@inproceedings{li2023distrifusion,
  title={DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models},
  author={Li, Muyang and Cai, Tianle and Cao, Jiaxin and Zhang, Qinsheng and Cai, Han and Bai, Junjie and Jia, Yangqing and Liu, Ming-Yu and Li, Kai and Han, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Acknowledgments

Our code is developed based on [huggingface/diffusers](https://github.com/huggingface/diffusers) and [lmxyy/sige](https://github.com/lmxyy/sige). We thank [torchprofile](https://github.com/zhijian-liu/torchprofile) for MACs measurement, [clean-fid](https://github.com/GaParmar/clean-fid) for FID computation and [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) for PSNR and LPIPS.

We thank Jun-Yan Zhu and Ligeng Zhu for their helpful discussion and valuable feedback. The project is supported by MIT-IBM Watson AI Lab, Amazon, MIT Science Hub, and National Science Foundation.