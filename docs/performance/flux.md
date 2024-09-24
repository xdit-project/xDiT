## Flux.1 Performance Overview
[Chinses Version](./flux_zh.md)

Flux.1, developed by Black Forest Labs and created by the original team behind Stable Diffusion, is a DiTs model featuring three variants: FLUX.1 [pro], FLUX.1 [dev], and FLUX.1 [schnell], all equipped with 12 billion parameters.

Deploying Flux.1 in real-time presents several challenges:

1. High Latency: Generating a 2048px image using the schnell variant with 4 sampling steps on a single A100 GPU takes approximately 10 seconds. This latency is significantly higher for the dev and pro versions, which require 30 to 50 steps.

2. VAE OOM: The VAE component experiences Out Of Memory (OOM) issues when attempting to generate images larger than 2048px on an A100 GPU with 80GB VRAM, despite the DiTs backbone's capability to handle higher resolutions.

To address these challenges, xDiT employs a hybrid sequence parallel [USP](https://arxiv.org/abs/2405.07719) and [VAE Parallel](https://github.com/xdit-project/DistVAE) to scale Flux.1 inference across multiple GPUs.

Currently, xDiT does not support PipeFusion for the Flux.1 schnell variant due to its minimal sampling steps, as PipeFusion requires a warmup phase which is not suitable for this scenario. However, applying PipeFusion for the Pro and Dev versions is considered necessary and is still under development.

Additionally, since Flux.1 does not utilize Classifier-Free Guidance (CFG), it is not compatible with cfg parallel.

### Scalability

We conducted performance benchmarking using FLUX.1 [schnell] with 4 steps.

On a machine with 8xA100 (80GB) GPUs interconnected via NVLink, generating a 1024px image, the optimal strategy with USP is to apply ulysses_degree=#gpu. After using `torch.compile`, the generation of a 1024px image takes only 0.82 seconds!

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-1K-A100.png" 
    alt="latency-flux_a100_1k">
</div>

On the same 8xA100 (80GB) NVLink-interconnected machine, generating a 2048px image, after using `torch.compile`, the generation of a 2048px image takes only 2.4 seconds!

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-2K-A100.png" 
    alt="latency-flux_a100_2k">
</div>

On a machine with 8xL40 GPUs interconnected via PCIe Gen4, even with a 4-card setup using xDiT, there is significant acceleration. Generating a 1024px image with `ulysses_degree=2` and `ring_degree=2` results in lower latency compared to using Ulysses or ring alone, with a generation time of 1.41 seconds. Using 8xL40 actually slows down due to the need for QPI communication. 
We anticipate that using PipeFusion will enhance the scalability of 8-card setups.

We compared the performance of `torch.compile` and `onediff` on 1024px image generation tasks. On 1 and 8 GPUs, `torch.compile` performs slightly better, while on 2 and 4 GPUs, onediff performs slightly better.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-1k-L40.png" 
    alt="latency-flux_l40_1k">
</div>

The performance of generating a 2048px image on 8xL40 GPUs is shown below. Due to the increased ratio of computation to communication, unlike the 1024px image generation tasks, using 8 GPUs results in lower latency compared to 4 cards, with the fastest image generation time reaching 3.67 seconds.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-2k-L40.png" 
    alt="latency-flux_l40_2k">
</div>

### Effect of VAE Parallel

On an A100 GPU, using Flux.1 on a single card for resolutions above 2048px leads to an Out Of Memory (OOM) error. 
This is due to the increased memory requirements for activations, along with memory spikes caused by convolution operators, both of which collectively contribute to the issue.

By leveraging Parallel VAE, xDiT is able to demonstrate its capability for generating images at higher resolutions, enabling us to produce images with even greater detail and clarity. Applying `--use_parallel_vae` in the [runing script](../../examples/run.sh).

prompt是"A hyperrealistic portrait of a weathered sailor in his 60s, with deep-set blue eyes, a salt-and-pepper beard, and sun-weathered skin. He’s wearing a faded blue captain’s hat and a thick wool sweater. The background shows a misty harbor at dawn, with fishing boats barely visible in the distance."

The quality of image generation at 2048px, 3072px, and 4096px resolutions is as follows. It is evident that the quality of the 4096px generated images is significantly lower.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/flux_image.png" 
    alt="latency-flux_l40">
</div>
