## Flux.1性能表现

Flux.1是由Black Forest Labs推出的DiTs模型，它由Stable Diffusion的原班人马开发，包含三种变体：FLUX.1 [pro]、FLUX.1 [dev]和FLUX.1 [schnell]，均拥有12B参数。

Flux.1实时部署有如下挑战：

1. 高延迟：在单个A100上schnell 4 step采样生成2048px图片也需要10秒钟！这对dev和pro版本需要30~50 steps的情况延迟更加惊人。

2. VAE OOM：生成超过2048px的图片，在80GB VRAM的A100上VAE部分会出现OOM，即使DiTs主干有生成更高分辨图片分辨率能力，但是VAE已经不能承受图片之大了。


为了应对这些挑战，xDiT采用了混合序列并行[USP](https://arxiv.org/abs/2405.07719)、[PipeFusion](https://arxiv.org/abs/2405.14430)和[VAE并行](https://github.com/xdit-project/DistVAE)技术，以在多个GPU上扩展Flux.1的推理能力。
由于Flux.1不使用无分类器引导(Classifier-Free Guidance, CFG)，因此它与cfg并行不兼容。

### Flux.1 Dev的扩展性

我们使用FLUX.1-dev进行了性能基准测试,采用28个扩散步骤。

下表是4xH100上，使用不同的USP策略的延迟（Sec）。因为H100优异的NVLink带宽，使用USP比使用PipeFusion更恰当。torch.compile优化对H100至关重要，4xH100上获得了2.6倍加速。

<div align="center">

| Configuration | PyTorch (Sec) | torch.compile (Sec) |
|--------------|---------|---------|
| 1 GPU | 6.71 | 4.30 |
| Ulysses-2 | 4.38 | 2.68 |
| Ring-2 | 5.31 | 2.60 |
| Ulysses-2 x Ring-2 | 5.19 | 1.80 |
| Ulysses-4 | 4.24 | 1.63 |
| Ring-4 | 5.11 | 1.98 |

</div>

下图展示Flux.1-dev在4xH100上的延迟指标。xDiT成功在1.6秒内生成1024px图片！

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-1K-H100.png" 
    alt="scalability-flux_h100">
</div>


下图展示了Flux.1-dev在两个8xL40节点(总共16xL40 GPU)上的可扩展性。
虽然无法使用cfg并行,但我们仍然可以通过使用PipeFusion作为节点间并行方法来实现增强的扩展性。
对于1024px任务,16xL40上的混合并行比8xL40低1.16倍,其中最佳配置是ulysses=4和pipefusion=4。
对于4096px任务,混合并行在16个L40上仍然有益,比8个GPU低1.9倍,其中配置为ulysses=2, ring=2和pipefusion=4。
但在2048px任务中,16个GPU并未获得性能改进。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/scalability/Flux-16L40-crop.png" 
    alt="scalability-flux_l40">
</div>

下图展示了Flux.1在8xA100 GPU上的可扩展性。
对于1024px和2048px的图像生成任务,SP-Ulysses在单一并行方法中表现出最低的延迟。在这种情况下,最佳混合策略也是SP-Ulysses。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/scalability/Flux-A100-crop.png" 
    alt="scalability-flux_l40">
</div>

注意,上图所示的延迟尚未包括使用torch.compile,这将提供进一步的性能改进。

### Flux.1 Schnell的扩展性
我们使用FLUX.1 [schnell]进行了性能基准测试,采用4个扩散步骤。
由于扩散步骤非常少,我们不使用PipeFusion。

在8xA100 (80GB) NVLink互联的机器上，生成1024px图片，USP最佳策略是把所有并行度都给Ulysses，使用torch.compile之后的生成1024px图片仅需0.82秒！

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-1K-A100.png" 
    alt="latency-flux_a100_1k">
</div>


在8xA100 (80GB) NVLink互联的机器上，生成2048px图片，使用torch.compile之后的生成1024px图片仅需2.4秒！

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-2K-A100.png" 
    alt="latency-flux_a100_2k">
</div>

在PCIe Gen4互联的8xL40上，4卡规模xDiT也有很好的加速。生成1024px图片，使用ulysses_degree=2, ring_degree=2延迟低于单独使用ulysses或者ring，1.41秒可以生图。
8卡相比反而会变慢，这是因为这是需要经过QPI通信。我们预期使用PipeFusion会提升8卡的扩展性。

我们在1024px图片生成任务上，对比`torch.compile`和`onediff`的性能差别。
在1，8 GPU上，`torch.compile`略好，在2,4 GPU上`onediff`略好。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-1k-L40.png" 
    alt="latency-flux_l40_1k">
</div>


8xL40上生成2048px图片性能如下图。因为计算和通信比例增大，所以和024px任务不同，8卡相比4卡延迟更低，生图最快可达3.67秒。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/Flux-2k-L40.png" 
    alt="latency-flux_l40_2k">
</div>

### VAE并行

在A100上，单卡使用Flux.1超过2048px就会OOM。这是因为Activation内存需求增加，同时卷积算子引发memory spike，二者共同导致的。

使用通过Parallel VAE让xDiT得以一窥更高分辨率生成能力的真容，我们可以生成更高分辨率的图片。使用`--use_parallel_vae` 在[运行脚本中](../../examples/run.sh).

prompt是"A hyperrealistic portrait of a weathered sailor in his 60s, with deep-set blue eyes, a salt-and-pepper beard, and sun-weathered skin. He’s wearing a faded blue captain’s hat and a thick wool sweater. The background shows a misty harbor at dawn, with fishing boats barely visible in the distance."

2048px，3072px和4096px生成质量如下，可以看到4096px生成已经质量很低了。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/flux/flux_image.png" 
    alt="latency-flux_l40">
</div>

