## Latte性能

Latte是文生视频模型，xDiT目前实现了USP方式对它进行并行推理加速。PipeFusion还在开发中。

在8xL20 (PCIe)的机器上，生成512x512x16视频的延迟表现如下图所示。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/latte/Latte-L20-512.png" 
    alt="latency-latte-l20-512">
</div>

生成1024x1024x16视频的延迟表现如下图所示，使用混合序列并行(`ulysses_degree`=2, `ring_degree=4`)可以获得最佳性能。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/latte/Latte-L20-1024.png" 
    alt="latency-latte-l20-1024">
</div>