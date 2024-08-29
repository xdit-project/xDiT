## HunyuanDiT性能

在8xA100（NVLink）机器上，在使用不同GPU数目时，最佳的并行方案都是不同的。这说明了多种并行和混合并行的重要性。
最佳的并行策略在不同GPU规模时分别是：在2个GPU上，使用ulysses_degree=2；在4个GPU上，使用ulysses_degree=2, cfg_parallel=2；在8个GPU上，使用pipefusion_parallel=8。


<div align="center">
    <img src="../../assets/performance/hunuyuandit/A100-HunyuanDiT.png" 
    alt="latency-hunyuandit_a100">
</div>

在8xL40 (PCIe)上的延迟情况如下图所示。同样，不同GPU规模，最佳并行策略都是不同的。

<div align="center">
    <img src="../../assets/performance/hunuyuandit/L40-HunyuanDiT.png" 
    alt="latency-hunyuandit_l40">
</div>

在A100和L40上，使用torch.compile会带来计算性能的显著提升。

在8xV100上的加速下如下图所示。

<div align="center">
    <img src="../../assets/performance/hunuyuandit/V100-HunyuanDiT.png" 
    alt="latency-hunyuandit_v100">
</div>

在4xT4上的加速下如下图所示。

<div align="center">
    <img src="../../assets/performance/hunuyuandit/T4-HunyuanDiT.png" 
    alt="latency-hunyuandit_t4">
</div>

⚠️ 我们还没有在V100和T4上测试torch.compile的效果。