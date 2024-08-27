## HunyuanDiT Performance
[Chinese Version](./hunyuandit_zh.md)

On an 8xA100 (NVLink) machine, the optimal parallelization scheme varies with the number of GPUs used, highlighting the importance of diverse and hybrid parallelism. The best parallel strategies for different GPU scales are as follows: with 2 GPUs, use `ulysses_degree=2`; with 4 GPUs, use `ulysses_degree=2, cfg_parallel=2`; with 8 GPUs, use `pipefusion_parallel=8`.

<div align="center">
    <img src="../../assets/performance/hunuyuandit/A100-HunyuanDiT.png" 
    alt="latency-hunyuandit_a100">
</div>

The latency on 8xL40 (PCIe) is shown in the figure below. Similarly, the optimal parallel strategy differs for different GPU scales.

<div align="center">
    <img src="../../assets/performance/hunuyuandit/L40-HunyuanDiT.png" 
    alt="latency-hunyuandit_l40">
</div>

On both A100 and L40, using `torch.compile` significantly enhances computational performance.

The acceleration on 8xV100 is shown in the figure below.

<div align="center">
    <img src="../../assets/performance/hunuyuandit/V100-HunyuanDiT.png" 
    alt="latency-hunyuandit_v100">
</div>

The acceleration on 4xT4 is shown in the figure below.

<div align="center">
    <img src="../../assets/performance/hunuyuandit/T4-HunyuanDiT.png" 
    alt="latency-hunyuandit_t4">
</div>

⚠️ We have not tested `torch.compile` on T4 and V100.