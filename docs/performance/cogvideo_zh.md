## CogVideo 性能表现

CogVideo 是一个文本到视频的模型。xDiT 目前整合了 USP 技术（包括 Ulysses 注意力和 Ring 注意力）和 CFG 并行来提高推理速度，同时 PipeFusion 的工作正在进行中。我们对基于 `diffusers` 库的单 GPU CogVideoX 推理与我们提出的并行化版本在生成 49帧（6秒）720x480 分辨率视频时的性能差异进行了深入分析。由于我们可以任意组合不同的并行方式以获得不同的性能。在本文中，我们对xDiT在1-12张L40（PCIe）GPU上的加速性能进行了系统测试。

如图所示，对于基础模型 CogVideoX-2b，无论是采用 Ulysses Attention、Ring Attention 还是 Classifier-Free Guidance（CFG）并行，均观察到推理延迟的显著降低。值得注意的是，由于其较低的通信开销，CFG 并行方法在性能上优于其他两种技术。通过结合序列并行和 CFG 并行，我们成功提升了推理效率。随着并行度的增加，推理延迟持续下降。在最优配置下，xDiT 相对于单GPU推理实现了 4.29 倍的加速，使得每次迭代仅需 0.49 秒。鉴于 CogVideoX 默认的 50 次迭代，总计 30 秒即可完成 24.5 秒视频的端到端生成。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-2b.png" 
    alt="latency-cogvideo-l40-2b">
</div>

针对更复杂的CogVideoX-5b模型，虽然参数增加以提升视频质量和视觉效果，导致计算成本显著上升，但在该模型上，所有方法仍然保持与CogVideoX-2b相似的性能趋势，且并行版本的加速效果进一步提升。相较于单GPU版本，xDiT实现了高达7.75倍的推理速度提升，将端到端视频生成时间缩短至约40秒。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-5b.png" 
    alt="latency-cogvideo-l40-5b">
</div>

在搭载A100 GPU的系统中，xDiT 在 CogVideoX-2b 和 CogVideoX-5b 上展现出类似的加速效果，具体表现可见下方两图。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-a100-5b.png" 
    alt="latency-cogvideo-a100-2b">
</div>


<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-a100-5b.png" 
    alt="latency-cogvideo-a100-5b">
</div>