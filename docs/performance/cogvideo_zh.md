## CogVideo 性能表现

CogVideo 是一个文本到视频的模型。xDiT 目前整合了 USP 技术（包括 Ulysses 注意力和 Ring 注意力）和 CFG 并行来提高推理速度，同时 PipeFusion 的工作正在进行中。由于 CogVideo 在视频生成尺寸上的限制，USP 的最大并行级别为 2。因此，xDiT 可以利用最多 4 个 GPU 来执行 CogVideo，尽管机器内可能有更多的 GPU。

在一台配备 L40（PCIe）GPU 的机器上，我们测试了使用不同 DiT 模型生成具有 30 帧、720px 宽和 480px 高的视频的推理延迟。

CogVideoX-2b 模型的结果显示在下图中。我们可以看到，随着并行度的增加，延迟有效减少。而且 xDiT 具有相较于 diffusers 软件包中的原始推理最多 3.1 倍的加速。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-2b.png" 
    alt="latency-cogvideo-l40-2b">
</div>

同样地，对于 CogVideoX-5b 模型，xDiT 实现了最多 3.9 倍的加速。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-5b.png" 
    alt="latency-cogvideo-l40-5b">
</div>