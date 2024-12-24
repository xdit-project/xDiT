## ConsisID Performance Report

[ConsisID](https://github.com/PKU-YuanGroup/ConsisID) 是一种身份保持的文本到视频生成模型，其通过频率分解在生成的视频中保持面部一致性。xDiT 目前整合了 USP 技术（包括 Ulysses 注意力和 Ring 注意力）和 CFG 并行来提高推理速度，同时 PipeFusion 的工作正在进行中。我们对基于 diffusers 库的单 GPU ConsisID 推理与我们提出的并行化版本在生成 49帧（6秒）720x480 分辨率视频时的性能差异进行了深入分析。由于我们可以任意组合不同的并行方式以获得不同的性能。在本文中，我们对xDiT在1-6张H100（Nvidia）GPU上的加速性能进行了系统测试。

如表所示，对于模型ConsisID，无论是采用 Ulysses Attention、Ring Attention 还是 Classifier-Free Guidance（CFG）并行，均观察到推理延迟的显著降低。值得注意的是，由于其较低的通信开销，CFG 并行方法在性能上优于其他两种技术。通过结合序列并行和 CFG 并行，我们成功提升了推理效率。随着并行度的增加，推理延迟持续下降。在最优配置下，xDiT 相对于单GPU推理实现了 3.21 倍的加速，使得每次迭代仅需 0.72 秒。鉴于 ConsisID 默认的 50 次迭代，总计 35 秒即可完成 49帧 视频的端到端生成，并且运行过程中占用GPU显存40G。

### 720x480 Resolution (49 frames, 50 steps)


| N-GPUs | ulysses_degree | ring_degree | cfg-parallel |   times   |
|:------:|:--------------:|:-----------:|:------------:|:---------:|
|   6    |       2        |      3      |      1       |   44.89s  |
|   6    |       3        |      2      |      1       |   44.24s  |
|   6    |       1        |      3      |      2       |   35.78s  |
|   6    |       3        |      1      |      2       |   38.35s  |
|   4    |       2        |      1      |      2       |   41.37s  |
|   4    |       1        |      2      |      2       |   40.68s  |
|   3    |       3        |      1      |      1       |   53.57s  |
|   3    |       1        |      3      |      1       |   55.51s  |
|   2    |       1        |      2      |      1       |   70.19s  |
|   2    |       2        |      1      |      1       |   76.56s  |
|   2    |       1        |      1      |      2       |   59.72s  |
|   1    |       1        |      1      |      1       |  114.87s  |
