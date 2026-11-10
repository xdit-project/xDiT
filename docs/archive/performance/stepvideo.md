## Step-Video-T2V 30B Performance

[Chinese Version](./stepvideo_zh.md)

Historical benchmark report: the results below apply to the listed models, hardware, and measurement settings. Performance with current software versions or other configurations may differ. See the [runner reference](../../runner/runner.md) for current runner usage and its supported models.

### Evaluation Protocol

The benchmark was conducted using the open-source Step-Video-T2V 30B model to evaluate SP (Sequence Parallelism) and TP (Tensor Parallelism) performance. We applied ulysses_degree as sp_degree.

Implementation reference:
`https://github.com/stepfun-ai/Step-Video-T2V/tree/main#multi-gpu-parallel-deployment`

### Nvidia H20 Cluster (8×NVLink)

#### Parallel Strategy Comparison

| GPUs | Parallel Type | Configuration | Latency | Parallel Efficiency | Memory Usage |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | `TP1 SP1` | 213.60s | 1.00x | 92,170M |
| 2 | TP | `TP2` | 108.97s | 0.98x | 57,458M ▼37.7% |
| 2 | SP | `SP2` | 108.13s | 0.99x | 86,258M ▼6.4% |
| 4 | TP | `TP4` | 57.61s | 0.93x | 36,566M ▼60.3% |
| 4 | SP | `SP4` | 57.01s | 0.94x | 78,226M ▼15.1% |
| 8 | TP | `TP8` | 30.40s | 0.88x | 30,028M ▼67.4% |
| 8 | SP | `SP8` | 30.10s | 0.89x | 79,684M ▼13.5% |

Parallel efficiency is `single-GPU latency / (GPU count × parallel latency)`. The values above express the fraction of ideal linear speedup, rounded to two decimals.

#### Results supported by the table

- Both TP and SP reduce latency as the GPU count increases, with parallel efficiency between 0.88 and 0.99 for the multi-GPU measurements.
- At eight GPUs, TP takes 30.40 seconds and SP takes 30.10 seconds for this benchmark.
- TP8 reduces recorded memory usage by 67.4% relative to the single-GPU baseline; SP8 reduces it by 13.5%. The difference is 53.9 percentage points.
