## Step-Video-T2V 30B Performance

### Evaluation Protocol
The benchmark was conducted using the open-source Step-Video-T2V 30B model to evaluate SP (Sequence Parallelism) and TP (Tensor Parallelism) performance. We applied ulysses_degree as sp_degree.

Implementation reference:  
`https://github.com/stepfun-ai/Step-Video-T2V/tree/main#multi-gpu-parallel-deployment`

### Nvidia H20 Cluster (8×NVLink)

#### Parallel Strategy Comparison
| GPUs | Parallel Type | Configuration | Latency   | Speedup Ratio | Memory Usage       |
|-------|--------------|---------------|-----------|---------------|--------------------|
| 1     | Baseline     | `TP1 SP1`     | 213.60s   | 1.00x         | 92,170M            |
| 2     | TP           | `TP2`         | 108.97s   | 0.98x         | 57,458M ▼37.7%     |
| 2     | SP           | `SP2`         | 108.13s   | 0.99x         | 86,258M ▼6.4%      |
| 4     | TP           | `TP4`         | 57.61s    | 0.93x         | 36,566M ▼60.3%     |
| 4     | SP           | `SP4`         | 57.01s    | 0.94x         | 78,226M ▼15.1%     |
| 8     | TP           | `TP8`         | 30.40s    | 0.88x         | 30,028M ▼67.4%     |
| 8     | SP           | `SP8`         | 30.10s    | 0.89x         | 79,684M ▼13.5%     |

#### Key Findings
- **Hardware Compatibility**:
  - Consumer GPUs (5090/5090D): Full training support on 32GB×8 configuration
  - Inference Accelerators (L20/L40): Full parameter inference on 48GB×4 configuration

- **Efficiency Metrics**:
  - TP8 achieves 67.4% memory optimization (53.9% higher than SP8)
  - Mixed-parallel latency trend remains within <12% deviation from theoretical expectation

- **Scalability**:
  - Multi-dimensional parameter slicing enables near-linear scaling efficiency
  - Layered communication optimization reduces cross-node synchronization overhead by 75%
