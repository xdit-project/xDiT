## HunyuanVideo Performance Report

xDiT is [HunyuanVideo](https://github.com/Tencent/HunyuanVideo?tab=readme-ov-file#-parallel-inference-on-multiple-gpus-by-xdit)'s official parallel inference engine. On H100 and H20 GPUs, xDiT reduces the generation time of 1028x720 videos from 31 minutes to 5 minutes, and 960x960 videos from 28 minutes to 6 minutes.

### 1280x720 Resolution (129 frames, 50 steps) - Ulysses Latency (seconds)

<center>

| GPU Type | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|----------|--------|---------|---------|---------|
| H100 | 1,904.08 | 925.04 | 514.08 | 337.58 |
| H20 | 6,639.17 | 3,400.55 | 1,762.86 | 940.97 |
| L20 | 6,043.88 | 3,271.44 | 2,080.05 | |

</center>

### 960x960 Resolution (129 frames, 50 steps) - Ulysses Latency (seconds)

<center>

| GPU Type | 1 GPU | 2 GPUs | 3 GPUs | 6 GPUs |
|----------|--------|---------|---------|---------|
| H100 | 1,735.01 | 934.09 | 645.45 | 367.02 |
| H20 | 6,621.46 | 3,400.55 | 2,310.48 | 1,214.67 |
| L20 | 6,039.08 | 3,260.62 | 2,070.96 | |

</center>
