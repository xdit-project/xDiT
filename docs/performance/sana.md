## Performance of SANA
[Chinese Version](./sana_zh.md)

We use the open-source version of `Sana_1600M_4Kpx_BF16_diffusers` for performance evaluation.

Currently, xDiT supports acceleration for SANA with Pipefusion, Ulysses, Ring, CFG, and any combination thereof. Due to the limitation of the Head channel in the SANA network, the maximum parallelism supported by Ulysses is 2. We tested latency on an 8xA100 (NVLink) machine by generating 4096x4096 images with 20 steps. The measured latencies are shown in the table below. It can be seen that CFG achieves the best acceleration effect, while the other three acceleration strategies have similar performance. In the case of 8 GPUs, up to 4.4x generation acceleration can be achieved.

| #GPUs | cfg | ulysses | ring | pp | Latency (seconds) |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 17.551 |
| 2 | 1 | 1 | 1 | 2 | 11.276 |
| 2 | 1 | 1 | 2 | 1 | 11.447 |
| 2 | 1 | 2 | 1 | 1 | 10.175 |
| 2 | 2 | 1 | 1 | 1 | 8.365 |
| 4 | 2 | 1 | 1 | 2 | 5.599 |
| 4 | 2 | 1 | 2 | 1 | 5.702 |
| 4 | 2 | 2 | 1 | 1 | 5.803 |
| 8 | 2 | 1 | 1 | 4 | 4.050 |
| 8 | 2 | 1 | 2 | 2 | 4.091 |
| 8 | 2 | 1 | 4 | 1 | 4.003 |
| 8 | 2 | 2 | 1 | 2 | 4.201 |
| 8 | 2 | 2 | 2 | 1 | 3.991 |