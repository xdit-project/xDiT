## ConsisID Performance Report

[ConsisID](https://github.com/PKU-YuanGroup/ConsisID) is an identity-preserving text-to-video generation model that keeps the face consistent in the generated video by frequency decomposition.xDiT currently integrates USP techniques, including Ulysses Attention, Ring Attention, and CFG parallelization, to enhance inference speed, while work on PipeFusion is ongoing. We conducted an in-depth analysis comparing single-GPU ConsisID inference, based on the diffusers library, with our proposed parallelized version for generating 49 frames (6 seconds) of 720x480 resolution video. By flexibly combining different parallelization methods, we achieved varying performance outcomes. In this study, we systematically evaluate xDiT's acceleration performance across 1 to 6 Nvidia H100 GPUs.

As shown in the table, the ConsisID model achieves a significant reduction in inference latency with Ulysses Attention, Ring Attention, or Classifier-Free Guidance (CFG) parallelization. Notably, CFG parallelization outperforms the other two techniques due to its lower communication overhead. By combining sequence parallelization and CFG parallelization, inference efficiency was further improved. With increased parallelism, inference latency continued to decrease. Under the optimal configuration, xDiT achieved a 3.21× speedup over single-GPU inference, reducing iteration time to just 0.72 seconds. For the default 50 iterations of ConsisID, this enables end-to-end generation of 49 frames in 35 seconds, with a GPU memory usage of 40 GB.

### 720x480 Resolution (49 frames, 50 steps)


| N-GPUs | Ulysses Degree | Ring Degree | Cfg Parallel |  Times  |
| :----: | :------------: | :---------: | :----------: | :-----: |
|   6    |       2        |      3      |      1       | 44.89s  |
|   6    |       3        |      2      |      1       | 44.24s  |
|   6    |       1        |      3      |      2       | 35.78s  |
|   6    |       3        |      1      |      2       | 38.35s  |
|   4    |       2        |      1      |      2       | 41.37s  |
|   4    |       1        |      2      |      2       | 40.68s  |
|   3    |       3        |      1      |      1       | 53.57s  |
|   3    |       1        |      3      |      1       | 55.51s  |
|   2    |       1        |      2      |      1       | 70.19s  |
|   2    |       2        |      1      |      1       | 76.56s  |
|   2    |       1        |      1      |      2       | 59.72s  |
|   1    |       1        |      1      |      1       | 114.87s |

## Resources

Learn more about ConsisID with the following resources.
- A [video](https://www.youtube.com/watch?v=PhlgC-bI5SQ) demonstrating ConsisID's main features.
- The research paper, [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440) for more details.
