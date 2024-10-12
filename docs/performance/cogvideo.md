## CogVideo Performance
[Chinese Version](./cogvideo_zh.md)

CogVideo functions as a text-to-video model. xDiT presently integrates USP techniques (including Ulysses attention and Ring attention) and CFG parallelism to enhance inference speed, while work on PipeFusion is ongoing. Due to constraints in video generation dimensions in CogVideo, the maximum parallelism level for USP is 2. Thus, xDiT can leverage up to 4 GPUs to execute CogVideo, despite the potential for additional GPUs within the machine.

In a system equipped with L40 (PCIe) GPUs, we compared the inference performance of single-GPU CogVideoX utilizing the `diffusers` library with our parallelized versions for generating 49-frame (6-second) 720x480 videos.

As depicted in the figure, across the baseline model CogVideoX-2b, inference latency reductions were observed when employing Ulysses Attention, Ring Attention, or CFG parallelism. Notably, CFG parallelism demonstrated superior performance due to its lower communication overhead. By combining sequence parallelism with CFG parallelism, we further enhanced inference efficiency. As the degree of parallelism increased, the latency consistently decreased. Under optimal settings, xDiT achieved a 3.53x speedup over single-GPU inference, reducing each iteration to 0.6 seconds. Given CogVideoX's default 50 iterations, a 6-second video can be generated end-to-end within 30 seconds. 

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-2b.png" 
    alt="latency-cogvideo-l40-2b">
</div>

For the more complex CogVideoX-5b model, which incorporates additional parameters for improved video quality and visual effects, albeit with increased computational costs, similar performance trends were maintained. However, the acceleration ratio of the parallel versions was further enhanced. In comparison to the single-GPU version, xDiT attained a speedup of up to 3.91x, enabling end-to-end video generation in just over 80 seconds.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-5b.png" 
    alt="latency-cogvideo-l40-5b">
</div>

Similarly, on systems equipped with A100 devices, xDiT exhibited comparable acceleration ratios.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-a100-5b.png" 
    alt="latency-cogvideo-a100-5b">
</div>