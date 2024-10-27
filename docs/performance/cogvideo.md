## CogVideo Performance
[Chinese Version](./cogvideo_zh.md)

<<<<<<< HEAD
CogVideo is a model that converts text to video. xDiT currently integrates USP technology (including Ulysses Attention and Ring Attention) and CFG parallel processing to improve inference speed, while work on PipeFusion is ongoing. We conducted a thorough analysis of the performance differences between a single GPU CogVideoX inference based on the diffusers library and our proposed parallel version when generating a 49-frame (6-second) 720x480 resolution video. We can combine different parallel methods arbitrarily to achieve varying performance. In this paper, we systematically tested the acceleration performance of xDiT on 1-12 L40 (PCIe) GPUs.

As shown in the figures, for the base model CogVideoX-2b, significant reductions in inference latency were observed whether using Ulysses Attention, Ring Attention, or Classifier-Free Guidance (CFG) parallel processing. It is noteworthy that due to its lower communication overhead, the CFG parallel method outperforms the other two technologies in terms of performance. By combining sequential parallelism and CFG parallelism, we successfully increased inference efficiency. With increasing parallelism, the inference latency continues to decrease. In the optimal configuration, xDiT achieves a 4.29x acceleration relative to single GPU inference, reducing the time for each iteration to just 0.49 seconds. Given the default 50 iterations of CogVideoX, the end-to-end generation of a 24.5-second video can be completed in a total of 30 seconds.

=======
CogVideo functions as a text-to-video model. xDiT presently integrates USP techniques (including Ulysses attention and Ring attention) and CFG parallelism to enhance inference speed, while work on PipeFusion is ongoing. Due to constraints in video generation dimensions in CogVideo, the maximum parallelism level for USP is 2. Thus, xDiT can leverage up to 4 GPUs to execute CogVideo, despite the potential for additional GPUs within the machine.

In a system equipped with L40 (PCIe) GPUs, we compared the inference performance of single-GPU CogVideoX utilizing the `diffusers` library with our parallelized versions for generating 49-frame (6-second) 720x480 videos.

As depicted in the figure, across the baseline model CogVideoX-2b, inference latency reductions were observed when employing Ulysses Attention, Ring Attention, or CFG parallelism. Notably, CFG parallelism demonstrated superior performance due to its lower communication overhead. By combining sequence parallelism with CFG parallelism, we further enhanced inference efficiency. As the degree of parallelism increased, the latency consistently decreased. Under optimal settings, xDiT achieved a 3.53x speedup over single-GPU inference, reducing each iteration to 0.6 seconds. Given CogVideoX's default 50 iterations, a 6-second video can be generated end-to-end within 30 seconds. 
>>>>>>> main

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-2b.png" 
    alt="latency-cogvideo-l40-2b">
</div>

<<<<<<< HEAD
For the more complex CogVideoX-5b model, although increasing parameters to enhance video quality and visual effects leads to a significant rise in computational costs, all methods maintain similar performance trends to CogVideoX-2b on this model, with further improvements in the acceleration effects of the parallel versions. Compared to the single GPU version, xDiT achieves up to a 7.75x increase in inference speed, reducing the end-to-end video generation time to around 40 seconds.
=======
For the more complex CogVideoX-5b model, which incorporates additional parameters for improved video quality and visual effects, albeit with increased computational costs, similar performance trends were maintained. However, the acceleration ratio of the parallel versions was further enhanced. In comparison to the single-GPU version, xDiT attained a speedup of up to 3.91x, enabling end-to-end video generation in just over 80 seconds.
>>>>>>> main

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-l40-5b.png" 
    alt="latency-cogvideo-l40-5b">
</div>

<<<<<<< HEAD
On systems equipped with A100 GPUs, xDiT demonstrates similar acceleration effects on CogVideoX-2b and CogVideoX-5b, as shown in the two figures below.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-a100-2b.png" 
    alt="latency-cogvideo-a100-5b">
</div>
<div align="center">
=======
Similarly, on systems equipped with A100 devices, xDiT exhibited comparable acceleration ratios.

<div align="center">
>>>>>>> main
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/performance/cogvideo/cogvideo-a100-5b.png" 
    alt="latency-cogvideo-a100-5b">
</div>