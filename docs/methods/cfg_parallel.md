## Classifier-Free Guidance (CFG) Parallel 
[Chinese Version](./cfg_parallel_zh.md)

The Classifier-Free Guidance (CFG) has become an important trick  diffusion models by providing broader conditional control, reducing training burden, enhancing the quality and details of generated content, and improving the practicality and adaptability of the model.

For an input prompt, using CFG requires generating both unconditional guide and text guide simultaneously, which is equivalent to inputting input latents batch_size = 2 of DiT blocks. CFG Parallel separates the two latents for computation, and after each Diffusion Step forward is completed and before the Scheduler executes, it performs an Allgather operation on the latent space results. Its communication overhead is much smaller than Pipefusion and Sequence Parallel. Therefore, when using CFG, CFG Parallel must be used.