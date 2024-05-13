# ParallelDiffusion: Parallel Diffusion Model Inference on Multiple Devices

The project provides a suite of efficient parallel inference methods for Diffusion models. 

1. Tensor Parallelism.
2. Sequence Parallelism, including Ulysses and Ring Attention.
2. Displaced Patch Parallelism from DistriFusion.
3. Displaced Pipeline Paralelism proposed by us


Known Issues

1. Dit VAE decode has CUDA memory spike issue, [diffusers/issues/5924](https://github.com/huggingface/diffusers/issues/5924). 
So we set output_type='latent' to avoid calling vae decode by default.