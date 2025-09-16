## Running DiT Backbone and VAE Module Separately

The DiT model typically consists of DiT backbone (encoder + transformers) and VAE module.
The DiT backbone module has high computational requirements but stable memory usage.
For high-resolution images, the VAE module has high memory consumption due to temporary memory spikes from convolution operators, despite its low computational requirements. This often leads to OOM (Out of Memory) issues caused by the VAE module.

Therefore, separating the encoder + DiT backbone from the VAE module can effectively alleviate OOM issues.
We use Ray to implement the separation of backbone and VAE functionality, and allocate different GPU parallelism for VAE and DiT backbone.

In `ray_run.sh`, we define different model configurations.
For example, if we use 3 GPUs and want to allocate 1 GPU for VAE and 2 GPUs for DiT backbone, the settings in `ray_run.sh` would be:

```
N_GPUS=3 # world size
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1"
VAE_PARALLEL_SIZE=1
DIT_PARALLEL_SIZE=2
```

Here, `VAE_PARALLEL_SIZE` specifies the parallelism for VAE, DIT_PARALLEL_SIZE defines DiT parallelism, and PARALLEL_ARGS contains the parallel configuration for DiT backbone, which in this case uses PipeFusion to run on 2 GPUs.


