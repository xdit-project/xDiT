# Parallel Diffusion Model Inference Toolkit

The project provides a suite of efficient parallel inference approaches for Diffusion Models.
The backend networks of the diffusion model primarily include U-Net and Transformers (DiT). Both of these can be applied to DiT, and some methods can also be used for U-Net.

1. Tensor Parallelism. (DiT, U-Net)
2. Sequence Parallelism, Hybrid Sequence Parallelism including DeepSpeed-Ulysses, Ring-Attention: (DiT)
3. Displaced Patch Parallelism, named DistriFusion (DiT, U-Net)
4. Displaced Patch Pipeline Paralelism, named PipeFusion, first proposed in this repo. (DiT)

The communication and memory cost of the above parallelism for DiT is listed in the following table. (* indicates comm. can be hidden by computation, but needs extra buffers.)

<div align="center">

|          | attn-KV | communication cost | param | activations | extra buff |
|:--------:|:-------:|:-----------------:|:-----:|:-----------:|:----------:|
| Tensor Parallel | fresh | $4O(p \times hs)L$ | $\frac{1}{N}P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| DistriFusion* | stale | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $2AL = (KV)L$ |
| Ring Seq Parallel* | fresh | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| Ulysses Seq Parallel | fresh | $\frac{4}{N}O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| PipeFusion* | stale- | $2O(p \times hs)$ | $\frac{1}{N}P$ | $\frac{2}{M}A = \frac{1}{M}QO$ | $\frac{2L}{N}A = \frac{1}{N}(KV)L$ |



### Usage

1. install pipefuison from local.
```
python setup.py install
```

3. Usage Example
In [./scripts/pixart_example.py](./scripts/pixart_example.py), we provide a minimal script for running DiT with PipeFusion.

```python
import torch

from pipefuser.pipelines import DistriPixArtAlphaPipeline
from pipefuser.utils import DistriConfig
from pipefusion.modules.opt.chunk_conv2d import PatchConv2d

# parallelism choose from ["patch", "naive_patch", "pipefusion", "tensor", "sequence"],
distri_config = DistriConfig(
    parallelism="pipefusion",
    pp_num_patch=2
)

pipeline = DistriPixArtAlphaPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path=args.model_id,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

output = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
        num_inference_steps=20,
        output_type="pil,
    )
if distri_config.rank == 0:
    output.save("astronaut.png")
```

## Benchmark

You can  adapt to [./scripts/benchmark.sh](./scripts/benchmark.sh) to benchmark latency and memory usage of different parallel approaches.

## Evaluation Image Quality

To conduct the FID experiment, follow the detailed instructions provided in the assets/doc/FID.md documentation.

<div align="center">
    <img src="./assets/image_quality.png" alt="image_quality">
</div>

