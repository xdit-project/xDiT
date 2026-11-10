# Installation and containers

Use Python 3.10 or newer and a PyTorch build matching your device: CUDA, ROCm,
or Ascend. Some [registered models](models.md) require newer Diffusers and Transformers
versions than the package minimum. Use an environment or container that supports
your selected checkpoint.

## Pip or source

Install the base package, then add the extras needed by your workload:

```bash
pip install xfuser
# Cache-DiT and torchao quantization, when needed:
pip install "xfuser[cache-dit,torchao]"
```

To use this checkout and its current model support:

```bash
git clone https://github.com/xdit-project/xDiT.git
cd xDiT
pip install -e ".[cache-dit,torchao]"
```

Combine extras in either install command. Base dependencies
include DistVAE, Yunchang, Diffusers, Transformers, Accelerate, and PEFT.
A published `xfuser` release can differ from this checkout. Inspect your
installed parser with `python -m xfuser.runner --help`.

| Extra | When to install it |
| --- | --- |
| `cache-dit` | DBCache, stage-aware PipeFusion caching, and model adapters using Cache-DiT. The current package requires `cache-dit>=1.5.0,<1.6.0`. |
| `torchao` | Supported torchao GEMM and text-encoder quantization paths. AITER has separate installation requirements. |
| `flash-attn` | FlashAttention 2 (`>=2.7.0`). Other FlashAttention generations need their corresponding package; Ring requirements depend on the selected backend. |
| `optimum-quanto` | FP8 T5 conversion in legacy pipeline examples; separate from the runner's GEMM/text-encoder controls. |
| `ray` | `RayDiffusionPipeline` workflows. |
| `opencv-python` | ConsisID examples; installs `opencv-python-headless`. |

Install the device runtime separately. Check
[attention backend](attention.md) and [quantization](quantization.md) requirements
before selecting optional kernels.
Follow the [HTTP service guide](../developer/Http_Service.md) to install
FastAPI, Ray, and Uvicorn.

## Device environments

For external setup guidance, see [platform and ecosystem guides](../resources.md).

### CUDA

Use a CUDA-compatible PyTorch build and attention kernels supported by your GPU.
For a container, build the
[CUDA Dockerfile](https://github.com/AMD-AGI/diffusion-models-inference/blob/main/docker/Dockerfile.cuda)
locally using its [build instructions](https://github.com/AMD-AGI/diffusion-models-inference/blob/main/docker/README.md).

### ROCm

Use a ROCm-compatible PyTorch build. AITER kernels have separate architecture and
package requirements; see [attention](attention.md) and [quantization](quantization.md).
The [container build guide](https://github.com/AMD-AGI/diffusion-models-inference/blob/main/docker/README.md)
covers Instinct and RDNA 4. Release images are available at
[rocm/pytorch-xdit](https://hub.docker.com/r/rocm/pytorch-xdit); choose a tag matching
your hardware and model. Check the tag's dependency versions and xDiT revision for supported features.

For adapted hardware configurations, see [hardware recipes](examples.md#hardware-recipes).
The linked YAML files configure a separate benchmark harness. The runner has no
`xdit --config` option.

### Ascend

Install a compatible Ascend runtime and PyTorch NPU environment. The Ring path
also needs `yunchang.ring.utils.update_npu_out`; one compatible helper revision is:

```bash
pip install "yunchang @ git+https://github.com/feifeibear/long-context-attention.git@56118e0"
```

Check that helper's signature when using another revision. See
[USP](../methods/usp.md) for backend and Ring constraints.
