# FLUX.1 + AMD Quark Native Inference Example

A self-contained xDiT example that runs FLUX.1 with **AMD Quark native
inference**. It quantizes the diffusion transformer and freezes it into real
low-precision kernels, so inference dispatches to the Aiter GEMM path instead of
simulating quantization on top of the high-precision GEMM.

Script: [`flux_quark_example.py`](./flux_quark_example.py)

## What "native inference" means

Plain fake-quantized inference keeps the original high-precision GEMM and only
*simulates* quantization (quantize → dequantize) around it. Native inference
instead converts each linear layer into a real low-precision kernel:

```
nn.Linear  ->  QuantLinear  ->  QParamsLinear  ->  Aiter native linear
```

This conversion happens during `freeze`:

```python
quantizer.freeze(model, runtime_options=RuntimeOptions(native_linear_mode="fp8_per_tensor"))
```

| Quantization mode | `native_linear_mode` | Format         |
|-------------------|----------------------|----------------|
| `fp8`             | `fp8_per_tensor`     | FP8 E4M3       |
| `mxfp4`           | `mxfp4`              | OCP MXFP4      |

Because the frozen modules are `QParamsLinear` subclasses holding real quantized
weights, the model can be exported straight to HuggingFace safetensors via
`--quark_export_dir` (no extra conversion step).

## Requirements

```bash
pip install -e .        # xDiT
pip install amd-quark   # AMD Quark
```

The Aiter native linear kernels are used at inference time. This path targets
AMD GPUs on ROCm; make sure Aiter is available in your environment.

## Usage

### Single GPU (FP8)

```bash
python examples/flux_quark_example.py \
    --model /path/to/FLUX.1-dev \
    --prompt "a cat holding a sign that says hello world" \
    --height 1024 --width 1024 \
    --num_inference_steps 28 \
    --quark_quantization_mode fp8
```

### Single GPU (MXFP4)

```bash
python examples/flux_quark_example.py \
    --model /path/to/FLUX.1-dev \
    --prompt "a cat" \
    --quark_quantization_mode mxfp4
```

### Multi-GPU with torchrun (Ulysses sequence parallel)

```bash
torchrun --nproc_per_node 2 examples/flux_quark_example.py \
    --model /path/to/FLUX.1-dev \
    --prompt "a cat" \
    --ulysses_degree 2 \
    --quark_quantization_mode fp8
```

### Export the frozen native model to safetensors

```bash
python examples/flux_quark_example.py \
    --model /path/to/FLUX.1-dev \
    --prompt "a cat" \
    --quark_quantization_mode fp8 \
    --quark_export_dir ./flux_fp8_native
```

## Arguments

Standard xDiT arguments are available (`--model`, `--prompt`, `--height`,
`--width`, `--num_inference_steps`, `--guidance_scale`, `--seed`,
`--ulysses_degree`, `--ring_degree`, `--pipefusion_parallel_degree`, etc.).

This example adds:

| Argument                    | Default | Description                                                     |
|-----------------------------|---------|-----------------------------------------------------------------|
| `--quark_quantization_mode` | `fp8`   | Native inference format: `fp8` or `mxfp4`.                      |
| `--quark_export_dir`        | `None`  | If set, export the frozen native model as HF safetensors here. |

## How it works

1. xDiT builds the `xFuserFluxPipeline` and moves it to the target GPU.
2. `quantize_transformer_native` inserts dynamic-quant `QuantLinear` layers
   (no calibration data required) and freezes the transformer with
   `RuntimeOptions(native_linear_mode=...)`, converting it to the Aiter native
   linear path.
3. The quantized transformer replaces `pipe.transformer` in place.
4. Inference and image saving proceed through the unmodified xDiT pipeline.

Output images are written to `./results/` with a filename that records the
quantization mode and parallel configuration. Peak parameter/activation memory
and epoch time are printed by the last rank.

## Notes

- Activations use **dynamic** quantization, so no calibration dataset is needed.
- FP8 typically has minimal quality impact; MXFP4 offers higher compression with
  a larger accuracy trade-off.
- xDiT is not modified — all Quark logic is contained in the example script.
