# GEMM and text-encoder quantization

GEMM means general matrix multiplication. Quantization reduces precision in
the linear layers declared by each model. Other components keep their normal
data type. Compare generated output against a run without quantization.

## Formats and hardware

Select `--gemm_quantization fp8`, `fp4`, `fp6`, `a6w4`, or `int8` on a model
that supports it. The older `--use_fp8_gemms`, `--use_fp4_gemms`, and
`--use_int8_gemms` flags are deprecated. The backend also checks the installed
packages and hardware:

| Format | Hardware and backend requirements |
| --- | --- |
| FP8 | AITER block-scale FP8 on RDNA 4 (`gfx1200`/`gfx1201`) when AITER is installed; otherwise torchao. CUDA FP8 requires compute capability 8.9 or newer. |
| FP4 | AITER MXFP4 on ROCm `gfx950`/`gfx1250`, or torchao NVFP4 on CUDA compute capability 10.0 or newer. RDNA 4 does not support this path. |
| FP6 / A6W4 | Supported Wan runners on ROCm `gfx950` with the corresponding AITER ASM kernels. A6W4 requires eager, GPU-resident loading. |
| INT8 | CUDA with torchao, currently declared by Z-Image and Z-Image-Turbo. ROCm is rejected. Small linear layers remain unquantized. |

To quantize a supported text encoder, set `--use_fp8_text_encoder` and a GEMM
profile containing FP8. For AITER streaming text-encoder FP8
on RDNA 4, use Transformers 5 or newer. Older Transformers uses post-load
conversion where supported; memory-efficient FSDP rejects that fallback.
Native torchao loading also depends on compatible Diffusers/Transformers APIs.
The loader reports its selected route and any fallback.

<a id="per-model-fp4-and-int8-targets"></a>

## Targets and schedules

The runner declares eligible transformer blocks and text encoders. Some FP4
models retain selected blocks or layers at higher precision for quality. On
dual-transformer Wan 2.2, the FP4 targets cover the first transformer and the
second stays at higher precision. See the selected model's `settings` for its
exact targets and overrides.

Profiles such as `--gemm_quantization low=fp4,high=fp8` select two precision formats.
On supported ROCm/AITER paths, `--use_hybrid_gemm_schedule` uses higher precision
at the start and end of denoising. CUDA does not support this per-step precision schedule.
Advanced target overrides and explicit schedules belong in
[`examples/gemm_config.yaml`](https://github.com/xdit-project/xDiT/blob/main/examples/gemm_config.yaml):

```bash
xdit --model Wan2.2-T2V --prompt "Clouds over a mountain lake" \
    --gemm_quantization low=fp4,high=fp6 \
    --gemm_config examples/gemm_config.yaml
```

This profile requires `gfx950` and the corresponding AITER kernels. The YAML
configures the selected profile; it does not select a precision by itself.

<a id="flag-combinations-and-exclusions"></a>
Explicit `--gemm_quantization` takes precedence over deprecated format selectors.
`--gemm_config` takes precedence over the deprecated prefix/suffix overrides;
an explicit YAML schedule cannot be combined with the simple schedule flags.
Do not mix INT8 with other formats. Advanced target overrides require a
profile with two precision formats.

## Examples

FP8 on RDNA 4, including the supported FLUX text encoder:

```bash
xdit --model FLUX.1-dev --prompt "A lighthouse in a winter storm" \
    --gemm_quantization fp8 --use_fp8_text_encoder
```

NVFP4 on CUDA Blackwell:

```bash
xdit --model FLUX.2-dev --prompt "A studio photograph of a glass sculpture" \
    --gemm_quantization fp4
```

Check [model capabilities](models.md) and installed kernels before using either
command. For loading routes and offload restrictions, see [memory controls](memory.md).
