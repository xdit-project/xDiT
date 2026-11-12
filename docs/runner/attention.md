# Attention backends and schedules

Start with the model's default backend. If you omit the flag, the model's declared
preference takes precedence. Without a model preference, xDiT selects by platform,
installed-package priority, and availability. Selection does not benchmark kernels.

Set `--attention_backend NAME` to override the main kernel. On a supported model,
use `--cross_attention_backend NAME` to select a separate cross-attention kernel.
Names are case-insensitive. Availability also depends on the model, head dimensions,
causal behavior, and [Ring configuration](../methods/usp.md).
Cross-attention overrides require model support; otherwise cross-attention follows
the main backend.

## Backend reference

These are all names in the current backend enum, grouped by kernel family.
**Ring support** below also requires a compatible model and installed kernel.
The [backend specifications](https://github.com/xdit-project/xDiT/tree/main/xfuser/core/attention/backends)
define exact symbol, device, and call-shape checks.
SSTA means Sparse Spatio-Temporal Attention. MHA means multi-head attention.

| Family | CLI names | Key requirements and Ring support |
| --- | --- | --- |
| PyTorch dispatcher / math | `sdpa`, `sdpa_math` | PyTorch; no Ring. |
| PyTorch kernels | `sdpa_flash`, `sdpa_efficient`, `cudnn` | Device/build-dependent kernels; Ring supported. `cudnn` requires CUDA. |
| FlashAttention 2 / 3 | `flash`, `flash_3`, `flash_3_fp8` | Corresponding [FlashAttention](https://github.com/Dao-AILab/flash-attention) package; Ring supported. Version 3 requires CUDA. |
| FlashAttention 4 | `flash_4`, `flash_4_fp4` | CUDA and FlashAttention 4; no Ring. FP4 requires compute capability 10.0+. |
| Transformer Engine | `nvte_fp8` | CUDA and [Transformer Engine](https://github.com/NVIDIA/TransformerEngine); no Ring. |
| SageAttention | `sage` | CUDA and [SageAttention](https://github.com/thu-ml/SageAttention); Ring supported. |
| AITER dense | `aiter` | AITER on `gfx942`, `gfx950`, `gfx1200`, `gfx1201`, or `gfx1250`; Ring supported. |
| AITER per-tensor FP8 | `aiter_fp8` | AITER FP8 and Hadamard APIs; no Ring. Accepts eligible pre-quantized communication inputs. |
| AITER FlyDSL | `aiter_flydsl`, `aiter_flydsl_fp8` | AITER FlyDSL on `gfx1201`; no Ring. FP8 additionally needs the quantization API. |
| AITER MHA v4 | `aiter_bf16`, `aiter_bf16fp8`, `aiter_i8fp8`, `aiter_mxfp8`, `aiter_f8f6`, `aiter_mxfp6`, `aiter_f6f4`, `aiter_mxfp4`, `aiter_f4f4` | AITER MHA v4 on `gfx942`/`gfx950`; Ring requires `gfx950` and its packed-LSE API. |
| AITER MHA v4 FP6 | `aiter_f6f6` | Same MHA v4 requirements, limited to `gfx950`. |
| AITER Sage dense | `aiter_sage`, `aiter_sage_v2` | Corresponding AITER Sage API; Ring supported. Version 2 needs Hadamard rotation and a matching head width. |
| AITER Sage sparse | `aiter_sparse_sage`, `aiter_sparse_sage_v2`, `aiter_sparge`, `aiter_sparge_v2` | Model SSTA or Sparge support and AITER mask APIs; no Ring. |
| AITER MHA v4 Sparge | `aiter_bf16_sparge`, `aiter_bf16fp8_sparge`, `aiter_i8fp8_sparge`, `aiter_fp8_sparge` | Model Sparge support and AITER MHA v4 block-mask API on `gfx942`/`gfx950`; no Ring. |
| AITER MHA v4 MX Sparge | `aiter_mxfp8_sparge`, `aiter_f8f6_sparge`, `aiter_mxfp6_sparge`, `aiter_f6f4_sparge`, `aiter_mxfp4_sparge`, `aiter_f4f4_sparge` | Same Sparge requirements, limited to `gfx950`; no Ring. |
| Flex block attention | `flex_block_attn`, `flex_block_sparge` | `flex_block_attn` package and model SSTA/Sparge support; no Ring. |
| Sol-Attn | `sol_attn` | Optional Sol-Attn on CUDA compute capability 8.0+; supported BF16 shapes use Sol-Attn, others fall back to SDPA. No Ring. |
| AITER VSA | `aiter_vsa` | AITER Jenga kernels and compatible model/spatial layout; no Ring. |
| FastH3 VSA | `flex_vsa_h3`, `triton_vsa_h3` | Compatible FastH3 model; Triton path can fall back to FlexAttention. No Ring. |
| Prism block-sparse | `triton_bsa` | Compatible model token grid and Triton; no Ring. |
| Ascend | `npu` | Ascend runtime and PyTorch NPU fused-attention API; Ring supported. |

### Model and shape restrictions

Check model support as well as kernel availability. The runner validates
capabilities, allowed backends, attention head widths, and Ulysses head layouts.
[Model tables](models.md) identify declared features; model source links show
additional restrictions. Sparse backends require the corresponding model support
and spatial metadata. Packed variable-length keys, causality, dropout, and
cross-attention further restrict individual kernels.

MHA v4's fast path uses non-causal attention with head dimension 128. Unsupported
calls can fall back to `aiter`. MHA v4 Sparge variants require head dimension 128,
self-attention, matching query/KV head counts, and no packed keys. These conditions mean a
selected backend need not execute the same kernel for every attention call.

### FlyDSL FP8 behavior

`aiter_flydsl_fp8` is an active RDNA 4 backend, separate from BF16
`aiter_flydsl`. It chooses FP8 only for BF16 input shapes whose head width,
head count, and sequence length meet the measured crossover table. Other
supported shapes use BF16. Incompatible layouts fall back to PyTorch flash SDPA.
The unfused FP8 conversion creates an extra Q/K/V copy and can increase peak
memory. Short conditioning cross-attention normally remains BF16.

With [FP8 communication](communication.md), eligible pre-quantized Q/K/V bypass
that local conversion decision. Packed variable-length keys are unsupported by
FlyDSL, and Ring remains unsupported. Selecting an FP8 backend alone does not
quantize model weights or Ulysses communication.

See [installation](installation.md) for package extras,
[communication precision](communication.md) for FP8 exchanges and Attention A2A,
and the [backend enum](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/attention/spec.py)
for source names.

## Hybrid attention precision

On supported models, `--use_hybrid_attn_schedule` switches kernels by denoising
step. The simple schedule uses a high-precision kernel at both endpoints and a
low-precision kernel in the middle:

```bash
xdit --model Wan2.2-T2V --prompt "Clouds over a mountain lake" \
    --ulysses_degree 8 --num_inference_steps 40 \
    --use_hybrid_attn_schedule \
    --hybrid_attn_low_precision_backend aiter_fp8 \
    --hybrid_attn_high_precision_backend aiter \
    --num_hybrid_attn_high_precision_steps 2
```

This uses high precision for the first two and last two steps and requires a
compatible AITER environment. Set the endpoint count explicitly; it must fit
within the total step count. Compare quality and timing against the default.

For an explicit sequence, use `--hybrid_attn_schedule` with one comma-separated
backend name per transformer forward. Wan with sequential CFG uses two forwards
per denoising step, so its list must be twice as long. CFG parallelism removes
that multiplier. Set `--use_hybrid_attn_schedule` and omit the low/high pair.
The current loader also requires a resolved endpoint count for explicit lists;
set `--num_hybrid_attn_high_precision_steps 0` if the model has no default.
Neither schedule can also set `--attention_backend`.
All selected kernels must support the model and parallel layout. An explicit
cross-attention backend remains fixed; otherwise cross-attention follows the
active kernel.

Attention precision, [GEMM precision](quantization.md), and
[communication precision](communication.md) control different operations.
Each can affect generated output; test combinations on your workload.
