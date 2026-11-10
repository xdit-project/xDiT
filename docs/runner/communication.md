# Communication options

[Ulysses](../methods/usp.md) exchanges attention tokens and heads using all-to-all
collectives. The default path uses the selected device runtime and attention
backend, including supported CUDA, ROCm, and Ascend environments. Start with that
path and measure your workload before changing the transport.

| Control | What changes | Current requirements |
| --- | --- | --- |
| Default | Regular Ulysses collectives | A supported model, layout, and device backend |
| `--use_fp8_comms` | Quantizes eligible Q/K/V and attention-output exchanges | Model support and a pre-quantization attention backend; current kernels use ROCm/AITER |
| `--attention_a2a` | Replaces the input exchange with AITER's intranode transport | Supported Wan models, ROCm/AITER, and MORI SHMEM |

These controls affect communication tensors. [GEMM quantization](quantization.md)
controls linear-layer compute precision. [Attention backend selection](attention.md)
controls the attention kernel. Check the [model tables](models.md) before combining features.

## FP8 communication

Q, K, and V are the attention queries, keys, and values.
For a supported ROCm/AITER environment, this example uses four Ulysses ranks:

```bash
xdit --model FLUX.1-dev --prompt "A lighthouse in a winter storm" \
    --ulysses_degree 4 --attention_backend aiter_fp8 --use_fp8_comms
```

Requirements:

- Ulysses degree must exceed 1, and the model must declare FP8 communication support.
- At least one selected or scheduled backend must accept pre-quantized inputs.
  Current choices are `aiter_fp8` and `aiter_flydsl_fp8`; the latter requires
  `gfx1201`. Their Ring path is unsupported, so use `--ring_degree 1`.
  An FP8 attention backend that lacks this capability cannot enable FP8 comms.
- CPU offload and PipeFusion are rejected. Joint/replicated text can impose further
  model restrictions: FLUX.1-dev rejects FP8 comms when text cannot be split evenly.
- The required AITER APIs and supported tensor layouts must be available. Check
  the installed kernel requirements before enabling the flag.

The runner uses the generation inputs for an additional calibration run before
inference and compilation. It discards that output and keeps per-layer Q/K/V/output scales.
FP8 can reduce exchanged tensor sizes, but adds quantization work and can change
output. It does not guarantee lower latency. Compare the same prompts and seeds with regular communication.

| Scale option | Use |
| --- | --- |
| `--fp8_comms_scale` | Positive fixed scale; skips automatic per-layer calibration. Use only with a scale validated for your workload. |
| `--fp8_comms_safety_factor` | Calibration headroom in `(0, 1]`, default `0.85`. A smaller value increases the scale and leaves more room before saturation. |

## Attention A2A

Supported Wan2.2 T2V and Wan2.1/2.2 I2V adapters can use AITER's intranode
Attention A2A transport for the input Ulysses exchange. The output exchange uses
the regular collective. `--attention_a2a` and `--use_fp8_comms` are mutually
exclusive. Choose one transport optimization.

```bash
xdit --model Wan2.2-T2V --prompt "A motorcycle on a coastal highway" \
    --ulysses_degree 8 --attention_backend aiter_fp8 \
    --attention_a2a auto --use_torch_compile
```

This recipe requires `gfx950`. The transport requires compatible AITER and MORI
SHMEM, batch size 1, head dimension 128, and intranode Ulysses degree 2, 4, or 8.
Ring and Sparge head balancing are unsupported. **Auto and MX profiles require
`gfx950`**; `gfx942` can use supported explicit non-MX profiles such as
`--attention_a2a e4m3-e4m3` with `aiter_fp8`.

Auto selects a recipe from the dense attention backend:

| Backend | Profile |
| --- | --- |
| `aiter_fp8` | `e4m3-e4m3` |
| `aiter_i8fp8` | `int8-e4m3` |
| `aiter_mxfp8` | `mxfp8-e4m3` |
| `aiter_f8f6` | `e4m3-mxfp6` |
| `aiter_mxfp6` | `mxfp6-e4m3` |
| `aiter_f6f6` | `mxfp6-mxfp6` |
| `aiter_f6f4` | `mxfp6-mxfp4` |
| `aiter_mxfp4`, `aiter_f4f4` | `mxfp4-mxfp4` |

An explicit profile requires the matching attention backend from this table.
For hybrid attention, set `--attention_a2a auto`. Every scheduled backend must
have an A2A recipe. A BF16 endpoint such as `aiter` has no recipe, so the ordinary
BF16/FP8 hybrid example cannot also enable A2A. These transport requirements are
stricter than selecting the attention kernel alone.

`--attention_a2a_hadamard auto` uses the recipe's validated rotation placement.
Overrides are `preprocess`, `transport`, `epilogue`, and `none`.
The `int8-e4m3` profile rejects `transport` rotation; use `auto`/`none` for its
native recipe or `preprocess` for explicit rotation. The runner rejects unsupported
Torch/AITER interfaces. Set `--attention_a2a none` to use the default transport.

See the [CLI reference](cli.md#attention-backends-and-communication) for syntax,
[FP8 communication implementation](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/fp8_comms.py),
and [A2A validation](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/attention_a2a.py)
for current contracts.
