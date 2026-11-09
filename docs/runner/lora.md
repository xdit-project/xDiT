# Startup LoRA fusion for FLUX.1-dev

The model runner accepts one LoRA repository or local file for FLUX.1-dev:

```bash
xdit --model black-forest-labs/FLUX.1-dev \
  --lora_path alvdansen/frosting_lane_flux \
  --lora_weight_name flux_dev_frostinglane_araminta_k.safetensors \
  --lora_scale 0.75 \
  --prompt "A small robot, frstingln illustration"
```

The runner uses Diffusers/PEFT to load and safely fuse the adapter into the base
weights before device placement, compilation, or caching. It then removes the
adapter modules, so generation has no separate LoRA computation. This changes
the in-memory weights only; the downloaded base checkpoint is not modified.

This support is specific to the FLUX.1-dev runner and to
[MiniMax-H3 Turbo](#minimax-h3-turbo). For FLUX.1-dev, quantized loading,
FSDP, meta loading, and PipeFusion are rejected before loading weights. Startup
fusion does not provide runtime adapter switching or unloading; restart the
runner to change the adapter. The legacy pipeline example CLI does not expose
these options.

`--lora_weight_name` is optional when Diffusers can select the weight file.
`--lora_scale` defaults to 1.0 and must be finite. A scale of zero preserves the
base weights. LoRA weights must be compatible with the selected base model.

The offline CPU integration test uses a locally saved tiny Flux transformer and
real PEFT fusion to check the weight delta and adapter removal. Run the opt-in
full-model test with access to the official gated Flux weights:

```bash
XDIT_FLUX_LORA_E2E=1 python -m pytest tests/e2e/test_flux_startup_lora.py -v
```

It checks that a real LoRA changes the expected projection weights and generates
a 256×256 image with two denoising steps. This is a functional smoke test, not
an image-quality evaluation. Multi-GPU correctness needs separate validation.

## MiniMax-H3 Turbo

`MiniMax-H3-Turbo` runs the base MiniMax-H3 checkpoint with a distilled
step-reduction LoRA from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
fused into the transformer at startup. Without `--lora_path` it downloads the
FL2VA Turbo 8-step v1.0 768p adapter and samples 8 steps with video shift 6, the
settings LightX2V Studio serves:

```bash
xdit --model MiniMax-H3-Turbo --task i2va \
  --ulysses_degree 8 --gemm_quantization fp8 \
  --input_images first_frame.png --prompt "..."
```

`--lora_path` selects another adapter: a safetensors file, a directory, or a
repository, with `--lora_weight_name` naming the file in the latter two.
`--lora_scale` multiplies the merged delta. The adapter's `alpha` comes from
the file's metadata (8 when it is missing, as in v0.1). For the adapters in
the [model specs](https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs),
`--num_inference_steps` and `--flow_shift` default to the published values and
other step counts are refused:

| Adapter (`--lora_weight_name`) | Steps | Video shift |
|--------------------------------|-------|-------------|
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` (default) | 8 | 6 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | 4 | 6 |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | 8 (or 4) | 12 |
| `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | 4 | 12 |

The 544p adapters are trained at about 0.5 MP (for example `--height 544
--width 960`), not at the 768×1344 default. Any other adapter, including later releases in the repository, needs
`--num_inference_steps` and `--flow_shift`. The audio shift stays at the
checkpoint's 3.0 for all of them.

Each targeted Linear gets `W += (alpha / rank) * scale * (B @ A)`, which matches
PEFT. The merge runs before the QKV projections are fused and before FP8
conversion, so `MiniMax-H3-Turbo` accepts quantized GEMMs; FSDP and
PipeFusion are still rejected. Startup fails if any adapter tensor has no
matching module, misses its A or B half, or has the wrong shape. Ref2VA and
ComfyUI-format adapters are refused, and the base `MiniMax-H3` and FastH3
runners do not accept `--lora_path`.
