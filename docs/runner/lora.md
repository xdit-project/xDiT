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

This initial support is specific to the FLUX.1-dev runner. Quantized loading,
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
