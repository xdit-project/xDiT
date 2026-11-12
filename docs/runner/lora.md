# Startup LoRA fusion for FLUX.1-dev

The model runner accepts one LoRA repository or local file for FLUX.1-dev:

```bash
xdit --model black-forest-labs/FLUX.1-dev \
  --lora_path alvdansen/frosting_lane_flux \
  --lora_weight_name flux_dev_frostinglane_araminta_k.safetensors \
  --lora_scale 0.75 \
  --prompt "A small robot, frstingln illustration"
```

The runner uses Diffusers/PEFT to load the adapter and safely merge it into the base
weights. This fusion runs before device placement, compilation, or caching.
The runner then removes the adapter modules, so generation has no separate LoRA
computation. Fusion changes weights in memory and preserves the downloaded base checkpoint.

This support is specific to the FLUX.1-dev runner and to
[MiniMax-H3 Turbo](#minimax-h3-turbo). For FLUX.1-dev, quantized loading,
FSDP, meta loading, and PipeFusion are rejected before loading weights. Restart
the runner to change or remove the adapter; startup fusion cannot switch adapters
during execution. The legacy pipeline example CLI does not expose
these options.

`--lora_weight_name` is optional when Diffusers can select the weight file.
`--lora_scale` defaults to 1.0 and must be finite. A scale of zero preserves the
base weights. LoRA weights must be compatible with the selected base model.

## MiniMax-H3 Turbo

`MiniMax-H3-Turbo` runs the base MiniMax-H3 checkpoint with a distilled
step-reduction LoRA from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
fused into the transformer at startup. Without `--lora_path` it downloads the
FL2VA Turbo 8-step v1.0 768p adapter and samples 8 steps with video shift 6, the
runner's default adapter settings:

```bash
xdit --model MiniMax-H3-Turbo --task i2va \
  --ulysses_degree 8 --gemm_quantization fp8 \
  --input_images first_frame.png --prompt "..."
```

`--lora_path` selects another adapter: a safetensors file, a directory, or a
repository, with `--lora_weight_name` naming the file in the latter two.
`--lora_scale` multiplies the weight update. The adapter scaling factor `alpha`
comes from the file's metadata. If it is missing, the runner uses 8, as in v0.1.
For the adapters in the [model specs](https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs),
`--num_inference_steps` and `--flow_shift` default to the published values and
other step counts are refused:

| Adapter (`--lora_weight_name`) | Steps | Video shift |
| --- | --- | --- |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` (default) | 8 | 6 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | 4 | 6 |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | 8 (or 4) | 12 |
| `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | 4 | 12 |

The 544p adapters were trained at about 0.5 megapixels. Set a matching resolution,
such as `--height 544 --width 960`, instead of the 768×1344 default.
For any adapter outside the table, set `--num_inference_steps` and `--flow_shift`.
This includes later releases in the repository. The audio shift stays at the
checkpoint's 3.0 for all of them.

Fusion runs before FP8 conversion, so `MiniMax-H3-Turbo` accepts supported
quantized GEMMs. Do not combine it with FSDP, memory-efficient meta loading, or PipeFusion.
Startup fails if an adapter tensor has no matching module, lacks its A or B half,
or has the wrong shape. The runner rejects Ref2VA and ComfyUI-format adapters.
The base `MiniMax-H3` and FastH3
runners do not accept `--lora_path`.
