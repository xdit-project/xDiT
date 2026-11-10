# Model and hardware examples

For a first image-generation command, use the [quick start](runner.md).
These examples cover task-specific inputs and hardware settings. Check the
[model list](models.md) and [installation](installation.md) before adapting them.
For StepVideo and Mochi, see [external model integrations](#external-model-integrations).

## Text-to-video

```bash
xdit --model Wan2.2-T2V \
    --prompt "A cat playing with a ball" \
    --height 720 \
    --width 1280 \
    --num_frames 49 \
    --ulysses_degree 8
```

## Distilled image-to-video

```bash
xdit --model Wan2.2-Distilled-I2V \
    --distilled_transformer_path   /path/to/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors \
    --distilled_transformer_2_path /path/to/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors \
    --input_images /path/to/image.jpg \
    --prompt "A cat walking in a garden" \
    --ulysses_degree 8
```

## Image-to-video with audio (Prism)

Prism animates one reference image and generates a matching 48 kHz audio track.
`TRITON_BSA` runs Prism's block-sparse attention on the video self-attention; every
other attention call uses `--cross_attention_backend`.

```bash
torchrun --nproc_per_node=8 -m xfuser.runner \
    --model Prism \
    --input_images ./reference.png \
    --prompt "A woman speaks to the camera in a quiet room" \
    --audio_prompt "A woman speaks in a calm, low voice" \
    --ulysses_degree 8 \
    --attention_backend TRITON_BSA \
    --cross_attention_backend AITER
```

## Text-to-video with audio (VDN-H3)

VDN-H3 combines window softmax attention with Video-Delta linear attention.
First, prepare its transformer with the materialization script below. The script
merges two VDN LoRA adapters into the MiniMax-H3 base and adds the linear-branch
weights from `OpenVDN/vdn-minimax-h3`. The runner loads the remaining pipeline
components from `MiniMaxAI/MiniMax-H3`.

```bash
python scripts/vdn/materialize_transformer.py --output /path/to/vdn-h3

torchrun --nproc_per_node=8 -m xfuser.runner \
    --model OpenVDN/vdn-minimax-h3 \
    --vdn_transformer_dir /path/to/vdn-h3 \
    --task t2va \
    --prompt "A red fox trots through a snowy pine forest." \
    --num_frames 124 \
    --ulysses_degree 8 \
    --use_parallel_vae \
    --gemm_quantization fp8 \
    --attention_backend SDPA \
    --vdn_window_backend aiter_ck \
    --use_torch_compile
```

The `stage-dmd-step-250` checkpoint uses eight transformer forward passes.
`--num_inference_steps` defaults to 8, and the runner rejects other values.
The model card expresses the same schedule as Diffusers `num_inference_steps=9`,
which counts nine sigma points.

<a id="amd-hardware-examples"></a>

## Hardware recipes

These examples adapt workloads from AMD's
[configuration snapshot](https://github.com/AMD-AGI/diffusion-models-inference/tree/13fbb792d3c447e7356114ad219940f5b337c778/benchmark_configs/xdit).
Use a matching AMD environment with the selected kernels and model support.
The [RDNA 4 Dockerfile](https://github.com/AMD-AGI/diffusion-models-inference/blob/13fbb792d3c447e7356114ad219940f5b337c778/docker/Dockerfile.rdna4)
and [Instinct CI Dockerfile](https://github.com/AMD-AGI/diffusion-models-inference/blob/13fbb792d3c447e7356114ad219940f5b337c778/docker/Dockerfile.ci)
use different xDiT, AITER, and dependency revisions. Check your selected image
rather than treating those Dockerfile defaults as minimum package requirements.

### FLUX.1-dev on one RDNA 4 GPU

Adapted from `flux.1gpu.rdna4` in
[flux.yaml](https://github.com/AMD-AGI/diffusion-models-inference/blob/13fbb792d3c447e7356114ad219940f5b337c778/benchmark_configs/xdit/flux.yaml).
This targets `gfx1201`, with compatible AITER/FlyDSL FP8 attention kernels and
Cache-DiT installed. It uses the model's guidance default and one generation;
the benchmark source instead sets guidance to zero and repeats timed calls.
FLUX.1-dev requires Hugging Face model access.

```bash
xdit --model black-forest-labs/FLUX.1-dev \
    --prompt "A small cat" \
    --seed 42 \
    --height 1024 --width 1024 \
    --num_inference_steps 25 \
    --max_sequence_length 256 \
    --attention_backend aiter_flydsl_fp8 \
    --gemm_quantization fp8 \
    --use_fp8_text_encoder \
    --cache_method dbcache \
    --cache_config '{"residual_diff_threshold": 0.16}' \
    --use_torch_compile
```

### Wan2.2 image-to-video on eight Instinct GPUs

Adapted from `wan2_2.default.gfx942` in
[wan2_2.yaml](https://github.com/AMD-AGI/diffusion-models-inference/blob/13fbb792d3c447e7356114ad219940f5b337c778/benchmark_configs/xdit/wan2_2.yaml).
This targets eight `gfx942` GPUs and compatible AITER attention kernels.
Replace the image path with a local file visible inside your environment. The
source's `gfx950` default selects `aiter_bf16`; use the workload for your hardware
when selecting an attention backend.

```bash
xdit --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --input_images /path/to/wan_input.jpg \
    --prompt "A white cat wearing sunglasses on a surfboard at the beach" \
    --seed 42 \
    --height 720 --width 1280 \
    --num_frames 81 \
    --num_inference_steps 40 \
    --ulysses_degree 8 \
    --use_parallel_vae \
    --attention_backend aiter \
    --memory_efficient_replicated_load \
    --use_torch_compile
```

Quantization and caching affect computation and output quality. Evaluate their
settings on your workload; these adapted commands do not reproduce benchmark
measurements.

## External model integrations

These models run from separate repositories, with their own launchers and
dependency requirements. They are not registered `xdit --model` names, and their
features are not included in the runner's capability tables.

| Model | Integration | Parallelism |
| --- | --- | --- |
| Step-Video-T2V | [Upstream multi-GPU deployment](https://github.com/stepfun-ai/Step-Video-T2V#multi-gpu-parallel-deployment) | The upstream `run_parallel.py` uses Ulysses and xDiT's transformer TP applicator. Follow its instructions for separate text-encoder and VAE services. |
| Mochi-1 preview | [mochi-xdit](https://github.com/xdit-project/mochi-xdit) | Provides Ulysses, Ring, and CFG parallelism through a dedicated inference implementation. |

<a id="dedicated-pipeline-examples"></a>

## Legacy pipeline examples

These scripts use the separate legacy pipeline interface.
Run them from the repository root after a source installation. Pass a compatible
checkpoint ID or local directory to `--model` and follow each script's inputs.

| Model or workflow | Example script |
| --- | --- |
| PixArt-alpha / PixArt-Sigma | [alpha](https://github.com/xdit-project/xDiT/blob/main/examples/pixartalpha_example.py), [Sigma](https://github.com/xdit-project/xDiT/blob/main/examples/pixartsigma_example.py) |
| HunyuanDiT | [HunyuanDiT](https://github.com/xdit-project/xDiT/blob/main/examples/hunyuandit_example.py) |
| CogVideoX / CogVideoX1.5 | [CogVideoX](https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_example.py), [USP](https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_usp_example.py) |
| ConsisID | [ConsisID](https://github.com/xdit-project/xDiT/blob/main/examples/consisid_example.py), [USP](https://github.com/xdit-project/xDiT/blob/main/examples/consisid_usp_example.py) |
| Latte | [Latte](https://github.com/xdit-project/xDiT/blob/main/examples/latte_example.py) |
| SANA / SANA Sprint | [SANA](https://github.com/xdit-project/xDiT/blob/main/examples/sana_example.py), [Sprint](https://github.com/xdit-project/xDiT/blob/main/examples/sana_sprint_example.py) |
| SDXL | [SDXL](https://github.com/xdit-project/xDiT/blob/main/examples/sdxl_example.py) |
| FLUX.2 direct pipeline | [FLUX.2](https://github.com/xdit-project/xDiT/blob/main/examples/flux2_example.py) |
| FLUX ControlNet | [ControlNet](https://github.com/xdit-project/xDiT/blob/main/examples/flux_control_example.py) |

For a configurable launcher, edit the checkpoint, GPU count, and parallel layout
in [examples/run.sh](https://github.com/xdit-project/xDiT/blob/main/examples/run.sh)
and run `bash examples/run.sh`. Its defaults use eight GPUs.
Match the parallel degrees to the launched process count. Legacy examples can reserve
VAE-only ranks; the unified runner's parallel VAE reuses model ranks.

The direct FLUX.2 script requires a downloaded checkpoint directory containing
`model_index.json`:

```bash
torchrun --nproc_per_node=4 examples/flux2_example.py \
    --model /path/to/FLUX.2-klein-4B \
    --height 1024 --width 1024 --num_inference_steps 4 \
    --ulysses_degree 4 --prompt "A cute dog"
```

### Legacy tensor parallelism

The [FLUX ControlNet script](https://github.com/xdit-project/xDiT/blob/main/examples/flux_control_example.py)
and [Ray FLUX script](https://github.com/xdit-project/xDiT/blob/main/examples/ray/ray_flux_example.py)
use the legacy FLUX wrapper. With `--tensor_parallel_degree` greater than 1,
it shards feed-forward weights and combines their outputs with an all-reduce.
Attention and other transformer weights stay replicated across TP ranks.

These scripts use the legacy parser; the unified runner does not enable this TP path.
Set a compatible checkpoint and match the parallel degrees to the process count.
For ControlNet, set the hard-coded `test_imgs/016#.png` path to your control image.
The Ray script requires `xfuser[ray]` and uses Ray workers to launch the model.
