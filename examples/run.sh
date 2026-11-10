#!/bin/bash
# The *_ARGS variables each hold several flags and are word-split on purpose.
# shellcheck disable=SC2086
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
# "runner" runs the unified runner (docs/runner/runner.md), which takes a
# registered model name such as FLUX.1-dev instead of a local path.
# Stable Diffusion 3 medium no longer has an example script; the unified
# runner supports Stable Diffusion 3.5 with --model SD3.5.
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    ["Flux"]="runner FLUX.1-dev 28"
    ["FluxControl"]="flux_control_example.py /cfs/dit/FLUX.1-Depth-dev/ 28"
    ["HunyuanDiT"]="hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
    ["SDXL"]="sdxl_example.py /cfs/dit/stable-diffusion-xl-base-1.0 30"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5"

# Cache aliases below are deprecated; prefer --cache_method in the runner.
# TeaCache/FBCache require PipeFusion degree 1. The default PP=2 layout
# needs --cache_method pipefusion on a supported model instead.
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# On 8 GPUs, pp=2, ulysses=2, ring=2, cfg=1.
# Enabling CFG parallel doubles the rank count; adjust the other degrees or N_GPUS.
N_GPUS=8
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2"

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# Legacy scripts can use latent output to skip VAE decoding when measuring DiT speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Legacy scripts also support --use_onediff; the runner supports --use_torch_compile.
# These compilation options are mutually exclusive in legacy scripts.
# COMPILE_FLAG="--use_torch_compile"


# Legacy scripts can quantize the T5 encoder with optimum-quanto to reduce memory.
# Quantization may affect output quality; validate it for your workload.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

PROMPT="brown dog laying on the ground with a metal bowl in front of him."

if [ "$SCRIPT" = "runner" ]; then
# The runner supports --warmup_steps and --num_pipeline_patch, but this branch
# does not forward PIPEFUSION_ARGS. It also does not forward OUTPUT_ARGS or
# QUANTIZE_FLAG; --output_type, --no_use_resolution_binning, and
# --use_fp8_t5_encoder belong to the legacy-script CLI. For runner flags,
# including --use_fp8_text_encoder, see docs/runner/cli.md.
torchrun --nproc_per_node=$N_GPUS -m xfuser.runner \
--model $MODEL_ID \
$PARALLEL_ARGS \
--height 1024 --width 1024 --guidance_scale 3.5 \
--num_inference_steps $INFERENCE_STEP \
--prompt "$PROMPT" \
--output_directory ./results \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$CACHE_ARGS
else
torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "$PROMPT" \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS
fi
