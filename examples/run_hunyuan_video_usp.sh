#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="hunyuan_video_usp_example.py"
MODEL_ID="/cfs/dit/HunyuanVideo"
# MODEL_ID="tencent/HunyuanVideo"
INFERENCE_STEP=50

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 720 --width 1280 --num_frames 129"

# CogVideoX parallel configuration
N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 2"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A cat walks on the grass, realistic" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$ENABLE_MODEL_CPU_OFFLOAD \
$COMPILE_FLAG
