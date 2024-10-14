#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="cogvideox_example.py"
MODEL_ID="/cfs/dit/CogVideoX-2b"
INFERENCE_STEP=20

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 480 --width 720 --num_frames 9"

# CogVideoX parallel configuration
N_GPUS=4
PARALLEL_ARGS="--ulysses_degree 2 --ring_degree 1"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A small dog" \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG