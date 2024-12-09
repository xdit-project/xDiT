#!/bin/bash
set -x

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT=cogvideox_i2v_example.py

# MODEL_ID="/cfs/dit/CogVideoX1.5-5B-I2V"
MODEL_ID="/sse_ard/pretrained_models/cogvideox1.5-5b-i2v/"
INFERENCE_STEP=50

mkdir -p ./results/cuda

# CogVideoX specific task args
TASK_ARGS="--height 768 --width 1360 --num_frames 17"

# CogVideoX parallel configuration
N_GPUS=2
PARALLEL_ARGS="--ulysses_degree 1  --ring_degree 1"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"
ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"


torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
--image "/home/qinlei/Projects/astronaut.jpg" \
--seed 1024 \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG \
$ENABLE_MODEL_CPU_OFFLOAD

# --image "astronaut.jpg" \


