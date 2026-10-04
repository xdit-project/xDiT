#!/bin/bash
# The *_ARGS variables each hold several flags and are word-split on purpose.
# shellcheck disable=SC2086
set -x

# export NCCL_PXN_DISABLE=1
# # export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# export NCCL_P2P=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
# The model is downloaded to a specified location on disk, 
# or you can simply use the model's ID on Hugging Face, 
# which will then be downloaded to the default cache path on Hugging Face.

export MODEL_TYPE="Flux"
# Configuration for different model types
# model_id
declare -A MODEL_CONFIGS=(
    ["Flux"]="/cfs/dit/FLUX.1-schnell"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    MODEL_ID="${MODEL_CONFIGS[$MODEL_TYPE]}"
    export MODEL_ID
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

# The HTTP service is entrypoints/launch.py; see docs/developer/Http_Service.md.
# Prompt, image size, step count and guidance scale are set per request,
# for example with entrypoints/curl.sh. FLUX.1-schnell is meant to run
# with "num_inference_steps": 4.

N_GPUS=1

PARALLEL_ARGS="--ulysses_parallel_degree 1 --ring_degree 1 --pipefusion_parallel_degree 1"

# CFG_ARGS="--use_cfg_parallel"

python ./entrypoints/launch.py \
--model_path $MODEL_ID \
--world_size $N_GPUS \
$PARALLEL_ARGS \
$CFG_ARGS
