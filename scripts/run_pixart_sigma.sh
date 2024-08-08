#!/bin/bash
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

# Check if MODEL_ID is provided as an argument
if [ $# -eq 0 ]; then
    echo "Error: MODEL_ID is required."
    echo "Usage: $0 <MODEL_ID>"
    exit 1
fi

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export SCRIPT="$SCRIPT_DIR/../examples/pixartsigma_example.py"
export MODEL_ID="$1"

# On 8 gpus, pp=2, ulysses=12, ring=1, cfg_parallel=2 (split batch)
torchrun --nproc_per_node=8 $SCRIPT \
--model $MODEL_ID \
--pipefusion_parallel_degree 2 \
--ulysses_degree 2 \
--num_inference_steps 20 \
--warmup_steps 0 \
--prompt "A small dog" \
--use_cfg_parallel
