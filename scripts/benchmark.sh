export N_GPUS=4

# export NCCL_PXN_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# export NCCL_P2P=0

# docker exec -it 98437bb20829 bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

export PYTHONPATH=$PWD:/mnt/fjr/long-context-attention

# export SCRIPT=ditxl_example.py
# export MODEL_ID="/mnt/models/SD/DiT-XL-2-256"

# HEIGHT=512
HEIGHT=1024
# HEIGHT=2048
# HEIGHT=4096
# HEIGHT=8192

export SCRIPT=./scripts/pixart_example.py
export MODEL_ID="/mnt/models/SD/PixArt-XL-2-1024-MS"
export TASK_SIZE="--height $HEIGHT --width $HEIGHT --no_use_resolution_binning"




for N_GPUS in 4;
do


# Sequence Parallelism
# sp u=8, ulyssess
# SYNC_MODE="full_sync"
# export ACC_FLAG="--use_seq_parallel_attn --ulysses_degree 4 --use_use_ulysses_low"
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $ACC_FLAG $TASK_SIZE

# sp u=1, ring
# SYNC_MODE="full_sync"
# export ACC_FLAG="--use_seq_parallel_attn --ulysses_degree 2 --use_use_ulysses_low"
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $ACC_FLAG $TASK_SIZE

# Tensor Parallel
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT -p "tensor" --model_id $MODEL_ID $TASK_SIZE


# Patch Parallel

# no sync idea, wrong results
# SYNC_MODE="no_sync"
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $TASK_SIZE

# DistrFusion
# SYNC_MODE="corrected_async_gn"
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $TASK_SIZE

# Sync DistrFusion
# SYNC_MODE="full_sync"
# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $TASK_SIZE


# PipeFusion
pp_num_patchs=(4 8 16 32)
for pp_num_patch in "${pp_num_patchs[@]}"
do
    torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID -p pipefusion  \
    --height $HEIGHT --width $HEIGHT --no_use_resolution_binning --pp_num_patch $pp_num_patch 
done
done