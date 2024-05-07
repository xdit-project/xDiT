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

# docker exec -it 888c58e74578 bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

export PYTHONPATH=$PWD:/mnt/fjr/long-context-attention

# export SCRIPT=ditxl_example.py
# export MODEL_ID="/mnt/models/SD/DiT-XL-2-256"

export SCRIPT=pixart_example.py
export MODEL_ID="/mnt/models/SD/PixArt-XL-2-1024-MS"

# SYNC_MODE="corrected_async_gn"
# SYNC_MODE="full_sync"
# SYNC_MODE="no_sync"

# torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE

# export ACC_FLAG="--use_seq_parallel_attn"

# HEIGHT=2048
HEIGHT=1024
# HEIGHT=512


for N_GPUS in 1 2 4 8;
do
# Tensor Parallel
SYNC_MODE="no_sync"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT -p "tensor" --model_id $MODEL_ID \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning 

# Patch Parallel

# no sync idea
SYNC_MODE="no_sync"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning 

# async
SYNC_MODE="corrected_async_gn"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE  \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning

# sync
SYNC_MODE="full_sync"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning


# sp u=8
SYNC_MODE="full_sync"
export ACC_FLAG="--use_seq_parallel_attn --ulysses_degree $N_GPUS --use_use_ulysses_low"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $ACC_FLAG  \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning


# sp u=1
SYNC_MODE="full_sync"
export ACC_FLAG="--use_seq_parallel_attn --ulysses_degree 1 --use_use_ulysses_low"
torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID --sync_mode $SYNC_MODE $ACC_FLAG  \
--height $HEIGHT --width $HEIGHT --no_use_resolution_binning

# pipeline
num_micro_batchs=(4 8 16 32)
for num_micro_batch in "${num_micro_batchs[@]}"
do
    torchrun --nproc_per_node=$N_GPUS scripts/$SCRIPT --model_id $MODEL_ID -p pipeline  \
    --height $HEIGHT --width $HEIGHT --no_use_resolution_binning --num_micro_batch $num_micro_batch 
done
done