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

# docker exec -it 888c58e74578 bash
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# export CUDA_VISIBLE_DEVICES="4,5,6,7"


export SCRIPT=pixartalpha_example.py
export MODEL_ID="/mnt/models/SD/PixArt-XL-2-1024-MS"



for HEIGHT in 1024
do
for N_GPUS in 8;
do 

ARSG="--height $HEIGHT \
--width $HEIGHT \
--num_pipeline_patch 8
"

# On 4 gpus, pp=2, ulysses=1, ring=1, cfg_parallel=2 (split batch)
torchrun --nproc_per_node=$N_GPUS ./examples/pixartalpha_example.py \
--model $MODEL_ID --pipefusion_parallel_degree 2 --ulysses_degree 2 \
--num_inference_steps 20 \
--warmup_steps 0 \
--prompt "A small dog" \
--use_split_batch

done
done
