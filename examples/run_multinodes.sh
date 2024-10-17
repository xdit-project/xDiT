set -x

# nccl settings
#export NCCL_DEBUG=INFO

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_DISABLE=0
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_IB_QPS_PER_CONNECTION=4
#export NCCL_IB_TC=160
#export NCCL_IB_TIMEOUT=22
# export NCCL_P2P=0

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export PYTHONPATH=$PWD:$PYTHONPATH


NRANK=0
MASTERIP=127.0.0.1
MASTERPORT=6000
DISTARGAS="--nnodes=2 --node_rank=${NRANK} --master_addr=${MASTERIP}  --master_port=${MASTERPORT}"

SCRIPT=pixartalpha_example.py
MODEL_ID="/cfs/dit/PixArt-XL-2-1024-MS/"
INFERENCE_STEP=20

SIZE=1024
PARALLEL_ARGS="--ulysses_degree=1 --ring_degree=1 --pipefusion_parallel_degree=8"
TASK_ARGS="--height=${SIZE} --width=${SIZE} --no_use_resolution_binning"
OUTPUT_ARGS="--output_type=latent"
CFG_ARGS="--use_cfg_parallel"

# PARALLLEL_VAE="--use_parallel_vae"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=8 $DISTARGAS \
./examples/$SCRIPT \
--model=$MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps=0 \
--prompt="brown dog laying on the ground with a metal bowl in front of him." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG
