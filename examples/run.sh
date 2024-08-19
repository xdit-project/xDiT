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


# Select the model type from Pixart-alpha, Pixart-sigma, Sd3, or Flux
# The model is downloaded to a specified location on disk, 
# or you can simply use the model's ID on Hugging Face, 
# which will then be downloaded to the default cache path on Hugging Face.

export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_TYPE="HunyuanDiT"

# CFG_ARGS="--use_cfg_parallel"

if [ "$MODEL_TYPE" = "Pixart-alpha" ]; then
    export SCRIPT=pixartalpha_example.py
    export MODEL_ID="/mnt/models/SD/PixArt-XL-2-1024-MS"
    export INFERENCE_STEP=20
elif [ "$MODEL_TYPE" = "Pixart-sigma" ]; then
    export SCRIPT=pixartsigma_example.py
    export MODEL_ID="/cfs/dit/PixArt-Sigma-XL-2-2K-MS"
    export INFERENCE_STEP=20
elif [ "$MODEL_TYPE" = "Sd3" ]; then
    export SCRIPT=sd3_example.py
    export MODEL_ID="/mnt/models/SD/stable-diffusion-3-medium-diffusers"
    export INFERENCE_STEP=20
elif [ "$MODEL_TYPE" = "Flux" ]; then
    export SCRIPT=flux_example.py
    export MODEL_ID="/mnt/models/SD/FLUX.1-schnell"
    export INFERENCE_STEP=4
    # Flux does not apply cfg
    export CFG_ARGS=""
elif [ "$MODEL_TYPE" = "HunyuanDiT" ]; then
    export SCRIPT=hunyuandit_example.py
    export MODEL_ID="/mnt/models/SD/HunyuanDiT-v1.2-Diffusers"
    export INFERENCE_STEP=20
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi



mkdir -p ./results

for HEIGHT in 1024
do
for N_GPUS in 1;
do 

TASK_ARGS="--height $HEIGHT \
--width $HEIGHT \
--no_use_resolution_binning \
"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 1"

# Flux only supports SP, do not set the pipefusion degree
if [ "$MODEL_TYPE" = "Flux" ]; then
PARALLEL_ARGS="--ulysses_degree $N_GPUS"
elif [ "$MODEL_TYPE" = "HunyuanDiT" ]; then
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1"
fi


# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A small dog" \
$CFG_ARGS

done
done


