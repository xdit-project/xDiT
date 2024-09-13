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

export MODEL_TYPE="CogVideoX"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /mnt/models/SD/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py /cfs/dit/FLUX.1-schnell 4"
    ["HunyuanDiT"]="hunyuandit_example.py /mnt/models/SD/HunyuanDiT-v1.2-Diffusers 50"
    ["CogVideoX"]="cogvideox_example.py /cfs/dit/CogVideoX-2b 9"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

for HEIGHT in 1024
do
for N_GPUS in 1;
do 


# task args
if [ "$MODEL_TYPE" = "CogVideoX" ]; then
  TASK_ARGS="--height 480 --width 720 --num_frames 9"
else
  TASK_ARGS="--height $HEIGHT --width $HEIGHT --no_use_resolution_binning"
fi

# Flux only supports SP, do not set the pipefusion degree
if [ "$MODEL_TYPE" = "Flux" ] || [ "$MODEL_TYPE" = "CogVideoX" ]; then
PARALLEL_ARGS="--ulysses_degree $N_GPUS"
export CFG_ARGS=""
elif [ "$MODEL_TYPE" = "HunyuanDiT" ]; then
# HunyuanDiT asserts sp_degree <=2, or the output will be incorrect.
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 2 --ring_degree 1"
export CFG_ARGS="--use_cfg_parallel"
else
# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 1"
export CFG_ARGS="--use_cfg_parallel"
fi


# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
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

done
done


