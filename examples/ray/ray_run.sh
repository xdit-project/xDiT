set -x
# If using a Ray cluster across multiple machines, you need to manually start a Ray cluster like this:
# ray start --head --port=6379 for master node
# ray start --address='192.168.1.1:6379' for worker node
# otherwise, it is not necessary. (for single node)

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Sd3"]="ray_sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="ray_flux_example.py /cfs/dit/FLUX.1-dev 28"
    ["Pixart-alpha"]="ray_pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="ray_pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-1024-MS 20"
    ["HunyuanDiT"]="ray_hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"


N_GPUS=3 # world size
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1"
VAE_PARALLEL_SIZE=1
DIT_PARALLEL_SIZE=2
# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# It is necessary to set CUDA_VISIBLE_DEVICES for the ray driver and workers.
export CUDA_VISIBLE_DEVICES=4,5,6,7

python ./examples/ray/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "brown dog laying on the ground with a metal bowl in front of him." \
--use_ray \
--ray_world_size $N_GPUS \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
--use_parallel_vae \
--dit_parallel_size $DIT_PARALLEL_SIZE \
--vae_parallel_size $VAE_PARALLEL_SIZE
