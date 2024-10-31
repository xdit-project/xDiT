set -x

export PYTHONPATH=$PWD:$PYTHONPATH
export CAPTION_FILE="dataset_coco.json"
export SAMPLE_IMAGES_FOLODER="sample_images"

# Select the model type
export MODEL_TYPE="Pixart-alpha"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_generate.py /cfs/dit/PixArt-XL-2-256-MS 20"
    ["Flux"]="flux_generate.py /cfs/dit/FLUX.1-dev 28"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

# task args
TASK_ARGS="--height 256 --width 256 --no_use_resolution_binning"

N_GPUS=8
PARALLEL_ARGS="--pipefusion_parallel_degree 8 --ulysses_degree 1 --ring_degree 1"

torchrun --nproc_per_node=$N_GPUS ./benchmark/fid/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "brown dog laying on the ground with a metal bowl in front of him." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
--caption_file $CAPTION_FILE \
--sample_images_folder $SAMPLE_IMAGES_FOLODER \
