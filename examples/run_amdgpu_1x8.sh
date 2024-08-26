#!/usr/bin/bash
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )

set -x

# single node disable IB device
export NCCL_IB_DISABLE=1
export RCCL_IB_DISABLE=1

# enable DMA buffer
export NCCL_ENABLE_DMABUF_SUPPORT=1
export RCCL_ENABLE_DMABUF_SUPPORT=1

export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

# enable sharp in multinodes config

# Select the model type from Pixart-alpha, Pixart-sigma, Sd3, or Flux
# The model is downloaded to a specified location on disk, 
# or you can simply use the model's ID on Hugging Face, 
# which will then be downloaded to the default cache path on Hugging Face.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0

OUTPUT_BASEPATH=$SCRIPT_ROOT

# pipeline parallel
PP=${PP:-2}

# tensor parllel
TP=${TP:-1}

# ring deg
CP=${CP:-1}



export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_TYPE="Pixart-alpha"

CFG_ARGS="--use_cfg_parallel"

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

# for HEIGHT in 1024
# do
# for N_GPUS in 8;
# do 

HEIGHT=1024
N_GPUS=8

DISTR_ARGS="
  --nproc_per_node $N_GPUS \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
"

TASK_ARGS="--height $HEIGHT \
--width $HEIGHT \
--no_use_resolution_binning \
"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
PARALLEL_ARGS="--pipefusion_parallel_degree $PP --ulysses_degree 2 --ring_degree $CP"

# Flux only supports SP, do not set the pipefusion degree
if [ "$MODEL_TYPE" = "Flux" ]; then
PARALLEL_ARGS="--ulysses_degree $N_GPUS"
elif [ "$MODEL_TYPE" = "HunyuanDiT" ]; then
echo "change PP from $PP to 1"
PP=1
PARALLEL_ARGS="--pipefusion_parallel_degree $PP --ulysses_degree 8 --ring_degree $CP"
fi

echo "PARALLEL ARGS : ${PARALLEL_ARGS}"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

mkdir -p ${OUTPUT_BASEPATH}/log/${MODEL_TYPE}

torchrun $DISTR_ARGS $SCRIPT_ROOT/examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A small dog" \
$CFG_ARGS \
  &> ${OUTPUT_BASEPATH}/log/${MODEL_TYPE}/${NODE_RANK}.log

# done
# done


