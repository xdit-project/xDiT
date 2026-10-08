#!/bin/bash
set -x

# This sweep drives the scripts in examples/. FLUX.1-dev and Stable Diffusion 3
# no longer have one there; benchmark them with the unified runner instead
# (for example `xdit --model FLUX.1-dev ...`, see docs/runner/runner.md).

MODEL="/mnt/models/SD/PixArt-XL-2-1024-MS"
SCRIPT="./examples/pixartalpha_example.py"

# MODEL="/mnt/models/SD/HunyuanDiT-v1.2-Diffusers"
# SCRIPT="./examples/hunyuandit_example.py"

export PYTHONPATH=$PWD:$PYTHONPATH

python benchmark/single_node_latency_test.py \
--model_id $MODEL \
--script $SCRIPT \
--sizes 1024 \
--no_use_resolution_binning \
--num_inference_steps 20 \
--no_use_cfg_parallel \
--n_gpus 4