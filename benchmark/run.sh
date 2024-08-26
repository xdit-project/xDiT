set -x

# MODEL="/mnt/models/SD/PixArt-XL-2-1024-MS"
# SCRIPT="./examples/pixartalpha_example.py"

# MODEL="/mnt/models/SD/stable-diffusion-3-medium-diffusers"
# SCRIPT="./examples/sd3_example.py"

MODEL="/mnt/models/SD/HunyuanDiT-v1.2-Diffusers"
SCRIPT="./examples/hunyuandit_example.py"

export PYTHONPATH=$PWD:$PYTHONPATH

python benchmark/single_node_latency_test.py \
--model_id $MODEL \
--script $SCRIPT \
--sizes 1024 \
--no_use_resolution_binning