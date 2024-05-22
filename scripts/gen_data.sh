output_list=("bus" "dog" "kid" "parking" "parrot")

for output in "${output_list[@]}"; do
    if [ "$output" == "bus" ]; then
        prompt="A double decker bus driving down the street."
    elif [ "$output" == "kid" ]; then
        prompt="A kid wearing headphones and using a laptop."
    elif [ "$output" == "dog" ]; then
        prompt="A brown dog laying on the ground with a metal bowl in front of him."
    elif [ "$output" == "parking" ]; then
        prompt="A pair of parking meters reflecting expired times."
    elif [ "$output" == "parrot" ]; then
        prompt="A multi-colored parrot holding its foot up to its beak."
    fi
    echo "$prompt"
    torchrun --nproc_per_node=8 scripts/pixart_example.py \
    --model_id /mnt/models/SD/PixArt-XL-2-1024-MS --height 1024 --width 1024 --no_use_resolution_binning \
    --output_type pil --num_micro_batch 8 --output_file "$output" --prompt "$prompt"

    torchrun --nproc_per_node=8 scripts/pixart_example.py \
    --model_id /mnt/models/SD/PixArt-XL-2-1024-MS --height 1024 --width 1024 --no_use_resolution_binning \
    --output_type pil --num_micro_batch 8 --output_file "$output" --prompt "$prompt" -p pipeline

    torchrun --nproc_per_node=8 scripts/pixart_example.py \
    --model_id /mnt/models/SD/PixArt-XL-2-1024-MS --height 1024 --width 1024 --no_use_resolution_binning \
    --output_type pil --num_micro_batch 4 --output_file "$output" --prompt "$prompt" -p pipeline

done
