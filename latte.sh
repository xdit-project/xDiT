HF_ENDPOINT=https://hf-mirror.com CUDA_VISIABLE_DEVICES=0,1,2,3\
    torchrun --nproc_per_node=4 examples/latte_example.py \
    --model ../../Latte-1 \
    --height 512 \
    --width 512 \
    --ulysses_degree 4 \
    --num_inference_steps 50 \
    --prompt "a cat wearing sunglasses and working as a lifeguard at pool." \
    --output_type "pt"