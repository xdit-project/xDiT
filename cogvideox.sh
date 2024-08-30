HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=2 examples/cogvideox_example.py \
    --model ../CogVideoX-2b \
    --height 480 \
    --width 720 \
    --ulysses_degree 2 \
    --num_inference_steps 50 \
    --prompt "A cat is playing with a ball" \
	