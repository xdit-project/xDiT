HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=2 examples/cogvideox_example.py \
    --model ../CogVideoX-2b \
    --height 480 \
    --width 720 \
    --ulysses_degree 2 \
    --num_inference_steps 50 \
    --prompt "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
	