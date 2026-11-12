# DiTFastAttn

[中文](ditfastattn_zh.md)

DiTFastAttn approximates attention using window attention with residual caching,
reuse across denoising steps, and reuse between CFG branches. Use the dedicated PixArt examples for this feature.
The unified `xdit` runner does not support it. DiTFastAttn supports one GPU or
data parallelism. It cannot combine with USP, CFG parallelism, or PipeFusion.

## Run the dedicated example

Download COCO captions for attention-strategy calibration:

```bash
wget https://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

From the repository root, run:

```bash
torchrun --nproc_per_node=1 examples/pixartalpha_example.py \
    --model PixArt-alpha/PixArt-XL-2-1024-MS \
    --prompt "A small dog" \
    --height 1024 --width 1024 --guidance_scale 4.5 \
    --num_inference_steps 20 --warmup_steps 0 \
    --no_use_resolution_binning \
    --use_fast_attn --window_size 512 --n_calib 4 --threshold 0.15 \
    --coco_path annotations/captions_val2014.json --use_cache
```

`--n_calib` selects the number of calibration captions. `--threshold` controls
compression selection. `--window_size` sets the local attention window.
`--use_cache` reuses a saved compression strategy when one exists. It is different
from the unified runner's step-cache methods.

This integration requires PixArt's attention layout.
Compare fixed-seed outputs and inference time with the same example without
`--use_fast_attn`. Calibration adds startup work. Compression can affect quality.

See the [attention implementation](https://github.com/xdit-project/xDiT/tree/main/xfuser/core/fast_attention),
[DiTFastAttn project](https://github.com/thu-nics/DiTFastAttn), and
[paper](https://arxiv.org/abs/2406.08552) for details.
