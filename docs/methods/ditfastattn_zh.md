# DiTFastAttn

[English Version](ditfastattn.md)

DiTFastAttn 通过带残差缓存的窗口注意力、跨去噪步骤复用和 CFG 分支间复用来
近似注意力计算。xDiT 在专用 PixArt 示例中提供此功能，统一 `xdit` 运行器
不提供它。支持单 GPU 或数据并行，不支持与 USP、CFG 并行或 PipeFusion 组合。

## 运行专用示例

下载用于注意力策略校准的 COCO 描述数据：

```bash
wget https://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

从仓库根目录运行：

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

`--n_calib` 指定校准描述数量；`--threshold` 控制压缩策略选择，
`--window_size` 设置局部注意力窗口。`--use_cache` 在已有策略文件时复用它，
与统一运行器的步骤缓存方法不同。

这里保留 PixArt 是因为此集成针对其注意力布局；仅将模型 ID 换成 FLUX
不会得到 FLUX 实现。使用固定种子，与不启用 `--use_fast_attn` 的同一示例
比较输出和推理时间。校准会增加启动工作，压缩可能影响质量。

更多信息见[注意力实现](https://github.com/xdit-project/xDiT/tree/main/xfuser/core/fast_attention)、
[DiTFastAttn 项目](https://github.com/thu-nics/DiTFastAttn)和
[论文](https://arxiv.org/abs/2406.08552)。
