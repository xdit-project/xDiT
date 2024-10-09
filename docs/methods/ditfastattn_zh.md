### DiTFastAttn

[DiTFastAttn](https://github.com/thu-nics/DiTFastAttn)是一种针对单卡DiTs推理的加速方案，利用Input Temperal Reduction通过如下三种方式来减少计算量：

1. Window Attention with Residual Caching to reduce spatial redundancy.
2. Temporal Similarity Reduction to exploit the similarity between steps.
3. Conditional Redundancy Elimination to skip redundant computations during conditional generation

目前使用DiTFastAttn只能数据并行，或者单GPU运行。不支持其他方式并行，比如USP和PipeFusion等。我们未来计划实现并行版本的DiTFastAttn。

## 下载COCO数据集
```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

## 运行

在脚本中修改数据集路径，然后运行

```
bash examples/run_fastditattn.sh
```

## 引用

```
@misc{yuan2024ditfastattn,
      title={DiTFastAttn: Attention Compression for Diffusion Transformer Models}, 
      author={Zhihang Yuan and Pu Lu and Hanling Zhang and Xuefei Ning and Linfeng Zhang and Tianchen Zhao and Shengen Yan and Guohao Dai and Yu Wang},
      year={2024},
      eprint={2406.08552},
      archivePrefix={arXiv},
}
```