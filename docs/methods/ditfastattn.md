### DiTFastAttn: Attention Compression for Diffusion Transformer Models

[DiTFastAttn](https://github.com/thu-nics/DiTFastAttn) is an acceleration solution for single-GPU DiTs inference, utilizing Input Temporal Reduction to reduce computational complexity through the following three methods:

1. Window Attention with Residual Caching to reduce spatial redundancy.
2. Temporal Similarity Reduction to exploit the similarity between steps.
3. Conditional Redundancy Elimination to skip redundant computations during conditional generation

Currently, DiTFastAttn can only be used with data parallelism or on a single GPU. It does not support other parallel methods such as USP and PipeFusion. We plan to implement a parallel version of DiTFastAttn in the future.

## Download COCO Dataset
```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

## Running

Modify the dataset path in the script, then run

```
bash examples/run_fastditattn.sh
```

## Reference

```
@misc{yuan2024ditfastattn,
      title={DiTFastAttn: Attention Compression for Diffusion Transformer Models}, 
      author={Zhihang Yuan and Pu Lu and Hanling Zhang and Xuefei Ning and Linfeng Zhang and Tianchen Zhao and Shengen Yan and Guohao Dai and Yu Wang},
      year={2024},
      eprint={2406.08552},
      archivePrefix={arXiv},
}
```