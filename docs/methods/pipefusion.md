## PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models
[Chinese Blog 1](https://zhuanlan.zhihu.com/p/699612077); [Chinese Blog 2](https://zhuanlan.zhihu.com/p/706475158)

PipeFusion is the innovative method first proposed by us. 
It is a sequence-level pipeline parallel method, similar to [TeraPipe](https://proceedings.mlr.press/v139/li21y.html), demonstrates significant advantages in weakly interconnected network hardware such as PCIe/Ethernet. 

PipeFusion innovatively harnesses input temporal redundancy—the similarity between inputs and activations across diffusion steps, a diffusion-specific characteristics also employed in DistriFusion. PipeFusion not only reduces communication volume but also streamlines pipeline parallelism with TeraPipe, avoiding the load balancing issues inherent in LLM models with Causal Attention.
It significantly surpasses other methods in communication efficiency, particularly in multi-node setups connected via Ethernet and multi-GPU configurations linked with PCIe.

<div align="center">
    <img src="../../assets/overview.png" alt="PipeFusion Image">
</div>

The above picture compares DistriFusion and PipeFusion.
(a) DistriFusion replicates DiT parameters on two devices. 
It splits an image into 2 patches and employs asynchronous allgather for activations of every layer.
(b) PipeFusion shards DiT parameters on two devices.
It splits an image into 4 patches and employs asynchronous P2P for activations across two devices.

We briefly explain the workflow of PipeFusion. It partitions an input image into $M$ non-overlapping patches.
The DiT network is partitioned into $N$ stages ($N$ < $L$), which are sequentially assigned to $N$ computational devices. 
Note that $M$ and $N$ can be unequal, which is different from the image-splitting approaches used in sequence parallelism and DistriFusion.
Each device processes the computation task for one patch of its assigned stage in a pipelined manner. 

The PipeFusion pipeline workflow when $M$ = $N$ =4 is shown in the following picture.

<div align="center">
    <img src="../../assets/workflow.png" alt="Pipeline Image">
</div>


We have evaluated the accuracy of PipeFusion, DistriFusion and the baseline as shown bolow. To conduct the FID experiment, follow the detailed instructions provided in the [documentation](../../docs/fid/FID.md).

<div align="center">
    <img src="../../assets/image_quality.png" alt="image_quality">
</div>


For more details, please refer to the following paper.

```
@article{wang2024pipefusion,
      title={PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models}, 
      author={Jiannan Wang and Jiarui Fang and Jinzhe Pan and Aoyu Li and PengCheng Yang},
      year={2024},
      eprint={2405.07719},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

