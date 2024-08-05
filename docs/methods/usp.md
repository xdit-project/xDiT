## USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
[Chinese Blog 1](https://zhuanlan.zhihu.com/p/698031151); [Chinese Blog 2](https://zhuanlan.zhihu.com/p/689067888)

DeepSpeed-Ulysses and Ring-Attention are not mutually exclusive options. 
Both should be used in a mixed manner to jointly split the sequence dimension. 
By adjusting their parallelism degrees to ensure that ulysses-degree multiplied by ring-degree equals sp-degree, we refer to this as Unified-SP.
The advantage of Unified-SP is that it encompasses the capabilities of both original methods without any loss, only offering additional benefits.
Firstly, it eliminates the restriction that Ulysses' sp-degree must be less than the number of attention heads. 
Moreover, the communication pattern of mixed parallelism is more friendly to heterogeneous networks, providing acceleration over PCIe and in multi-machine multi-GPU environments compared to using Ulysses or Ring alone. Therefore, we recommend using the Unified-SP implementation as the default sequence parallelism solution.

In xDiT, we utilize the USP implementation from [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention). Since DiT does not use Causal Attention, there is no need for load balancing operations on Ring-Attention. For more details, please refer to the following [paper](https://arxiv.org/abs/2405.07719).

```
@article{fang2024unified,
  title={USP: A Unified Sequence Parallelism Approach for Long Context Generative AI},
  author={Fang, Jiarui and Zhao, Shangchun},
  journal={arXiv preprint arXiv:2405.07719},
  year={2024}
}
```