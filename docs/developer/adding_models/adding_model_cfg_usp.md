# Parallelize new models with CFG parallelism and USP provided by xDiT

The following two tutorials provide detailed instructions on how to implement CFG parallelism and USP (Unified Sequence Parallelism) supported by xDiT for a new DiT model:

[Parallelize new models with CFG parallelism provided by xDiT](adding_model_cfg.md)

[Parallelize new models with USP provided by xDiT](adding_model_usp.md)

[Parallelize new models with USP provided by xDiT (text replica)](adding_model_usp_text_replica.md)

Both parallelization techniques can be concurrently employed. To achieve this, specify the level of parallelization for both CFG parallelism and USP as demonstrated below. The number of GPUs should be twice the product of the degrees of ulysses attention and ring attention:

```python
from xfuser.core.distributed import initialize_model_parallel
initialize_model_parallel(
    sequence_parallel_degree=<ring_degree x ulysses_degree>,
    ring_degree=<ring_degree>,
    ulysses_degree=<ulysses_degree>,
    classifier_free_guidance_degree=2,
)
# restriction: dist.get_world_size() == 2 x <ring_degree> x <ulysses_degree>
```

Following this, both CFG parallelism and USP can be simultaneously implemented. For a comprehensive example script showcasing this approach, refer to [adding_model_cfg_usp.py](adding_model_cfg_usp.py).
