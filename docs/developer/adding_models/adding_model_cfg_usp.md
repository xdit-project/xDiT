# Parallelize new models with CFG parallelism and USP provided by xDiT

The following two tutorials provide detailed instructions on how to implement CFG parallelism and USP (Unified Sequence Parallelism) supported by xDiT for a new DiT model:

[Parallelize new models with USP provided by xDiT](adding_model_usp.md)

[Parallelize new models with CFG parallelism provided by xDiT](adding_model_cfg.md)

Both parallelization methods can be employed simultaneously. To do so, the level of parallelization for both CFG parallelism and USP needs to be specified as shown below. The number of GPUs should be equal to 2 times the degrees of ulysses attention times the degrees of ring attention:

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

Subsequently, both CFG parallelism and USP can be applied concurrently. For a complete example script demonstrating this, refer to [adding_model_cfg_usp.py](adding_model_cfg_usp.py).