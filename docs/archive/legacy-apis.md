# Legacy APIs

We conducted a major upgrade of this project in August 2024, introducing a new set of APIs that are now the preferred choice for all users.

The legacy APIs are applied in early stage of xDiT to explore and compare different parallelization methods.
They are located in the [legacy](https://github.com/xdit-project/xDiT/tree/legacy) branch, are now considered outdated and do not support hybrid parallelism. Despite this limitation, they offer a broader range of individual parallelization methods, including PipeFusion, Sequence Parallel, DistriFusion, and Tensor Parallel.

For users working with Pixart models, you can still run the examples in the [scripts/](https://github.com/xdit-project/xDiT/tree/legacy/scripts) directory under the `legacy` branch. However, for all other models, we strongly recommend adopting the formal APIs to ensure optimal performance and compatibility.
