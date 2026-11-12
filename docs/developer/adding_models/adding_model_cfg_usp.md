# Combine CFG parallelism and USP

Combine [branch partitioning](adding_model_cfg.md) with [sequence partitioning](adding_model_usp.md)
when the model supports both. The [Z-Image runner adapter](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/z_image.py)
supports CFG, Ulysses, and Ring. Z-Image-Turbo supports sequence parallelism but
not CFG parallelism.

## Rank layout

Without other parallel methods, the process count is
`2 × ulysses_degree × ring_degree`. For example, CFG degree 2 and Ulysses degree 4
use eight ranks. Each branch's sequence group cooperates on attention. CFG groups exchange
predictions between corresponding ranks of the two branches.
See [hybrid layouts](../../methods/hybrid.md) for layouts with other degrees.

Initialize process groups and runtime state through the runner's configuration
path. Attention reads backend settings and sequence metadata as well as group
membership. Creating process groups alone is insufficient.

## Execution order

The [Z-Image transformer wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_z_image.py)
selects the local CFG batch first. It then partitions image, caption, and combined
sequences with matching masks and positional embeddings. Its attention processor
pads heads when needed and calls `USP`. Each refinement phase gathers sequence
outputs and removes padding before the next phase. After the final sequence gather
and unpatchification, a CFG gather restores positive-first prediction order.
Diffusers' original pipeline applies guidance and updates the latents.

Keep branch and sequence groups distinct. A world-group gather would mix branches
or replicas. Preserve branch order, token positions, masks, and padding when
adapting another model. Validate CFG and USP separately, then together with the
same inputs, seed, precision, and active guidance settings.
