# Launch across devices

Start with Ulysses sequence parallelism on a supported model:

```bash
xdit --model FLUX.1-dev --prompt "A cat running in a garden" \
    --ulysses_degree 4
```

A rank is a distributed worker process. `xdit` uses `torchrun` to start one
process per rank. It calculates the total rank count as:

```text
data_parallel_degree × ulysses_degree × ring_degree
× tensor_parallel_degree × pipefusion_parallel_degree
× (2 when use_cfg_parallel is set, otherwise 1)
```

Each degree defaults to 1. CFG parallelism is optional and requires a model that
supports classifier-free guidance. For a single node, the product is the number
of local devices needed. Set `--nproc_per_node` to override the inferred process
count. The parallel configuration must still match the total number of launched ranks.

`--fully_shard_degree` and `--text_encoder_tp_degree` reuse model ranks and do not
increase the launcher count. Fully Sharded Data Parallel (FSDP) over four ranks requires a matching
parallel layout, such as `--ulysses_degree 4 --fully_shard_degree 4` on a model
supporting both. FSDP and transformer tensor parallelism cannot both exceed degree 1.
See [parallel layouts](../methods/hybrid.md) for layout guidance and
[communication precision](communication.md) for reducing supported Ulysses exchange
traffic. [PipeFusion](../methods/pipefusion.md) has additional scheduling and
composition restrictions.

## Launcher options

`xdit` handles these five launcher options. It does not pass them to the model
runner. Use underscores or dashes and either `--flag value` or `--flag=value`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--nnodes` | `1` | Number of nodes passed to torchrun. Automatic process-count inference requires a positive integer that divides the total rank count. |
| `--node_rank` | `0` | Index of this node in the launch; use a distinct rank on each node. |
| `--master_addr` | `localhost` | Address of the rendezvous host, reachable from every node. |
| `--master_port` | `29500` | Rendezvous port shared by all nodes. |
| `--nproc_per_node` | Inferred | Local processes: total rank count divided by `--nnodes`. An explicit value is passed through to torchrun. |

For several nodes, run the same model command on each node. Use matching degrees,
node count, rendezvous address, and port. Set a different `--node_rank` on each node.
For example, `--ulysses_degree 8 --nnodes 2` starts four processes per node.

The `xdit` launcher does not forward arbitrary torchrun options. For torchrun
features beyond the five above, launch the runner directly:

```bash
torchrun --nproc_per_node=4 -m xfuser.runner \
    --model FLUX.1-dev --ulysses_degree 4 --prompt "A cat running in a garden"
```
