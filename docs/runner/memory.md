# Memory and loading

Start with default loading. Add memory controls only where needed.
Fully Sharded Data Parallel (FSDP) divides model parameters among existing ranks.
Transformer tensor parallelism (TP) divides transformer computation.

## Memory controls

| Control | Effect and restrictions |
| --- | --- |
| `--fully_shard_degree N` | FSDP parameter sharding across existing model ranks. Pair with a matching layout, such as `--ulysses_degree N`; it does not launch more processes. Transformer TP and FSDP cannot both exceed 1. |
| `--fully_shard_components ...` | Select pipeline components for FSDP. With PipeFusion, select replicated components such as a supported text encoder, rather than the stage-local transformer. |
| `--text_encoder_tp_degree N` | Shards a supported text encoder across existing ranks. MiniMax-H3 and MiniMax-H3-Image require this degree to be 1 or match `--ulysses_degree`. MiniMax-H3 multi-GPU runs use this instead of text-encoder CPU offload. MiniMax-H3-Image encodes on the first rank by default; use this only if the encoder does not fit. |
| `--memory_efficient_sharding` | Requires FSDP degree greater than 1 and a supported load route. Loads checkpoint blocks through rank 0 before filling shards. This avoids loading the full model on every device first. |
| `--memory_efficient_replicated_load` | Reduces load-time memory for supported multi-rank replicated layouts; weights remain replicated after loading. Supported FLUX PipeFusion runners instead load only each stage's transformer blocks. A single-rank run uses standard loading. |
| `--enable_model_cpu_offload` | Moves pipeline components between host and device. Supported with FSDP; MiniMax-H3 and MiniMax-H3-Image use a specialized text-encoder-only offload on one GPU. |
| `--enable_group_cpu_offload` / `--enable_sequential_cpu_offload` | Finer-grained host offload; both are rejected with FSDP. AITER FP4 also rejects group offload. MiniMax-H3 and MiniMax-H3-Image reject both. |
| `--enable_tiling` / `--enable_slicing` | Model-specific VAE memory controls. See [parallel VAE](../methods/parallel_vae.md) for their relationship to distributed decoding. |

Do not combine memory-efficient replicated loading with transformer tensor
parallelism or ordinary FSDP. Supported PipeFusion layouts can use the two
memory-efficient flags for different components. Each stage loads its own
transformer blocks; FSDP loads selected replicated components.

Meta loading constructs modules without allocating their weight tensors.
SD3.5 does not support this loading path for PipeFusion stages.

Supported routes vary by model, including between related checkpoint variants.
The model's `capabilities`, `load_support`, and `settings` declarations in
[runner models](https://github.com/xdit-project/xDiT/tree/main/xfuser/model_executor/models/runner_models)
define supported loading paths. The runner rejects unsupported requests before
allocation. FSDP support alone does not establish support for memory-efficient loading.

```bash
xdit --model Qwen-Image --prompt "An isometric botanical library" \
    --ulysses_degree 4 --memory_efficient_replicated_load
```

For parameter sharding, start with `--fully_shard_degree 4` and a matching
four-rank layout, then add `--memory_efficient_sharding` if the model supports
that loading route. [Quantization](quantization.md) is an additional memory
tradeoff; [model capabilities](models.md) distinguish these features.
