# Unsupported in version 1

This document lists what version 1 does not support and how the harness detects it. Each item has a future-work
entry in [future_work/index.md](future_work/index.md).

## 1. Configurations

The workers check these conditions after they parse the run arguments and build the model. If one condition is
true, every case of the configuration gets UNSUPPORTED with the reason. The instance then ends, and B does not start.

| Feature | Detection | Reason |
|---|---|---|
| Hybrid attention schedule | `use_hybrid_attn_schedule` is set | The schedule selects the attention backend for each forward call through a global step counter. A replay advances the counter and runs with the backend of another step. |
| Hybrid GEMM schedule | `use_hybrid_gemm_schedule` is set. A YAML `gemm_config` with `hybrid_gemm_schedule` also sets it ([args.py:552-561](../../../xfuser/config/args.py#L552-L561)). | Same reason as the hybrid attention schedule, for the GEMM precision. |
| VSA | The `Spec` of the selected attention backend declares `Sparsity.VSA` | VSA changes its state for each step ([transformer_wan.py:476-477](../../../xfuser/model_executor/models/transformers/transformer_wan.py#L476-L477)). |
| PipeFusion | `pipefusion_parallel_degree` > 1 | Patch state is global and changes during the run. |
| Step caching | `cache_method` is set, or `use_teacache` or `use_fbcache` is set | A step cache keeps state across steps. |
| FSDP sharding | `fully_shard_degree` > 1 | B cannot safely convert FSDP flat parameters to the reference dtype after the load. FSDP rejects a mixture of dtypes in one unit ([sharding.py:395-402](../../../xfuser/core/distributed/sharding.py#L395-L402)). |

The first configuration has none of these conditions. It sets neither schedule flag and no `gemm_config`. Its
backend `aiter_mxfp6` is not a VSA backend. The schedules are set up only when the flags are on
([base_model.py:1525-1529](../../../xfuser/model_executor/models/runner_models/base_model.py#L1525-L1529)).

## 2. Single cases

These conditions make one case UNSUPPORTED. The other cases continue.

1. A listed component receives an fp8 or MX-format tensor. B cannot use it without dequantization, which needs the
   pairing of data and scales.
2. A listed component receives an argument that the serializer does not support ([files.md](files.md),
   section 7).

## 3. Not unsupported

1. `use_torch_compile` in the YAML entry: the harness forces it off. This is not a reason for UNSUPPORTED.
2. Models other than the first configuration: the harness runs them as they are and reports their failures.
