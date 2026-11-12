# Step caching

Select `--cache_method teacache`, `fbcache`, `dbcache`, or `pipefusion` only for a
model that declares support for that method. Install `xfuser[cache-dit]` for
Cache-DiT methods. The old `--use_teacache` and `--use_fbcache` selectors are deprecated.

Ordinary TeaCache/FBCache/DBCache reject pipeline parallelism.
For caching across PipeFusion stages, set `--cache_method pipefusion` and
`--pipefusion_parallel_degree` greater than 1. This mode does not support
multi-transformer cache adapters. Cache-DiT cache decisions also require
`--data_parallel_degree 1`. TeaCache requires `--tensor_parallel_degree 1`.
See the [PipeFusion composition guide](../methods/pipefusion.md#composition-and-validation-boundaries)
for the supported combinations.

Use a JSON object with `--cache_config` to override model cache presets. Cache-DiT accepts
`DBCacheConfig` fields; in-tree TeaCache/FBCache accept only
`residual_diff_threshold`. Some FBCache model presets use the Cache-DiT engine,
so accepted keys depend on the selected model's adapter. Compare output quality
on your workload because caching changes the computation.

```bash
xdit --model FLUX.1-dev --prompt "An isometric botanical library" \
    --ulysses_degree 4 --cache_method dbcache \
    --cache_config '{"residual_diff_threshold": 0.16}'
```

Begin with the model's cache preset before tuning the threshold. Compare output
quality and [timings](benchmarking.md) against a run without caching. For
supported methods by model, see [model capabilities](models.md).
