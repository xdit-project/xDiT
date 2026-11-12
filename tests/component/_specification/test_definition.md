# Test definition

This document describes how a test module defines its configurations.

## 1. Test modules and configurations

1. A test module covers many or all configurations of one xDiT model. This is a convention. The harness does not
   check it.
2. Test modules go directly into `xDiT/tests/component/`, as `test_<model>.py`.
3. The harness package goes into `xDiT/tests/component/harness/`.
4. Each configuration is one separate call of the harness: the configuration call (section 2). Configurations
   share settings only through plain Python values, for example one dictionary of components.
5. The test module calls `make_test` one time, with all its configurations. `make_test` makes the test function and
   its cases ([pytest_integration.md](pytest_integration.md), sections 2 and 3).
6. The definition is in Python for now. It can move to a YAML file later.
7. The harness makes the cases from the configuration calls. It must not build a model or start GPU work for this
   step.
8. **Errors in a definition.** The errors in section 3.1 and section 7, and the syntax errors of component paths
   ([component_paths.md](component_paths.md), section 1.3), raise an exception during collection.
   1. By default, pytest then runs no test of the session
      ([main.py:386-389](../../../../../venv/lib/python3.12/site-packages/_pytest/main.py#L386-L389)).
   2. The benefit: the user sees the error at once, before any GPU run.
   3. The cost: one wrong configuration stops all configurations until the user corrects it.

## 2. The configuration call

### 2.1 Example

The parameters are the requirement. The two configurations share their settings through plain Python values.

```python
import harness

BASE_CONFIG = "/app/.ci/benchmark_configs/wan2_2.yaml"
FEW_STEPS = {"args": {"num_inference_steps": 5}}
REFERENCE = harness.Reference(
    dtype="float32",
    self_attention_backend=...,  # open question, see instance_b.md, section 3.2
    cross_attention_backend="sdpa_math",
)
COMPONENTS = {
    "pipe.transformer": {"[0]": harness.Tolerance(max_abs=0.5, rel_l2=0.05)},
    "pipe.transformer.blocks[0]": {"": harness.Tolerance()},
    "pipe.transformer_2:5": {"[0]": harness.Tolerance()},
    "pipe.vae.decode": {"[0]": harness.Tolerance()},
}

QUANT_ATTN = harness.Configuration(
    yaml_path=BASE_CONFIG,
    entry="wan2_2.quantgemm_quantattn.gfx950",
    overrides=FEW_STEPS,
    reference=REFERENCE,
    components=COMPONENTS,
)

BF16_ATTN = harness.Configuration(
    yaml_path=BASE_CONFIG,
    entry="wan2_2.quantgemm_quantattn_bf16.gfx950",
    overrides=FEW_STEPS,
    reference=REFERENCE,
    expect_deterministic_a=True,
    expect_deterministic_b=True,
    components=COMPONENTS,
)

test_component = harness.make_test(QUANT_ATTN, BF16_ATTN)
```

### 2.2 Parameters

1. `yaml_path`: the path of a CI benchmark YAML file. Optional. The first test module passes the literal
   `/app/.ci/benchmark_configs/wan2_2.yaml`.
2. `entry`: the `name` of one entry in that file. Optional. `yaml_path` and `entry` must be given together or not
   at all.
3. `overrides`: a mapping with the structure of a YAML entry. It changes the entry. See section 3.1. The overrides
   apply to A and to B.
4. `reference`: the parameters of B. See section 4.
5. `expect_deterministic_a` and `expect_deterministic_b`: the determinism expectation of each instance. The
   default is `True`. `False` is the determinism override of that instance. See [checks.md](checks.md),
   section 3.
6. `components`: a mapping from a component path to the tolerances of its outputs. Each path is one case. See
   [component_paths.md](component_paths.md) and section 5.
7. `deadlines`: optional changes to the default deadlines. See section 6.

The configuration call has no `name` parameter. The configuration name is the `name` field of the merged entry
(section 7).

## 3. Run arguments

### 3.1 The merged entry and the flags

1. **The merged entry.**
   1. With `yaml_path` and `entry`, a helper reads the YAML file and selects the entry whose `name` is `entry`. An
      `entry` that is not in the file is an error.
   2. Without `yaml_path` and `entry`, the helper starts from an empty entry. The merged entry then comes from
      `overrides` alone.
   3. The helper applies `overrides` to the entry. The result is the merged entry.
2. **Fields.** The fields of an entry are the fields of the `Experiment` class of CI
   ([run.py:85-93](../../../../.ci/run.py#L85-L93)): `name`, `tags`, `runner`, `model`, `args`, `entrypoint`,
   `num_gpus` and `revision`. Another field in the YAML entry or in `overrides` is an error. CI also rejects other
   fields ([run.py:622](../../../../.ci/run.py#L622-L622)).
3. **Merge rules.**
   1. `args` is merged key by key. A key in `overrides["args"]` replaces the same key of the entry. The value `None`
      removes the key.
   2. Each other field in `overrides` replaces the field of the entry. The value `None` removes the field.
4. **Required fields.** The merged entry must have `name` and `model`. Without `args`, there are no other flags.
5. **Use of the fields.**
   1. `name`: the configuration name (section 7).
   2. `model`: the flag `--model`.
   3. `args`: the other flags (item 6).
   4. `runner`: if present, it must be `xdit`. Another runner is an error, because the harness runs only the xDiT
      runner.
   5. `num_gpus`: the number of workers. CI gives it to the `xdit` runner as `--nproc_per_node`
      ([run.py:540-545](../../../../.ci/run.py#L540-L545)). Without it, the launcher computes the number as the xDiT
      CLI does ([processes.md](processes.md), section 1).
   6. `tags`: no effect in version 1. Hardware tags are future work
      ([future_work/index.md](future_work/index.md), section 2.6).
   7. `revision` and `entrypoint`: no effect. CI uses `revision` only to download the model
      ([run.py:257-263](../../../../.ci/run.py#L257-L263)), and it does not use `entrypoint` with the `xdit` runner
      ([run.py:540-542](../../../../.ci/run.py#L540-L542)).
6. **Flags.** The helper builds the same command-line flags that CI builds in
   [run.py:581-608](../../../../.ci/run.py#L581-L608):
   1. `--model <model>` first;
   2. a boolean `True` gives the flag alone, and `False` gives nothing. Thus the value `False` in
      `overrides["args"]` drops a boolean flag;
   3. a list gives the flag followed by each item;
   4. a mapping gives the flag followed by one JSON string;
   5. any other value gives the flag followed by `str(value)`.
7. The worker parses the flags with `xFuserArgs.add_runner_args(parser).parse_args(flags)`. This is the parser that
   the xDiT runner uses in [runner.py:148](../../../xfuser/runner.py#L148-L148).
8. All errors in this section are definition errors (section 1, item 8).

### 3.2 Order of the layers

1. The `args` of the merged entry: the YAML entry with the `overrides` of the configuration (section 3.1).
2. The forced overrides of the harness, in both instances:
   1. `warmup_calls` 0;
   2. `num_iterations` 1;
   3. `use_torch_compile` off.
3. The reference profile, in B only. See [instance_b.md](instance_b.md), section 2.

The forced overrides remove the extra runs of the run loop. The run loop runs the pipeline code of the runner once
for each warmup call and once for each iteration
([base_model.py:1239-1243](../../../xfuser/model_executor/models/runner_models/base_model.py#L1239-L1243) and
[base_model.py:1262](../../../xfuser/model_executor/models/runner_models/base_model.py#L1262-L1262)). The pipeline
code of the runner can still call a pipeline more than once
([component_paths.md](component_paths.md), section 4.4).

### 3.3 The step override of the first configuration

The below explains `num_inference_steps` set to 5 in the example. It is specific to Wan2.2 model, so another
model might have a different appropriate override value, but the principle remains.

1. The first configuration uses `num_inference_steps` 5 instead of 40, through
   `overrides={"args": {"num_inference_steps": 5}}`. This keeps the run time of A and B low.
2. The checkpoint of the first configuration has a boundary timestep of 900 (`boundary_ratio` 0.9). The pipeline
   calls `transformer` for steps with `t >= boundary_timestep` and `transformer_2` for the other steps
   ([pipeline_wan_i2v.py:185-192](../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L185-L192)).
3. With 5 steps, `transformer` gets 2 steps and `transformer_2` gets 3 steps. This is true for flow shift 3 (the
   checkpoint) and for flow shift 5 (the runner default). The UniPC scheduler of the checkpoint gives these
   timesteps:
   1. shift 3: 999, 923, 818, 667, 429;
   2. shift 5: 999, 952, 882, 769, 556.
4. Without classifier-free-guidance parallelism, the pipeline calls the transformer two times in each step: the
   conditional call at
   [pipeline_wan_i2v.py:207](../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L207-L207) and the
   unconditional call at
   [pipeline_wan_i2v.py:226](../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L226-L226). Thus
   `pipe.transformer` has 4 calls and `pipe.transformer_2` has 6 calls.
5. Fewer steps can remove a component from the run. For example, with 3 steps and shift 3, `transformer` gets only
   1 step. A case whose selected calls do not occur fails (see [checks.md](checks.md), section 4).

### 3.4 Shapes

The overrides must not make the shapes of A and B different. A and B use the same run arguments, so this is true
when the reference profile does not change a shape argument.

## 4. Parameters of B

1. `dtype`: the reference dtype. The default is `"float32"`.
2. `self_attention_backend`: the attention backend of B for self-attention. The value is an `--attention_backend`
   name. The default is an open question (see [instance_b.md](instance_b.md), section 3.2).
3. `cross_attention_backend`: the attention backend of B for cross-attention. The default is `sdpa_math`.
4. xDiT parses a backend name as the member name of `AttentionBackendType`, without regard to case. See
   `_parse_attention_backend` in
   [base_model.py](../../../xfuser/model_executor/models/runner_models/base_model.py).
5. The validation of these parameters is in [instance_b.md](instance_b.md), section 3.

## 5. Tolerances and output names

### 5.1 Requirements

1. For each component path, the definition must list every tensor output of the component.
2. Each output has a `Tolerance` with two metrics: `max_abs` and `rel_l2`.
3. A metric without a value gets the default tolerance. The default is a very small value, `1e-10`, to cause a test
   failure, helping to come up with initial reasonable tolerances.
4. One tolerance applies to all ranks.
5. The test function checks the names in both directions:
   1. A tensor output without a tolerance is a failure.
   2. A tolerance for a name that is not an output is a failure.
   3. The failure message lists the actual output names.
6. Non-tensor outputs need no entry. They must be exactly equal in A and B.
7. The test function reports the cosine similarity, but it does not check it.

### 5.2 Output names

1. An output name is a path relative to the return value of the component.
2. A bare tensor has the empty name `""`.
3. `[i]` selects item `i` of a tuple or a list.
4. `.name` selects an attribute, for example `.sample` of a diffusers output object.
5. `["key"]` selects the item `key` of a mapping (Not confirmed).
6. Nested containers are flattened to their tensor leaves. The name of a leaf joins the parts, for example
   `[0].sample`.
7. Examples:
   1. The Wan transformer, called with `return_dict=False`, returns a tuple. Its tensor output is `[0]`.
   2. A transformer block that returns one tensor has the output `""`.

## 6. Deadlines

1. Default deadlines:

   | Instance | Init | Pipeline run | Each replayed call |
   |---|---|---|---|
   | A | 10 minutes | 20 minutes | 2 minutes |
   | B | 15 minutes | 40 minutes | 8 minutes |

2. The `deadlines` parameter can change each value.
3. [processes.md](processes.md), section 4 tells how the launcher applies the deadlines.

## 7. Configuration name

1. **Source.** The configuration name is the `name` field of the merged entry (section 3.1).
   1. The default is thus `entry`.
   2. `overrides={"name": ...}` sets another name. Example: two configurations that use the same entry with
      different parameters of B need different names.
   3. Without `yaml_path` and `entry`, `overrides` must set `name`.
2. **Use.** The name selects the data directory `_data/<name>/`, the lock file `_locks/<name>.lock` and the
   metrics file `_metrics/<name>.jsonl` (see [files.md](files.md)). It is also the first part of each case ID
   ([component_paths.md](component_paths.md), section 6).
3. **Characters.** A name contains only letters, digits, `_`, `.` and `-`. Thus it is a valid file name, `-k`
   accepts it, and it contains no `:`. The entry names in the CI benchmark files meet this rule.
4. **Errors during collection.** The harness raises an error if the merged entry has no name (i.e. either name
   is absent, or set to None, or not a string, or an empty string), if the name contains
   other characters, or if the name is not unique (item 5). See section 1, item 8.
5. **Unique names.** Two configurations with the same name would share the data directory and the metrics file.
   1. `make_test` registers the name of each of its configurations in one registry for the whole pytest process,
      together with the test module that defines the configuration.
   2. If a name is already in the registry, `make_test` raises an exception during collection. The message names
      both test modules, so the user can find both definitions.
   3. pytest imports all collected test modules before it runs a test, and it deselects tests only after the import.
      Thus the check covers all configurations that the session collects, also the configurations that `-k`
      deselects.
   4. The check does not see configurations in test modules that the session does not import. Unique names across
      such modules and across sessions are the responsibility of the user.
   5. A separate check of the configuration fixture stops a configuration that starts a second time in one session
      ([pytest_integration.md](pytest_integration.md), section 3, item 2).
