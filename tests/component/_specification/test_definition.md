# Test definition

This document describes how a test module defines one configuration.

## 1. One test module for each configuration

1. Each configuration is one test module, `xDiT/tests/component/test_<something>.py`.
2. Test modules go directly into `xDiT/tests/component/`.
3. The harness package goes into `xDiT/tests/component/harness/`.
4. The test module defines the configuration with one call of the harness. The definition is in Python for now. It
   can move to a YAML file later.
5. The harness makes the cases from this call. It must not build a model or start GPU work for this step.

## 2. The configuration call

### 2.1 Example

The names in this example are a proposal (Not confirmed). The parameters are the requirement.

```python
import harness

CONFIG = harness.Configuration(
    yaml_path="/app/.ci/benchmark_configs/wan2_2.yaml",
    entry="wan2_2.quantgemm_quantattn.gfx950",
    overrides={"num_inference_steps": 5},
    reference=harness.Reference(
        dtype="float32",
        self_attention_backend=...,  # open question, see instance_b.md, section 3.2
        cross_attention_backend="sdpa_math",
    ),
    expect_deterministic_a=True,
    expect_deterministic_b=True,
    components={
        "pipe.transformer": {"[0]": harness.Tolerance(max_abs=0.5, rel_l2=0.05)},
        "pipe.transformer.blocks[0]": {"": harness.Tolerance()},
        "pipe.transformer_2:5": {"[0]": harness.Tolerance()},
        "pipe.vae.decode": {"[0]": harness.Tolerance()},
    },
)

test_component = CONFIG.make_test()
```

### 2.2 Parameters

1. `yaml_path`: the path of a CI benchmark YAML file. The first test module passes the literal
   `/app/.ci/benchmark_configs/wan2_2.yaml`.
2. `entry`: the `name` of one entry in that file.
3. `overrides`: a mapping that changes the `args` of the entry. See section 3. The overrides apply to A and to B.
4. `reference`: the parameters of B. See section 4.
5. `expect_deterministic_a` and `expect_deterministic_b`: the determinism expectation of each instance. The
   default is `True`. `False` is the determinism override of that instance. See [checks.md](checks.md),
   section 3.
6. `components`: a mapping from a component path to the tolerances of its outputs. Each path is one case. See
   [component_paths.md](component_paths.md) and section 5.
7. `name`: the name of the configuration. The default is the stem of the test module file name. See section 7.
8. `deadlines`: optional changes to the default deadlines. See section 6.

## 3. Run arguments

### 3.1 The YAML helper

1. A helper reads the YAML file and selects the entry by its `name`.
2. The helper merges `overrides` into the `args` of the entry:
   1. a key in `overrides` replaces the same key in `args`;
   2. the value `None` removes the key;
   3. the value `False` drops a boolean flag.
3. The helper builds the same command-line flags that CI builds in
   [run.py:581-608](../../../../.ci/run.py#L581-L608):
   1. `--model <model>` first;
   2. a boolean `True` gives the flag alone, and `False` gives nothing;
   3. a list gives the flag followed by each item;
   4. a mapping gives the flag followed by one JSON string;
   5. any other value gives the flag followed by `str(value)`.
4. The worker parses the flags with `xFuserArgs.add_runner_args(parser).parse_args(flags)`. This is the parser that
   the xDiT runner uses in [runner.py:148](../../../xfuser/runner.py#L148-L148).

### 3.2 Order of the layers

1. The `args` of the YAML entry.
2. The `overrides` of the test module.
3. The forced overrides of the harness, in both instances:
   1. `warmup_calls` 0;
   2. `num_iterations` 1;
   3. `use_torch_compile` off.
4. The reference profile, in B only. See [instance_b.md](instance_b.md), section 2.

The forced overrides make one runner run call the pipeline exactly once. The runner calls the pipeline once for
each warmup call and once for each iteration
([base_model.py:1239-1243](../../../xfuser/model_executor/models/runner_models/base_model.py#L1239-L1243) and
[base_model.py:1262](../../../xfuser/model_executor/models/runner_models/base_model.py#L1262-L1262)).

### 3.3 The step override of the first configuration

1. The first configuration uses `num_inference_steps` 5 instead of 40. This keeps the run time of A and B low.
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

1. For each component, the definition must list every tensor output of the component.
2. Each output has a `Tolerance` with two metrics: `max_abs` and `rel_l2`.
3. A metric without a value gets the default tolerance. The default is a very large value, `1e30` (the value is
   Not confirmed). The user sets real values later.
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

1. The default name is the stem of the test module file name.
2. The `name` parameter can set another name.
3. The name selects the data directory `_data/<name>/` and the metrics file `_metrics/<name>.jsonl`. See
   [files.md](files.md).
4. During collection, the harness must raise an error in these cases:
   1. two test modules in the same directory get the same name;
   2. a name is not a plain file name, for example because it contains `/`.
