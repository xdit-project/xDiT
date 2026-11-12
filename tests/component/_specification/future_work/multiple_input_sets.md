# Future work: multiple input sets

This document is for coding agents and for humans. It describes the limitation of version 1, the design for
multiple input sets, and the challenges that are already known.

## 1. Limitation

1. Version 1 uses one input set: the run arguments of the configuration.
2. Thus each component sees only the shapes and values of that one run. A defect that occurs only with another
   resolution, another number of frames, or another prompt length stays hidden.

## 2. Design

1. An input set is a set of run-argument overrides on top of the configuration. Example:
   `{"height": 480, "width": 832, "num_frames": 49}`.
2. The test definition lists the sets.
3. Each instance runs an outer loop over the sets, after one init. The weights do not depend on the shapes, so the
   model loads only once.
4. For each set, each instance does a pipeline run, a capture and the replays, as in version 1.

## 3. Challenges

### 3.1 Preparation for each set

1. The runner prepares shape-dependent state before a pipeline run: `prepare_run`
   ([base_model.py:1449-1451](../../../../xfuser/model_executor/models/runner_models/base_model.py#L1449-L1451)).
   Today it sets the VAE tile overlap for the sample shape
   ([vae_manager.py:106-116](../../../../xfuser/model_executor/models/runner_models/vae_manager.py#L106-L116)).
2. Each instance must call `prepare_run` for each set, before the pipeline run and before the replays of that set.

### 3.2 Hidden context of the last shape

1. Parent forwards write shape-derived values into shared dictionaries. Examples in Wan: `thw`, `valid_kv_len` and
   `_sp_pad` ([index.md](index.md), section 2.9).
2. After the pipeline run of set k, these dictionaries hold the values of the last call of set k. A replay of a
   sub-component for set j (j not equal to k) would read the wrong values.
3. **Option: a stage-setting component.** The test definition names a component that the harness replays first,
   to put the right values into the dictionaries. Example: replay `pipe.transformer` of set j before a block of set
   j. This is optional and given by the user.
4. **Option: replays right after each pipeline run.** The harness replays the calls of set k before the pipeline
   run of set k + 1. Then the dictionaries hold the values of set k.
5. The proper fix is the explicit per-call context ([index.md](index.md), section 2.9, item 2.3).

### 3.3 A pipeline run and a fidelity check of B for each set

1. The fidelity check of B covers only the shapes of its own pipeline run. Thus B must run its pipeline for each set.
2. The fp32 pipeline run of B is slow. The run time grows with the number of sets.

### 3.4 Set definitions depend on the model

Valid values depend on the model and on the parallel layout:

1. `num_frames - 1` must be divisible by the temporal factor of the VAE
   ([pipeline_wan_i2v.py:75-79](../../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L75-L79)).
2. The Wan attention processor fixes the text context at 512 tokens.
3. Ulysses needs head counts that it can divide, and it pads the sequence for each rank.

Thus generic default sets are not possible. Each configuration must define its own sets.

### 3.5 Test IDs

1. Test IDs get a set dimension, for example `pipe.transformer@480p`.
2. The characters must stay valid for `-k` ([../component_paths.md](../component_paths.md), section 6). `@` is
   not valid in `-k`.

### 3.6 Data layout and re-runs

1. The configuration directory gets one level for each set.
2. The rules for fresh runs and reuse runs ([../files.md](../files.md), section 5) apply for each case and set.
3. The metrics file gets a field for the set.

### 3.7 Disk space and run time

1. The dumps grow with the number of sets.
2. The run time of A and B grows with the number of sets, mostly through the pipeline runs.

### 3.8 Compiled mode

With several shapes, compiled code recompiles ([compiled_mode.md](compiled_mode.md), section 4.5). The harness must
group the replays by shape and reset dynamo between the groups.

### 3.9 Shapes of sub-components only

A sub-component can get a new shape only through a new whole-model shape, because its parent derives its context
from the whole-model shape. Synthetic inputs for a sub-component alone need the explicit per-call context and
model-specific input factories (for example for RoPE tables of a new length). Arbitrary shape grids belong to
kernel-level tests.
