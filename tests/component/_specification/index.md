# xDiT component test harness: specification

This specification describes version 1 of a test harness for components of xDiT diffusion models. Start here. The
child documents in section 8 give the details.

## 1. Priority over the testing skill

Where this specification conflicts with the testing skill in
[SKILL.md](../../../.github/skills/testing/SKILL.md), this specification takes priority.

## 2. Purpose

1. The harness tests individual components of xDiT diffusion models. Examples of components are a transformer,
   one transformer block, one attention module, and the decode method of a VAE.
2. The harness tests each component with the configuration and the weights that the xDiT runner uses in
   production.
3. The harness takes each component from the model that the xDiT runner builds. The harness never builds a
   component itself. Thus the tests stay valid when xDiT changes.
4. Two instances run each component:
   1. Instance A runs the component as production runs it.
   2. Instance B is an eager reference. It runs the component in the reference dtype. The default reference dtype
      is fp32.
5. The harness compares the outputs of A and B against tolerances.
6. A test fails for a deviation, a crash, a hang, nondeterminism, or any other error.

## 3. Version 1

### 3.1 Building principle

1. The idea of this harness rests on many risky assumptions. Version 1 must validate these assumptions first.
2. Thus version 1 is a simple solution. It must run end to end for the best case. It does not handle special
   cases.
3. Use this principle for every decision that this specification does not make.
4. Rejected alternatives that are worth an implementation are in [future_work/index.md](future_work/index.md).

### 3.2 Scope

1. **Eager mode only.** The harness forces `use_torch_compile` off in both instances. Compiled mode is future work.
2. **The first configuration must run end to end.** It is the entry `wan2_2.quantgemm_quantattn.gfx950` in
   [wan2_2.yaml:211-229](../../../../.ci/benchmark_configs/wan2_2.yaml#L211-L229):
   1. model `Wan-AI/Wan2.2-I2V-A14B-Diffusers`;
   2. `ulysses_degree` 8 and `use_parallel_vae`;
   3. `attention_backend` `aiter_mxfp6` and `cross_attention_backend` `aiter_bf16`;
   4. `gemm_quantization` `low=fp4,high=fp6`;
   5. 720 × 1280 pixels, 81 frames, 40 inference steps;
   6. `memory_efficient_replicated_load`.

   The entry also sets `use_torch_compile`. The harness ignores this setting (item 1).
3. **xDiT changes.** Version 1 changes xDiT only where the first configuration needs a change. See
   [xdit_changes.md](xdit_changes.md).
4. **Other models.** The harness runs other models as they are. It reports their failures. Version 1 does not fix
   them.
5. **Hardware.** Version 1 is built on an MI355X machine (gfx950). Later, the harness must also work on at least
   MI300X (gfx942).
6. **One input set.** The run arguments of the configuration are the only input set.

### 3.3 Not in version 1

1. Compiled mode ([future_work/compiled_mode.md](future_work/compiled_mode.md)).
2. Instance B1, a reference at world size 1 ([future_work/index.md](future_work/index.md), section 2.2).
3. Multiple input sets ([future_work/multiple_input_sets.md](future_work/multiple_input_sets.md)).
4. Recovery from crashes and hangs. Version 1 detects and attributes them, but it does not recover.
5. The configurations and features in [unsupported.md](unsupported.md).
6. pytest markers and CI integration.

## 4. Terms

1. **Runner model object.** The `xFuserModel` object that `xFuserModelRunner` creates. It is `runner.model` in
   [runner.py:57-66](../../../xfuser/runner.py#L57-L66). Component paths start at this object.
2. **Configuration.** One test module. It holds one CI YAML entry, the overrides, the parameters of B, the
   component list, and the tolerances.
3. **Group.** All cases of one configuration.
4. **Component path.** A string that leads from the runner model object to a component, for example
   `pipe.transformer.blocks[0]`. See [component_paths.md](component_paths.md).
5. **Case.** One listed component path. A case is one pytest test.
6. **Call.** One execution of a listed component in a pipeline run.
7. **Call address.** The tuple of call counters that identifies one call. See
   [component_paths.md](component_paths.md), section 5.
8. **Capture.** The record of the inputs and the output of one call during a pipeline run.
9. **Replay.** A new execution of a component with the captured inputs, outside the pipeline run.
10. **Fidelity check.** A bitwise comparison of a replay output with the captured output of the same instance.
11. **Determinism check.** A bitwise comparison of two replay outputs of the same instance.
12. **Determinism override.** A per-instance setting that permits nondeterminism. See
    [checks.md](checks.md), section 3.
13. **Reference dtype.** The dtype of B. The default is fp32.
14. **Launcher.** The process that runs one instance. It starts torchrun and survives crashes and hangs of the
    workers. See [processes.md](processes.md).
15. **Worker.** One rank of an instance.
16. **State file.** The file that holds the status and the file paths of one case. See [files.md](files.md).
17. **Metrics file.** The file that collects metric values across runs. See [files.md](files.md), section 9.

## 5. End-to-end flow

1. **Collection.** pytest imports the test module. The module defines the configuration. The harness makes one
   case for each component path. It does not build a model for this step.
2. **Group fixture.** The first selected test of the module starts the module-scoped group fixture. The fixture
   does these steps:
   1. It decides between a fresh run and a reuse of earlier data ([files.md](files.md), section 5).
   2. For a fresh run, it runs the launcher of A.
   3. If A ended clean, it runs the launcher of B.
3. **Instance A** ([instance_a.md](instance_a.md)):
   1. builds and initializes the model as the xDiT runner does;
   2. resolves the component paths;
   3. runs the pipeline once and captures the calls of the listed components;
   4. replays the captured calls for the fidelity and determinism checks;
   5. writes the case state files.
4. **Instance B** ([instance_b.md](instance_b.md)):
   1. builds and initializes the model with the reference profile;
   2. converts all components to the reference dtype;
   3. runs the pipeline once, captures its own calls, and replays them for its fidelity check;
   4. replays the inputs that A captured, at least two times, and dumps the outputs.
5. **Test function** ([checks.md](checks.md)). For each case, it does these steps:
   1. It reads the case state.
   2. It stops at once for a status without data, for nondeterminism without an override, and for a replay
      mismatch.
   3. It computes the metrics for each call, output and rank, and checks them against the tolerances.
   4. It writes the metrics to the metrics file.
   5. It deletes the files of the case if the case passed.

## 6. Open questions

1. **Default self-attention backend of B (Not confirmed).** Version 1 has no confirmed default. See
   [instance_b.md](instance_b.md), section 3.2.
2. Other statements marked "Not confirmed" in the child documents are proposals. The user did not confirm them.

## 7. Conventions in this specification

1. "Must" states a requirement.
2. Code references are links relative to each document. The line numbers apply to the xDiT tree at commit
   `0be63d4`.
3. "Not confirmed" marks a statement that the user did not confirm.

## 8. Document map

1. [test_definition.md](test_definition.md): the configuration call, the YAML helper, overrides, the parameters
   of B, tolerances, output names, deadlines.
2. [component_paths.md](component_paths.md): path syntax, path elements, resolution, hooks, call addresses, case
   IDs.
3. [instance_a.md](instance_a.md): the flow of A, capture, replays, fidelity and determinism.
4. [instance_b.md](instance_b.md): the reference profile, attention backends, conversion, the pipeline run of B,
   replays of the inputs of A.
5. [checks.md](checks.md): the checks in the test function and the metric definitions.
6. [statuses.md](statuses.md): all statuses and their pytest outcomes.
7. [processes.md](processes.md): launchers, workers, deadlines, progress records, crash and hang attribution.
8. [pytest_integration.md](pytest_integration.md): files, fixtures, test functions, selection, outcomes.
9. [files.md](files.md): path resolution, the data layout, state files, re-runs, safety guards, serialization,
   the metrics file.
10. [unsupported.md](unsupported.md): what version 1 does not support and how the harness detects it.
11. [xdit_changes.md](xdit_changes.md): the changes that version 1 needs in xDiT, and the xDiT interfaces that the
    harness uses.
12. [prototype_changes.md](prototype_changes.md): the changes that the `pytest_integration` prototype needs.
13. [future_work/index.md](future_work/index.md): all future work.
