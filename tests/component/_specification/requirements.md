# Requirements and use cases of the component test harness

This document describes the use cases and the requirements for the xDiT component testing harness.

## Use cases

**UC1. Check that a component works correctly.** A developer asks: "Does this component of this model
configuration work correctly?" The harness runs the component as production runs it (instance A) and as an
eager reference in potentially extended precision (instance B). It then compares the outputs against tolerances.
*Rationale:* some operations of xDiT diffusion models can be flaky (produce wrong numerics, or cause hangs or
crashes). Component-wise testing could help with QA of the implementation.

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang. The report names the instance and the case. At least for a deviation or nondeterminism, it also names the calls.

**UC3. Test the configurations of the CI benchmark files.** Tests run on selected components
of a model that is configured by an entry in a CI benchmark YAML file, or directly.
*Rationale:* the user specifies model configurations in the same way as the benchmark configurations.

**UC4. Cover many configurations of one model in one test module.**
*Rationale:* there are hundreds of configurations in total, and most of them are very similar.

**UC5. Test sub-components and selected calls.** Examples are a transformer block, an attention module, the
VAE decode method, and one specific call of a transformer.

**UC6. Tune tolerances over time.** A test reports its metrics in each run. The metrics history shows how the deviations change while the implementation changes.

**UC7. Run the tests again after failures.** By default, each run starts from the beginning and runs A and B again. On the user's request, a new run compares the stored data again, for example with new tolerances, and does not run A and B again. Also with such a request, after all cases pass, the next run starts from the beginning automatically. This behavior, however, can be modified if test functions executions on the same data without rerunning A and B is needed.

**UC8. Select single cases.** The user selects cases with pytest, for example with `-k` or a node ID. 

**UC9. Keep the tests valid while xDiT changes.**
*Rationale:* xDiT changes fast, and its components get new augmentations all the time. Hard-coded tests become
obsolete.

**UC10. Accept nondeterminism that is valid.** An override of one test changes the result of nondeterminism in that test from a failure to a warning. The determinism checks of all other tests stay active. B should have its own override.
*Rationale:* some operations are validly nondeterministic. A model can have one nondeterministic component
among many, and that component must be testable. Instance B generally uses different kernels that might have
other determinism expectations.

**UC11. Later use cases.** Compiled mode, a reference at world size 1 (B1), several input sets, MI300X and other than this machine hardware, and benchmark and determinism scripts that use the foundation of the harness.

## Requirements

### High-level requirements

**HL10.** The harness tests individual components of xDiT diffusion models, for example a VAE, a transformer, a
transformer block, an attention module, a text encoder, or a method such as the VAE decode. It uses exactly
the configuration and the weights that the xDiT runner uses for a model configuration.

**HL15.** Support for pytest `-k` argument is limited. The user is responsible to not request a rerun (`--xdit_allow_rerun`, PI20) after they change `-k` value (including removing it) or node ID to a selection that wasn't covered by previous fresh run. The harness doesn't track scope of different runs and doesn't warn if certain promised behavior may not work correctly due to an incompatible value of `-k` in a new run/rerun.

**HL20.** The harness takes each component from the model that the xDiT runner builds. It does not extract,
hard-code or build a component itself.
*Rationale:* the setup of a component depends on the model configuration. For example, the VAE of Wan2.2 can
be the VAE of `diffusers`, a DistVAE, or something else. Also, xDiT changes fast (UC9).

**HL30.** The details of a configuration are private to the configuration: Ulysses, ring, classifier-free
guidance, quantized GEMMs and others. The tests do not depend on them. The tests are generic, so that they can
run every configuration.

**HL40.** The reference is the same xDiT component, built by the xDiT runner, run potentially
with an extended precision (fp32 by default). It is not a `diffusers` implementation.

**HL50.** One test answers one question: "Does this component, in this regime, with these inputs, work
correctly?" A failed check belongs to one test, so only that test fails.
A misattribution of failure due to a crash or a hang is an accepted limitation of version 1.

**HL60.** A test fails for a deviation beyond its tolerance, a crash, a hang, nondeterminism, or any other
detectable issue. "Flaky" includes a crash and a hang.

**HL70.** **Working tests first.** Test coverage has priority over correct reports, for example a correct call identifier. A wrong report is acceptable as a documented limitation, possibly with a corresponding future-work item. A deviation from some requirements that a working test needs is permitted with a due documentation.

**HL80.** **Traceability and debuggability.** Each rank executing a model or its components should save a structured log of all events and situations related to execution decisions and defined by this document.

**HL90.** Where possible, xDiT gets a refactor that makes components testable, instead of a hack around hidden
coupling. Examples: compile code only compiles, and the warmup is not part of the compile function.
In version 1, xDiT changes only where the first configuration needs a change.

**HL100.** Tests and benchmarks are separate from the entry point on. Benchmark scripts and determinism-check scripts come later. They use the same foundation that the tests use to load and build a model and to get its components. Version 1 plans nothing for benchmarks, but it identifies the code that benchmarks will use.

**HL110.** One model load and initialization serves the tests of many, preferably all, components.
A process that loads and initializes a model, and dumps components execution data for testing later
is called "an instance". There are two instances: an instance A works with the model to test, and
an instance B works with the same model as A, but configured to perform computations in a higher
precision to obtain reference values to compare against.
*Rationale:* it may take 5–10 minutes to fully load and initialize a model, so we have to perform
test data acquisition in batches: one batch gathers data to test (A), and the other batch gather
reference data (B).

**HL120.** Multi-node model execution isn't supported, but single node multi-GPU is supported, hence a term "rank" is an alias to "local rank". A term "worker" is an alias to "local rank" too. An instance has N ranks with the numbers 0 to N-1. N is the number of processes that torchrun starts for the instance. The configuration fixture gives N to the launcher as a parameter, and N is the same for A and B (IB10). Every part of the harness takes the set of ranks from N and never infers it from the files that exist. For the pytest part, a rank log that does not exist is a log without records.

**HL130.** The pytest prototype integration in [_prototypes/pytest_integration/](../_prototypes/pytest_integration/) shows how a set of tests can share two expensive processes, select single cases, show live output, and leave no processes behind. The harness takes the parts that fit and changes them as this document requires to adapt to the requirements. The prototype is an implementation hint, not a binding rule, but the final implementation must have no weaker child process handling guarantees and child process console output visibility, than the prototype.

**HL140.** The harness itself must have a test set, ensuring the harness adheres to the requirements. Most important tests are (not exhaustive list):
- random rank fails fidelity check, or crashes, or hangs - correctly changes pytest outcomes for all cases of the configuration
- all possible values for logged statuses in rank log files correctly reflected by pytest outcomes

#### Statuses

**HLS10.** A component path in the list of components to test for a configuration (TD10) identifies a single component, for which a test case is created, and the component must be executed (called by the model) at least once during a model/pipeline execution. Statues are written in CAPITAL_LETTERS and carry information necessary for proper conversion of behavior observed during an attempt to run instances into pytest outcomes, while giving a user enough information for debugging.

**HLS15.** Statuses are the nodes of exactly one directed acyclic graph. The graph describes how the knowledge about a call, a case, or an instance grows. HLS17 is the normative list of the statuses and of the permitted transitions. A status has one of three scopes:
1. a component call on a rank: ABOUT_TO_START, DUMPED, UNSUPPORTED, FIDELITY_SKIPPED, PRE_FIDELITY, FIDELITY_PASSED, FIDELITY_FAILED, PRE_REPLAY, REPLAYED, and CRASHED or HANGED if the instance ended while the call was open (HLS16);
2. a case on a rank: UNRESOLVABLE_PATH, NOT_REACHED, CASE_CALLS_MISMATCH, FAILED_TO_START, UNSUPPORTED;
3. an instance, as one rank log shows it: UNSUPPORTED, CRASHED, HANGED. An instance-level status says only that the event happened. It never attributes the event to a case or a call, and it never gives a pytest outcome by itself. It is the reason for the case statuses that the launcher derives from it (TD120).

**TODO for an agent:** ALWAYS verify consistency of HLS15 with the other parts of the document!

**HLS16.** The last status of a call in a rank log puts the call into exactly one class:
1. *open*: ABOUT_TO_START, PRE_FIDELITY, PRE_REPLAY. The component can be executing.
2. *settled*: no step of the call is due. In A and in B: UNSUPPORTED, CRASHED, HANGED. In A only: FIDELITY_PASSED, FIDELITY_FAILED, FIDELITY_SKIPPED. In B only: REPLAYED.
3. *pending*: every other last status, that is DUMPED in A and in B, and FIDELITY_PASSED, FIDELITY_FAILED, or FIDELITY_SKIPPED in B. A step of the call is due and has not started.

**HLS17.** The table below is the normative list of the statuses and of the permitted transitions. "Previous" is the last status of the same call, or the records of the same case, before the new record. "Class" is the HLS16 class.

| # | Status | Scope | Writer | Written when | Previous | Class | Contribution to the outcome of the case |
|---|---|---|---|---|---|---|---|
| 1 | ABOUT_TO_START | call | rank | phase=run, the hook fires (RL60 step 1) | none | open | Cannot be the last status after TD120. If it is, the case cannot pass |
| 2 | DUMPED | call | rank | phase=run, inputs and outputs are saved (RL60 step 5) | ABOUT_TO_START | pending | Counts for TD115 and PM20. As the last status: the case cannot pass; row 14 or 15 gives the reason |
| 3 | UNSUPPORTED | call | rank | The rank cannot dump the inputs of the call (RL60 step 2, before the call) or its outputs (RL60 step 4, after the call). The rank writes the record when the call returns. Or: the pipeline run ended and the call did not return normally (RL80 item 4) | ABOUT_TO_START | settled | The case cannot pass |
| 4 | FIDELITY_SKIPPED | call | rank | Fidelity stage, override flag set (RL100 step 1) | DUMPED | A: settled, B: pending | As the last status in A: does not cause a failure from A. As the last status in B: the case cannot pass, because the replay of A's inputs is missing. PM80 warning in both |
| 5 | PRE_FIDELITY | call | rank | Fidelity stage (RL100 step 3) | DUMPED | open | As row 1 |
| 6 | FIDELITY_PASSED | call | rank | RL100 step 7 | PRE_FIDELITY | A: settled, B: pending | As the last status in A: does not cause a failure from A. As the last status in B: the case cannot pass, because the replay of A's inputs is missing; the reason is row 14 or 15, or the call does not exist in A (IB10 item 6) |
| 7 | FIDELITY_FAILED | call | rank | RL100 step 7 | PRE_FIDELITY | A: settled, B: pending | The case cannot pass (IA60), in A and in B, also when REPLAYED follows in B |
| 8 | PRE_REPLAY | call | rank of B | Replay of A's inputs (RL100 step 3) | FIDELITY_PASSED, FIDELITY_FAILED, FIDELITY_SKIPPED | open | As row 1 |
| 9 | REPLAYED | call | rank of B | RL100 step 8 | PRE_REPLAY | settled | As the last status in B: does not cause a failure from B, if the call has FIDELITY_PASSED or FIDELITY_SKIPPED (PM40) |
| 10 | CRASHED | call | launcher | TD120 item 2.2, the instance crashed | ABOUT_TO_START, PRE_FIDELITY, PRE_REPLAY | settled | The case cannot pass |
| 11 | HANGED | call | launcher | TD120 item 2.2, the launcher killed the instance for a missed deadline | as row 10 | settled | The case cannot pass |
| 12 | UNRESOLVABLE_PATH | case | rank | Start of phase=run, path resolution (RL60, CP120, CP140) | none | - | The case has no call on this rank, so the case cannot pass |
| 13 | NOT_REACHED | case | rank | The pipeline run ended and the case had no call (RL80) | none | - | As row 12 |
| 14 | CASE_CALLS_MISMATCH | case | rank, the same on all ranks | After the pipeline run, list comparison (RL80) | Any record of the case; no call of the case is open | calls keep their class and are not replayed | The case cannot pass |
| 15 | FAILED_TO_START | case | launcher | TD120 item 2.3 | no record, or call records | - | The case cannot pass |
| 16 | UNSUPPORTED | case | launcher, in all rank logs | TD120 item 1 | none | - | The case cannot pass |
| 17 | UNSUPPORTED | instance | rank | phase=initialization (RL40) | - | - | None directly. Cause of row 16 |
| 18 | CRASHED | instance | launcher, in all rank logs | TD120 item 2.1 | - | - | None directly. Reason for rows 10 and 15 |
| 19 | HANGED | instance | launcher, in all rank logs | TD120 item 2.1 | - | - | None directly. Reason for rows 11 and 15 |

The last column says only what a status contributes. PM10-PM50 give the outcome of a case from all ranks of A and B and from the tolerance checks. "The case cannot pass" means failed (PM50), or error where PM20 applies. A record sequence that the table does not permit is a harness defect, and PM50 makes such a case fail.

**HLS19.** Users do not need to see status identifiers: the harness raises verbose exceptions with explanations that a person can read.

**HLS20.** This document follows a convention: if a certain status X has _A or _B suffix, that only means that status X applies to a rank of instance A or B correspondingly. There are no distinct statuses X_A or X_B, there's only a status X and a rank of instance A or B on which it appears. It's always known to which instance a rank belongs, so instance specific statuses aren't generally needed.

**HLS30.** Status is written to the rank's log file immediately when it happens as `status=X` record (the record should have other fields providing additional information). In the rank's log, a `status=` record has one of three scopes. With `case=` and `call=`, the status applies to that call of that case. With `case=` and without `call=`, it applies to the case. Without `case=`, it applies to the instance (HLS15 item 3). A `call=` field without a `case=` field is an error. It's an error to have a `case=` field without `status=` or `pass=` fields (PI30), or `case=` and `call=` fields without `status=`.

**HLS40.** When the harness processes data obtained from instances A and B, it loads individual case data
from each of the instances and by aggregating statuses from all ranks decides how it's going
to handle case testing. During that, rank statuses might trigger standard pytest
states: error (an exception is thrown from setup/teardown or fixture code),
failure (an exception is thrown from a test function), test skip or test pass. If an exception
is raised, it's message contain all important information to explain the reason for that.

### Tests definition

**TD10.** The definition is in Python. One call defines one configuration: its xDiT settings, its list of components to test, the tolerances for each output of each component, special parameters of B instance, per component path determinism overrides, deadlines and other things the requirements mandate to be user-overridable.
Each entry (a component path) of the list of components defines a separate test case, selectable with `-k` pytest parameter. Each configuration is a separate call.

**TD20.** The xDiT settings usually come from an entry of a CI benchmark YAML file. The `yaml_path` configuration (TD10) parameter defines path to the file (in the same format as the benchmark configurations), and the `entry` field defines name of the entry in the file. There's a helper that reads an entry from a file.  If both `yaml_path` and `entry` are `None` (default), helper returns an empty dictionary. If `yaml_path` can be read and it has an entry with `name` field equal to the `entry`, - the helper returns the read dictionary of xDiT settings. Otherwise the helper raise an exception.

**TD30.** Test configuration call (TD10) can override every field of the xDiT settings, and overrides can be sufficient to define these settings alone, without a YAML file. `overrides` is a dictionary with the structure of a YAML entry. The permitted fields are only those of the `Experiment` class of CI; other fields are an error. After the helper of TD20 returns a dictionary read from `yaml_path` and `entry`, the `overrides` is used to override some fields: `args` key is a dictionary and it is merged key by key: if its key is `None`, the corresponding key is remove from final dictionary, otherwise its value replaces the value in the final xDiT settings dictionary. Similarly, other fields replace the value of the entry, and `None` removes the field.

**TD40.** The resulting xDiT settings dictionary of TD30 apply both to A and to B. Each configuration has its own fixture, which applies instance-specific overrides (TD90, IB10, IB70) to the xDiT settings, launches instances A and B, processes results and lets individual test fixtures + test cases of the configuration run later.

**TD50.** After the configuration fixture (TD40) prepares instance-specific xDiT settings, it passes them to the instance. The instance uses these settings to build the command-line flags as CI does, and then to parse them with the parser of the xDiT runner.

*Rationale:* the xDiT parser applies its defaults and conversions, for example to `gemm_quantization: low=fp4,high=fp6`, so it must not be skipped.

**TD60.** The configuration call has no explicit `name` parameter. The configuration name is the `name` field of the merged xDiT settings (TD30). Hence, if `yaml_path` and `entry` aren't used (i.e. have default `None` value), the `overrides` parameter must have a valid `name` value. The `name` value of the merged xDiT settings (TD30) becomes the configuration name. The configuration name is the name of the configuration's subdirectory in `--xdit_data_dir` and in `--xdit_metrics_dir` (FD20), and the first part of the names of the configuration's test cases (PI50). By convention, a configuration name also let identify the xDiT model to which it applies.

**TD70.** Configuration names are unique. The harness raises an error for two configurations with the same name. The check runs during collection, and it also covers the configurations that `-k` deselects. A separate check in the configuration fixture stops a configuration that starts a second time in one session. Unique names across sessions are the responsibility of the user.

**TD75.** During the test session, tests of the same configuration write files (data files with dumped data and rank log files) to the configuration's subdirectory of `--xdit_data_dir` (FD20). This directory may persist between tests execution sessions (see PI10). Rank log files naming is described in RL10. Data files have basenames according to the `<instance>_<rank>_<call_part>_<content_id>` scheme, where:
1. `<instance>` is `A` for instance A, or `B` otherwise;
2. `<rank>` is the LOCAL_RANK value of the rank that writes the file, in decimal without leading zeros;
3. `<call_part>` is the call identifier (CP100) encoded for file names (FD40);
4. `<content_id>` uniquely identifies the content of the file among all files that a rank writes for one call identifier, and includes the file extension. It contains only `[a-zA-Z0-9.-]` symbols (no `_`). The exact set of content identifiers depends on the implementation (for example, one file with all inputs/outputs, or many standalone files); it must be consistent and documented.

`<instance>`, `<rank>` and `<content_id>` contain no `_`, so the first two `_` and the last `_` of a basename split it into its parts in exactly one way, also when `<call_part>` contains `_`. `data_file_name()` and `parse_data_file_name()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases) implement this scheme.

**TD80.** A configuration name contains only `[a-zA-Z0-9_.-]` symbols, i.e. ASCII letters, digits, `_`, `.` and `-`, and has at most 200 characters. An empty name, the names `.` and `..`, and names that start with `.` or `-` are errors. The harness checks the names during collection, together with the TD70 check (`check_config_name()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases)).
*Rationale:* such a name is a valid directory name (FD20) and a valid part of a `-k` identifier (PI50) without any encoding.

**TD85.** The `--ff` option of pytest, `pytest-xdist` and random-order pytest plugins are not supported.

**TD90.** The harness forces these settings in A and B: no warmup calls, one execution ("iteration" in terms of xDiT runner), and `use_torch_compile` off.

**TD100.** Deadlines have default values, and a configuration (TD10) can override them. The defaults are:
1. A: 5 minutes for the `phase=loading` (RL40), 10 minutes for the `phase=run` (RL60-RL70).
2. B: 10 minutes for the `phase=loading` (RL40), 20 minutes for the `phase=run` (RL60-RL70).
3. each individual call as measured between the call's starting status (ABOUT_TO_START for RL60; PRE_FIDELITY or PRE_REPLAY for RL100) and ending status (DUMPED for RL60; FIDELITY_FAILED or FIDELITY_PASSED or REPLAYED for RL100): 2 minutes in A and 4 minutes in B. It's a limitation of version 1 that these timeouts include file operations.
4. 5 (for A) or 10 (for B) minutes since the last record in the log file in all other cases

**TD110.** **Deadlines enforcement.** For all configuration paths / cases if pytest `-k` wasn't used, or only for cases selected by `-k` otherwise, the configuration fixture (TD40) launches each instance sequentially (TD115) using a supplementary launcher process that takes care of deadlines, child processes cleanup, actually launching an instance with torchrun and statuses of instance's case in some cases. Each rank of the instance writes its deadline -related progress and timestamps to a certain rank-specific log file known to the launcher. The launcher then periodically reads these files and uses progress records and timestamps to enforce deadlines. A missed deadline leads to a warning message and killing the instance and all its descendant processes.

**TD115.** The launcher first runs instance A. If at least one call on any rank of A reached `status=DUMPED`, the launcher also runs instance B for the same test cases that A used.

**TD120.** Only the launcher knows of a missed deadline, a crash, or another premature exit. After the instance ends, for whatever reason, the launcher reads all rank logs and appends records as follows. It never appends a status to a call that is not open, so each call that already ran keeps its status.
1. *Unsupported instance:* at least one rank log has the instance-level record `status=UNSUPPORTED` (RL40). The launcher appends `status=UNSUPPORTED case=<case_id> reason=<the same reason>` for each case of the instance to each rank log, and nothing else.
2. *Crash or hang:* the instance ended in another abnormal way. The status X is HANGED if the launcher killed the instance for a missed deadline (TD110), and CRASHED otherwise. To each rank log the launcher appends:
   1. the instance-level record `status=X reason=<why>`. For HANGED the reason names the missed deadline of TD100. For CRASHED it gives the exit code or the signal;
   2. `status=X case=<case_id> call=<call_id>` for each open call (HLS16). Several calls can be open, because listed components can be nested;
   3. `status=FAILED_TO_START case=<case_id>` for each case of the instance that (a) has no NOT_REACHED, UNRESOLVABLE_PATH, or CASE_CALLS_MISMATCH record, (b) has no call with CRASHED or HANGED, and (c) has no call record, or has at least one pending call (HLS16).
3. *Normal end:* the launcher appends nothing.

**TD125.** *Sanity check* The launcher knows which cases the instance should run, so after completing the above it validates that each case is mentioned at least once in the log of each of the N ranks (HL120). It's a harness bug if the launcher detects cases that are still absent.

**TD130.** At the launcher start, if `HSA_DISABLE_COREDUMP_ON_EXCEPTION` environment variable isn't defined for the launcher process, it must set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` for self and all child processes it runs. For instance B only (to reduce memory pressure), if none of `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` environment variables are defined, the launcher sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for self and all child processes it runs.

**TD140.** The launcher exits with 0 when it worked; the detailed statuses and other run information go through rank log files. A nonzero launcher exit means that the launcher failed and that no data exists, so every case of the configuration fails triggering a pytest error.

**TD200.** Configurations with step caching or hybrid schedules aren't supported. The configuration fixture detects that before running the launcher for instance A, and errors out.

### Tolerances and metrics

**TM10.** Each output of each component path on a rank of instance A is compared to that of instance B taken as a reference (see "Instance A" and "Instance B" below; ranks are matched by their LOCAL_RANK value, reflect in rank log file names). Two comparison metrics are computed and asserted:
1. maximum absolute deviation (requires absolute tolerance value, default is `1e-10`) and
2. the relative L2 deviation (requires relative tolerance value, default is `1e-10`), relative to B. Zero in the denominator gives metric value of zero if the nominator is also zero, otherwise Inf.

For tensors, the cosine similarity is also computed but reported without an assertion.

**TM20.** Each output of each component path must be named in the configuration, and may have
a tolerance override. An output of a component that doesn't have a set name is an error, so
is a name that is not an output. The test function checks the names for each call when it loads the outputs of A and of B. A mismatch fails the test (PM50). The message lists, for each call, the outputs without a name and the names without an output. The check runs in pytest, so it has no status.
*Rationale:* there must be a guarantee that the test itself has no mistake and that it tests each output.

**TM30.** The metrics are computed on the CPU in float64, separately for each rank and call. The test always
reports its metric values, also when it passes.

**TM35.** Outputs structure must be the same on A and B. Different output structure, or NaNs or Infs in any of A or B trigger the test case failure.

**TM40.** Output names use the path notation relative to the return value: `[0]` for a tuple item, `.name` for
an attribute, and an empty name for a bare tensor. Nested containers flatten to their leaves.

**TM50.** The proposed way to specify a list of component paths, outputs and tolerances, to a configuration call (TD10) is as a python dictionary where keys are components paths and values are dictionaries mapping component output names to a named tuple (abs, rel) of tolerance overrides to apply.

**TM60.** For all calls of each test case, all computed metrics, thresholds and comparison results are logged to a permanent structured log file called "a metrics file". The file is meant to preserve a history of values and comparisons obtained for the test case calls and the harness never deletes that file; only creates it if absent and appends a record to it. Each record stores all information from a single test function run that was able to compute at least one metric value. The metrics file isn't updated otherwise.

**TM70.** Each metrics file record has these fields:
1. a UTC timestamp field with value in ISO 8601 `YYYY-MM-DDThh:mm:ssZ` format.
2. the xDiT commit, and whether the working tree had uncommitted changes; For reruns, the user is responsible to ensure the commit corresponds to the version that created data on which the rerun worked.
3. the GPU architecture;
4. run mode (fresh, or rerun)
5. list of calls made, where each element has:
   1. a call identifier
   2. flag is fidelity check passed
   3. list of outputs, where each element has:
      1. output name
      2. tolerance values, if the metric has them (cosine similarity is only computed, but not checked and have no tolerance)
      3. metrics dictionary mapping a metric name to:
         - a dictionary that maps rank id to a metric value (if the metric doesn't have tolerance assigned, such as cosine similarity)
         - a tuple of tolerance value and a dictionary that maps rank id to a tuple of metric value and a flag if it is below the tolerance (if the metric has tolerance)

### Components, paths and calls

**CP10.** The harness tests only components mentioned in an explicit list of component path strings. Each path string defines a component to test and is the path from the runner model object to the callable component or a method. Only the listed components are captured. A listed component must run at least once during pipeline run. A callable component/element is a Python object that can be called, or a PyTorch object with a `forward()` method (since it behaves like a callable in a forward pass of a model).

**CP20.** The list is known before the run. The harness makes the test cases from the list only (without a
model) and list's items (component paths) serve as test case identifiers, so there is an exact 1-to-1 relationship between a component path and a test case.

**CP30.** Component path syntax: dots separate the path elements, and `[i]` indexes a container, for example `pipe.transformer.blocks[0].attn1` and `pipe.transformer.blocks[-1]`. A path has no leading dot. `[key]` selects an item of an `nn.ModuleDict`.

**CP40.** The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
`refiner_pipe`. The harness does not check that the attribute holds a pipeline. An attribute that holds `None`
is a path error.

**CP50.** A call number follows the element after a colon, for example `pipe.transformer:3.blocks[0]:3`. A path without a call number selects all calls, and its one case checks every call. The case fails if one call fails, and the message names the failing calls. Version 1 has accepted limitation: the first path element ignores its call number.

**CP60.** Call numbers are not negative.
*Rationale:* then the harness can save the tensors of a call at once.

**CP70.** A path only tells how to reach the component. It does not describe the call stack, and the harness
never uses it to recover the call stack. Each path addresses only one component: the component that the path
reaches with its call numbers stripped. Call numbers only select calls of that component.

**CP80.** Each callable element of each listed path gets one global (i.e. same for different
component paths) call counter and a set of references to counters of direct callable descendants
(i.e. callable components that directly follow the element in any
path, ignoring non-callable components in between). The harness knows at path resolution which elements
are callable and what their path descendants are.
1. The counter is 0 before the pipeline starts and can only be modified during pipeline run (replays are excluded).
2. The element call handler first increments the counter, then walks over its own direct descendant set and
   invokes "reset_counter()" for each of the descendants. Upon getting a "reset_counter()" request, a counter
   sets own call number to 0 and invokes "reset_counter()" on each of counters of its own descendants, so
   counter resetting propagates to each indirect descendant of the original element that started the reset
   chain. Once all descendant counters reset, i.e. counter reset chain finishes, the element makes all other
   checks, for example whether to save the call.
3. So the first call has the number 1. A counter value of 0 means that the element was never called
   since the last call of its parent in some path that was called chronologically the last.
4. Every call of the element counts, whatever code makes it.
5. When an element counter is found to belong to more than one set of references to counters of direct
   descendants (which means that the element whose calls it counts is shared between several higher-level
   components) the harness issues a warning, explaining that interleaved calls through respective paths might
   not be counted properly due to the counter reset.

**CP90.** `:i` refers to the global counter of its element. Example: `a.b:2.c.d[2]:3` selects a call of `d[2]`
when its counter is 3 and the counter of `b` is 2. During the run of pipelines in A or B, the harness
writes the inputs of each selected call, random-number states and the outputs after the call returns
to data files (it doesn't store them in memory). These data files are read later for call replays and
checks. Only the corresponding test case function or fixture can delete these data files; also - configuration fixture preparing for a fresh run.

**CP100.** A call identifier `<call_id>` is a component path in which each tracked callable element carries its counter value at the time of the call, for example `pipe.transformer:3.blocks[0]:1`. The first path element, such as `pipe`, `second_pipe` or `refiner_pipe` is usually callable, but version 1 does not track it due to complications, and assumes it to be a non-callable instead. This identifier is unique within a run, reads with the CP30 rules, and is the same in A and B when their call order is the same.

**CP110.** The harness does not care who calls a component. It uses only the inputs and the outputs of each
call. Calls from another pipeline, from the runner, or from any other code are tested like all other calls.

**CP120.** Two entries of the same component with different call numbers are separate cases, for example
`pipe.transformer:1` and `pipe.transformer:2`. Overlapping entries are a path error: each of the overlapping cases gets UNRESOLVABLE_PATH (CP140), and the reason names the other entry. Two
entries on the same component are disjoint only if one object has a call number in both entries and the
numbers differ. Then `pipe.transformer:1` and `second_pipe.transformer:2` in the LTX-2.3 runner
are permitted.
*Rationale for the error:* it is better not to support such cases than to make the implementation more
complex.

**CP130.** Calls such as `pipe.vae.decode`, where `.vae` isn't callable or doesn't run in the pipeline,
and `decode` is a method, also must be testable. A path through a module that the run never calls must be valid.

**CP140.** A path must resolve in A and in B. If it doesn't, the rank logs `status=UNRESOLVABLE_PATH case=<case_id> reason=<why>` at the start of phase=run (RL60). Path errors that need a built model show only after the initialization of A; this cost is accepted.

**CP150.** A component path goes into the test case name in the form that PI50 defines, so every test case can be selected with `-k`.

### Instance A

**IA10.** Instance A runs the component as production does: low-precision math, custom kernels, and the other
settings of the configuration. In version 1 it runs eagerly. 

**IA20.** One instance A do call captures and the replays. There is no separate capture instance. The capture
saves the CPU and CUDA random-number states before each call, and each replay restores them.

**IA30.** A runs the pipeline once, captures the calls of the listed components, and replays them with the captured inputs afterwards (a standalone replay). If a component/case is listed, but never run on a rank, the rank get NOT_REACHED status for the case.

*Rationale for standalone replays:* replaying a component outside of its pipeline ensures that the
component is fully isolated from everything else in the pipeline and always computes a
purely functional `y=f(x)`. This is especially important for component benchmarking implementation later.
Given the huge component coupling observed in xDiT this will likely create an issue someday
that a certain component can't be correctly isolated: in that case, if a proper refactoring isn't
possible, a test case definition could be augmented to supply a code that is going to capture from a
pipeline run an additional information needed to make a correct replay, and then setup/teardown that
for doing the replay.
Potential sources of issues: use of `torch.Generator` by the component, internal caches,
asynchronous work that is still running, hooks by other features running, higher memory use, & etc.

**IA40.** **Fidelity check (A and B).** Each instance replays a captured call once to ensure the component was properly isolated and rerunning it yields binary the same result. B also needs it since it run different dtype specific code. Fidelity check may also fail for another reason: if the component is non-deterministic.

**IA50.** Some components may be validly non-deterministic, hence each case / component path may have an instance-specific fidelity check override flag (set on per-case basic during configuration call TD10), that if set, simply excludes all case calls from fidelity check replays. By default, each case (all calls of the case/component) must pass fidelity check in A and in B, so when the flag isn't set it's assumed to be `false`
The harness arranges the following behavior: if set, A's instance flag value is used for B if B doesn't set own flag value.

**IA60.** If the case policy requires fidelity check and the check fails on any rank of A or B, this necessary fails the pytest tests function outcome even if output tolerances are within bounds.

**IA80.** A crash or a hang ends the instance. Version 1 does not recover. An exception that ends the instance counts as a crash.

### Instance B

**IB10.** Instance B is an eager reference. It inherits requirements of A by default (see "Instance A"
section); requirements of this section overrides some of them. B uses the same configuration and
the same world size and parallel layout as A, with these changes:
1. the reference dtype for calculating in extended precision: fp32 by default; a parameter,
   because some models might need a smaller dtype; If it uses
   fp32, B forces full fp32 matrix precision: `torch.backends.cuda.matmul.allow_tf32 = False`,
   `torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`.
2. quantization off, so B loads normal weights;
3. dense attention;
4. vendor packages blocked;
5. no compile, no step cache, no FP8 communication, no hybrid schedules.
6. B gets the same list of component paths (cases) to work on as A. B knows where A stores its dump files, and for the case calls for which it can run replays, it can try to read dump files that A produced to replay them with higher precision. Calls that exist in A (have at least status DUMPED) but never replayed in B (this means replay on A's data, since other replay is fidelity check and it's irrelevant in the context, i.e. calls in B with status REPLAYED), or calls that have at least status DUMPED in B, but never had that status in A, necessary fail its pytest test function even if other calls of the case might pass tolerance checks.

**IB20.** B loads the model in bf16 or in more precise data type that A uses, and casts to the reference dtype if the data dtype offers less precision (less mantissa bits) than the reference dtype (so if the reference dtype is bf16 and there's fp32 data, that higher precision data stays unmodified). It does not load a checkpoint in the reference dtype. B converts all components at once.

**IB30.** After B loads a model, a `TorchDispatchMode` audit checks all dtypes used to run the model (including replays). It warns about dtypes with less precision than the reference dtype, and about custom calls. The check does not fail, only prints a warning informing about an imprecise dtype use, or a custom call. (it's a known limitation of version 1 needs to be addressed in future versions that these warnings aren't recorded and are produced only during instance B runs)

**IB40.** B runs the pipeline in the reference dtype with the same run arguments as A (changes of IB10 applied). It captures its own calls and runs its own fidelity check on them.

*Rationale:* if the replay is faithful with B's inputs, it is also faithful with A's inputs. That
holds if faithfulness depends on the context and state of the replay and not on the input values.
A component whose behavior depends on its inputs and on hidden state could break this assumption,
so a fidelity check on own inputs is needed.

**IB50.** After pipeline run and components fidelity check B replays the inputs of A (using A's random-number states, but restoring the original state after a replay).

**IB70.** B has three run-related parameters for the user: the reference dtype, the self-attention backend and the cross-attention backend, and also own per-path determinism override setting, and deadlines.

**IB80.** An attention backend that declares a fallback, or that needs a blocked vendor package, makes instance B UNSUPPORTED (RL40). A cross-attention parameter for a model without a separate cross-attention backend is an error. The default attention backend is `sdpa_efficient`.

### Rank lifecycle

**RL10.** **Logging.** In its lifetime a rank comes through a sequence of phases (in which all ranks must participate), described in the order below. To preserve events happened/observed during rank's lifetime, each rank creates in the configuration's directory (TD75) a structured log file that, during the instance execution, only the rank can write to. On a rank startup, rank gets from the launcher a path to data directory to store the dumps and the log file. The log file name is `<instance_id>_rank_<rank_number>.jsonl`, where `<instance_id>` is `A` for instance A, or `B` otherwise, and `<rank_number>` is just torchrun's LOCAL_RANK value. It's an error if the file existed when the rank started. Each rank appends (and flushes immediately) to their own log file progress records (TD110, TD120), status changes and all other information that requirements of this document mandate to save and pass to pytest/user.

Each log record must have at least one common field: a UTC timestamp in ISO 8601 `YYYY-MM-DDThh:mm:ssZ` format. Records related to a case must have another common field `case=<case_id>` - the case identifier, so one can use it to find entries related to the case. A case call related records must also have `call=<call_id>` field set. The file hence serves as a structured timestamped log of all important events the rank observed. Neither rank, nor launcher never deletes anything from the file, only appends to it.

After the instance ends, the ownership of the rank logs passes to the launcher (TD120). The launcher takes the number of ranks N of the instance from its parameter (HL120) and does not infer it from the files that exist. If the log of one of the ranks 0 to N-1 does not exist, the launcher creates it. TD120 then applies to it as to a log without case records.

**RL20.** Getting to the next phase is recorded in the rank's log with a corresponding field `phase=`. These records are used by the launcher to enforce deadlines (TD110).

**RL30.** When a rank just starts, it writes `started=true` log record.

**RL40.** **phase=initialization** This is where the rank parses its xDiT setting (TD50) and determines if it can run the configuration. If the rank cannot run the configuration, it writes the instance-level record `status=UNSUPPORTED reason=<what is not supported>` and exits with an error. This automatically forces torchrun to kill all other ranks. By reading the log files the launcher is able to correctly identify the cause and set all cases statuses to UNSUPPORTED too.
*Rationale for not setting case statuses in rank code:* In case of a hang or a crash, the launcher have to identify cases that didn't run to assign them appropriate statuses. We simply reuse this machinery for setting case statuses for unsupported configuration too.

**RL50.** **phase=loading** Doesn't have specific statuses.

**RL60.** **phase=run** During this phase rank runs the model pipeline and captures data. From the beginning, the rank has a set of cases it has to run; it makes a copy of this set into "unseen cases set". Before it installs hooks, the rank resolves the path of each case (CP30-CP50) and checks the entries for overlap (CP120). For each case with a path error it logs `status=UNRESOLVABLE_PATH case=<case_id> reason=<why>` and removes the case from the set of cases it has to run and from the "unseen cases set". Such a case gets no hook and no NOT_REACHED. The rank installs hooks to tracked components and when a hook fires, if the case call conditions are met, the rank:
1. removes the case from "unseen cases set" if it's still there; logs `status=ABOUT_TO_START case=<case_id> call=<call_id> files=<files>` (where `<files>` is a 2-element tuple, each element of which is a list of filepaths to files that will contain dumped data for the call of the case. The first element is for files having data necessary to start the call, such as inputs, rng states & etc, and the second element is for outputs; Filepaths follow a pattern and can be recovered by the rank later from the generic data path it has and a call identifier),
2. saves component inputs and other applicable runtime data to the case call specific dump file it previously allocated (or files, depending on what is more convenient),
3. runs the call and
4. similarly saves outputs on the call return
5. if dumps were successful: the rank writes a `status=DUMPED case=<case_id> call=<call_id>` record to the log and appends the call identifier to an internal ordered list of calls to replay (called "replay list").

If during steps 2 or 4 the rank finds that it can't process the call data correctly (for example, some of the data to dump isn't serializable, a tensor has quantized dtype, which can't be de-quantized without special knowledge and hence used for B, or a data file basename is longer than 255 bytes (FD40)), instead of step 5 that case call records `status=UNSUPPORTED case=<case_id> call=<call_id> reason=<why>`. The rank writes this record when the call returns, also when it found the problem in step 2. The call thus stays open (HLS16) while it runs. If the rank found the problem in step 2, it still runs the call, but it does not do step 4. Then the pipeline run proceeds.

**RL70.** **Duration of phase=run for the timeouts of TD110.** Since this phase execute a potentially lengthy code that isn't meant to be accounted for in the phase's deadline timeout calculation, the rank maintains an internal counter `duration_excluded` that is set to 0 when the phase starts. Execution of the code doesn't belong to the pipeline duration deadline threshold (such as dumping input, output & etc) is timed and added to the `duration_excluded`. During the phase, to each log record the rank produces, it automatically adds a  `duration_excluded=` field with the current value of the counter. Hence the launcher can always calculate a true duration of the phase for TD100-TD110 by subtracting from the `now()` timestamp, the timestamp of the latest `phase=` change, and the latest observed `duration_excluded=`.

**RL80.** Once the model finishes running, only these things could have happened to each of the calls of each of the cases that the rank should have run during the model run phase:
1. the call have obtained UNSUPPORTED status. It makes the call excluded from replays.
2. the call have completed successfully and got `status=DUMPED`.
3. the whole case never had a single call; the case is in the "unseen cases set" (RL60)
4. the call started but did not return normally, because the component raised an exception that the pipeline handled. The hook writes no record for the exception. When the pipeline run ends, the rank logs `status=UNSUPPORTED case=<case_id> call=<call_id> reason=<the exception>` for each such call.

For each case in the "unseen cases set" the rank logs `status=NOT_REACHED case=<case_id>`.
In version 1 no cross-rank synchronization about which cases to replay is performed, and the user is responsible for choosing components that runs equally successfully (in terms of statuses) on all ranks simultaneously.

*Rationale for the absence of replay synchronization:* Synchronization adds considerable complexity to the code. Instead we assume that each component runs on every rank the same number of calls, and each of the calls is handled by the rank successfully, so ordered "replay lists" are identical across all ranks, therefore letting each rank to use its private "replay list" is harmless. The user is responsible to ensure the component they select meets the assumptions.

However, to alleviate debugging, all ranks compare their "replay lists" with one `all_gather_object()` call. Cases whose lists differ get `status=CASE_CALLS_MISMATCH case=<case_id>` on all ranks, and are dropped on all ranks. B runs the same check on its list of A's calls.

**RL90.** **phase=replay** Two kind of replays are needed:
- both instance A and B should replay each call for fidelity check IA40-IA60;
- instance B also replays each call of its list of A's calls (RL80: the calls that got status=DUMPED on A; B can read logs of A for this) that is also in B's own replay list, using A's data casted to reference dtype to obtain reference outputs of the component. The result of B's fidelity check for the call does not matter here: it is local to a rank, so it must not decide which calls a rank replays (RL100). A failed fidelity check still makes the case fail (IA60, PM40).

Instance B runs both replays: first it runs fidelity check replays on its own inputs and outputs, then it runs replays on A's inputs.

**RL100.** Call replays run in "replay list" order (RL60), and in a lock step, with a stream synchronization and a barrier before and after replay. This helps to time execution properly to ensure correct call timeouts (TD100.3), and help to correctly attribute crashes, should they happen. For each call in ordered "replay list", the rank does:
1. if this is a fidelity check replay and call's case has the instance-specific fidelity check override flag set: the rank logs `status=FIDELITY_SKIPPED case=<case_id> call=<call_id>` and proceeds to the next call in the list (since all ranks obtain the same "replay list" (RL80) and have the same flag value, the same case calls will be skipped on all ranks simultaneously and that can't lead to a hang).
2. the rank logs `status=<status> case=<case_id> call=<call_id> file=<file>` where the `<status>` is `PRE_FIDELITY` if this is fidelity check, or `PRE_REPLAY` if this is B replaying inputs from A; `<file>` is a file path that will contain outputs of the call.
3. the rank reads inputs, rng states and outputs (only if it's a fidelity check), from the call replay files (the paths are recoverable from data directory path and the call identifier).
4. The rank executes a stream synchronization and a barrier.
5. the rank saves current CPU and CUDA rng states into a temporary variable; restores rng states from loaded from the input file values, execute the call with loaded inputs (B must have casted them to the reference dtype first), obtains the replayed call outputs and saves them to the `<file>`. Finally it restores the original CPU and CUDA rng states from the temporary variable.
6. The rank executes a stream synchronization and a barrier.
7. if this is a fidelity check, the rank perform bitwise comparison of each of the elements of obtained output with expected (read in point 3). Any mismatch in the outputs structure or content gives `status=FIDELITY_FAILED case=<case_id> call=<call_id>`. If the check passes, the rank logs instead `status=FIDELITY_PASSED case=<case_id> call=<call_id>`.
8. Otherwise if this isn't a fidelity check, the rank logs `status=REPLAYED case=<case_id> call=<call_id>`.

*Rationale:* strict call execution order and synchronization with barriers are necessary to ensure proper collectives functioning and along with detailed logging help to correctly attribute crashes. A fault that silently damages memory and shows only in a later case is an accepted limit.

**RL110.** Duration rules and `duration_excluded` counter of RL70 also apply to replay calls of phase=replay in RL100.

### pytest integration

**PI10.** Due to a cost of running instance A and instance B, there are 2 modes of running the test suite:
1. a "fresh run" runs everything in full (i.e. for each configuration in the test suite run instances A and B to perform dumping of inputs/outputs and then run test case functions to obtain results of testing), and
2. a "rerun" whenever possible reuses results of a previous fresh run (i.e. it never runs a single instance A or B for any configuration, instead it only uses previously created files to run test case functions).

**PI20.** The harness defines these custom pytest CLI arguments:
1. flags: `--xdit_no_cache_passed` and `--xdit_allow_rerun`. Both aren't set by default.
2. strings with absolute or relative paths (relative paths are resolved relative to `os.getcwd()`, see FD10):
    1. `--xdit_data_dir` (default: `./_data`) to set the data directory, which contains the configuration subdirectories (FD20) with the data files (TD75) and rank log files (RL10);
    2. `--xdit_metrics_dir` (default: `./_metrics`) to set the metrics directory. The metrics file (TM60) of a case is `<xdit_metrics_dir>/<configuration_name>/<case_part>.jsonl`, where `<configuration_name>` is the configuration name (TD60) and `<case_part>` is the case identifier (CP20) encoded for file names (FD40); see `metrics_file_path()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases).

**PI30.** If `--xdit_no_cache_passed` CLI flag (PI20) is NOT set, for each test passed in a fresh runs or a rerun, the harness appends to each rank log file a record `case=<case_id> pass=true` and deletes their data files. Subsequent reruns detect these records in the log files and "skip" corresponding tests with a message "Passed in previous run" (all ranks of the case must have this `pass=true`; otherwise the case fails with a message "inconsistent pass record for the case: ranks <..> have `pass=true` while ranks <..> doesn't have it").
If `--xdit_no_cache_passed` is set, a passed test doesn't add any record to the rank log files, and does not delete data files, so a subsequent rerun doesn't observe `case=<case_id> pass=true` in rank log files, expects to find data files and executes test case function in full.

**PI40.** If `--xdit_allow_rerun` CLI flag (PI20) is NOT set, which is the default, the harness ALWAYS clean up existing data and rank log files of the configurations that have cases to run (PI60.3), and do a fresh run. If the flag is set, the harness does a rerun only when PI60.2.2 chooses it: the flag permits a rerun, but doesn't guarantee it.

**PI50.** Test case names are generated automatically from a configuration call (TD10). The parametrization ID of a test case is `<configuration_name>/<case_part>`, where `<configuration_name>` is the configuration name (TD60-TD80) and `<case_part>` is the component path (CP20) encoded as follows:
1. `[a-zA-Z0-9_.:-]`, `[` and `]` symbols stay unchanged, so a component path that follows CP30 and CP50 does not change;
2. each other symbol becomes `+XX` for each byte of its UTF-8 encoding (`XX` is uppercase hexadecimal).

Neither part contains `/`, so the ID gives back the configuration name and the component path exactly. A test case function has only this one parametrization, so the pytest ID of a case is exactly this string (pytest joins the IDs of several parametrizations with `-`, which configuration names can contain). All symbols of the ID can be part of one `-k` identifier, and pytest does not escape them. The harness needs pytest 7.0 or later, because older versions do not accept `/` in `-k` identifiers. `test_case_id()` and `parse_test_case_id()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases) implement this scheme.

`-k` matches substrings without regard to letter case. For example, `-k "flux2.default/pipe.transformer"` also selects `flux2.default/pipe.transformer:1`, while `-k "[flux2.default/pipe.transformer]"` (with the brackets of the test name) selects only that case. `-k` cannot tell apart configuration names that differ only in letter case; a node ID can.

**PI60.** **Pytest life cycle.** Here's how execution steps maps onto pytest:
1. Each pytest module defines configurations (TD10) of a single xDiT model to test. A module level `pytest_generate_tests()` hook parses each configuration of the module and spawns an individual test case for each element of configuration's component list.
2. A global session scoped fixture determines the run mode (PI10) and sets a global `run_mode` flag defining the mode, so other test code could use it:
   1. if `--xdit_allow_rerun` CLI flag (PI20) is NOT set, it's always a fresh run;
   2. otherwise, since the fixture has access to all test case name that are about to execute, and each test case name allow to reconstruct the configuration name exactly, the fixture can determine if a fresh run, or a rerun is needed by inspecting rank log files of the configurations of these cases. If none of these files exists, it's a fresh run. Otherwise a rank log that does not exist is a log without records (HL120), and if and only if for all cases to run all rank log files contain `case=<case_id> pass=true` record for the case, the fixture chooses a fresh run mode for the session; otherwise it choose rerun mode. The user is responsible to not set the flag if any underlying code or configurations changed after the fresh run that created the data.
3. A configuration (TD10) level fixture depending on the value of the global `run_mode` flag either do a fresh run or rerun of the configuration (PI10). For a fresh run, before it launches instance A, the fixture deletes the directory of its configuration in `--xdit_data_dir` (TD75) with all its content (data files with dumps and rank log files). It deletes no other directory: the directories of the configurations that have no case to run in the session stay. The user is responsible the fresh run doesn't wipe out more data than they wanted: it deletes the data of all cases of the configuration, also of the cases that `-k` deselects. After a configuration level fixture runs the configuration's individual test case functions (that weren't skipped by pytest `-k` parameter, or by other valid reason that the fixture might detect). After that the loop repeats: another configuration level fixture runs, then its test cases.

#### Mapping statuses to pytest outcomes

**PM10.** A configuration level fixture, a test case-level fixture and a test case function implement different parts of the mapping from statuses to pytest outcomes. The rules apply in this order, and the first rule that applies gives the outcome of the case: (1) the launcher exited with a nonzero code (TD140): error; (2) PM20 item 1: error; (3) PM30: skip; (4) PM20 item 2: error; (5) PM40: passed; (6) PM50: failed. IA60 and IB10 item 6 name conditions under which a case cannot pass. They do not change this order.

**PM20.** **Errors.** In pytest, only setup/teardown or fixture code could emit errors. We generally turn into an error any situation where a test case doesn't have dumped data to work with:
1. a configuration level fixture inspects all rank log files and if not a single case of any single rank reached DUMPED status, it marks all test cases of the configuration as "error".
2. a case level fixture inspects all rank log files and if it finds that no single pair of ranks of A and B reached DUMPED status for the case, it marks the case as "error". If some rank pairs reached DUMPED, but others don't, the test function runs on the obtained data, but eventually necessary fails later.

**PM30.** **Skips.** Skips in pytest can be generated from both fixtures and test functions.
1. In a fresh run mode, a whole configuration (all cases of the configuration) can be "skipped" only if the configuration is explicitly not supported on the current hardware (not the case for harness version 1).
2. In a rerun mode a test case is "skipped" by the case fixture when all ranks of the case have `case=<case_id> pass=true` record. If only some ranks have the record, but others don't, the test function runs on the available data, but eventually necessary fails later.

**PM40.** **Passed.** Only a successfully ran test case function sets "passed" outcome. A test case is "passed" if and only if these conditions holds simultaneously:
   1. every call on all ranks of A and B reached the following last status:
      - instance A, which only does a fidelity check: the last call status is FIDELITY_SKIPPED when fidelity check override flag is set, or FIDELITY_PASSED otherwise;
      - instance B, which does a fidelity check and replay of A inputs: the call has FIDELITY_SKIPPED when the fidelity check override flag that applies to B (IA50) is set, or FIDELITY_PASSED otherwise, and the last call status is REPLAYED.
   2. for all calls observed for the case/component, all outputs obtained on all ranks of A are within specified tolerances from outputs obtained from corresponding ranks of B.

**PM50.** **Failed.** Every other status combination for a test case, whenever possible still runs tolerances checks, but even if the checks passes, necessary leads to a "failed" test outcome. Failure message MUST list all independent reasons that led to the failure.

**PM60.** A test reached test function (i.e. configuration and test fixture didn't generate errors or skips) independently checks captured outputs of each rank of A versus corresponding outputs of a replay with the inputs A that B rank did, against the same tolerance. The test runs as many checks as possible before it fails, like the `EXPECT_` macros of GoogleTest. It uses the installed `pytest-check` package. After/during doing checks, before generating failure, it writes a record to the case metric file with per-output tolerances, per-output per-rank values of tested metrics and the test per-output per-rank test outcome.

**PM70.** The error or failure message lists every rank with its status and, if known, the call identifier, the reason and other applicable information the requirements of this file require to present to a user.

**PM80.** Whatever the pytest outcome is, a test case fixture must issue a warning for each call that have `status=FIDELITY_SKIPPED` on at least one rank. Message is "WARNING: instances A and B didn't run fidelity check for the call <call_id> due to fidelity override flag set. You are responsible to ensure the component is properly isolated and its determinism state is known."

#### Files and directories

**FD10.** A session level fixture reads pytest CLI arguments (PI20.2) and resolves `--xdit_data_dir` and `--xdit_metrics_dir` to absolute paths, resolving symlinks. It's an error if the directories resolve to the same path, or if one of them turns out to be a descendant of the other (`check_roots()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases)). Then it stores directories somehow, so other fixtures or test functions work only with absolute paths for these directories. The user is responsible to never run two testing sessions in parallel using the same `--xdit_data_dir` or `--xdit_metrics_dir`. No code checks or prevents that.
*Rationale:* a fresh run deletes configuration directories under `--xdit_data_dir` (PI60.3), and the harness never deletes a metrics file (TM60). A data directory inside the metrics directory can be the metrics subdirectory of a configuration (FD20).

**FD20.** `--xdit_data_dir` and `--xdit_metrics_dir` each contain one subdirectory per configuration (TD10), named exactly as the configuration (TD60, TD80). The subdirectory in `--xdit_data_dir` contains the data files (TD75) and the rank log files (RL10) of the configuration. The subdirectory in `--xdit_metrics_dir` contains the metrics files (TM60, PI20.2.2) of the configuration's cases.

**FD30.** **Data files.** Tensors go into safetensors files. Structure and plain values go into JSON. Non-serializable types or data types that don't fit into the above give UNSUPPORTED for the affected call (RL60).

**FD40.** **Identifiers in file names.** A case identifier (CP20) or a call identifier (CP100) goes into a basename in encoded form. The encoding uses the identifier string only (it does not parse CP30), so it works for every `[key]` of an `nn.ModuleDict`:
1. `[a-zA-Z0-9_.-]` symbols stay unchanged, except a `.` or `-` at the first position;
2. `:` becomes `@`, `[` becomes `{`, and `]` becomes `}`;
3. each other symbol, and a first `.` or `-`, becomes `%XX` for each byte of its UTF-8 encoding (`XX` is uppercase hexadecimal).

Each identifier has exactly one encoded form, and decoding rejects every other string. The result never contains `/`, whitespace, `:` or a glob symbol (`*?[]`). A metrics file basename longer than 255 bytes is a collection error. A data file basename longer than 255 bytes makes the call UNSUPPORTED (RL60). `path_id_to_file_part()` and `path_id_from_file_part()` of [Appendix A](#appendix-a-names-of-files-directories-and-test-cases) implement this encoding.

## Appendix A. Names of files, directories and test cases

This is the reference implementation of the names that TD75, TD80, RL10, PI20.2.2, PI50, FD10, FD20 and FD40 define.

```python
"""Names of the files, directories and pytest test cases of the component test harness.

Layout (FD20, TD75, RL10, PI20.2.2); F is the file encoding (FD40), the configuration name is
used without change (TD80):
    <xdit_data_dir>/<config_name>/<instance>_rank_<rank>.jsonl                  rank log
    <xdit_data_dir>/<config_name>/<instance>_<rank>_<F(call_id)>_<content_id>   data file
    <xdit_metrics_dir>/<config_name>/<F(case_id)>.jsonl                         metrics file

Test case ID (PI50); T is the test ID encoding:
    <config_name>/<T(case_id)>
"""

import os
import re
import string

NAME_MAX = 255
"""Maximum basename length in bytes on ext4, XFS, Btrfs and tmpfs."""

CONFIG_NAME_MAX = 200
"""TD80."""

_ALNUM = string.ascii_letters + string.digits


class _Codec:
    """Reversible encoding of any non-empty ``str`` into a restricted alphabet.

    Each input character becomes exactly one output form: itself if it is in ``literal`` (and, for
    the first character, not in ``not_first``); its single-character ``replacement``; or
    ``escape`` + two upper-case hex digits for each byte of its UTF-8 encoding ("surrogatepass"
    makes this total over all Python strings). Literal characters, replacement characters and the
    escape character are disjoint, and every escape character starts a 3-character token, so a
    left-to-right scan restores the input.

    Guarantees: ``decode(encode(x)) == x`` for every non-empty ``x``; ``decode`` raises
    ``ValueError`` for every string that ``encode`` cannot produce, so each input has exactly one
    encoded form.
    """

    def __init__(self, literal: str, replacement: dict[str, str], escape: str, not_first: str = ""):
        self._literal = frozenset(literal)
        self._replacement = dict(replacement)
        self._original = {v: k for k, v in replacement.items()}
        self._escape = escape
        self._not_first = frozenset(not_first)
        self._token = re.compile(rf"{re.escape(escape)}([0-9A-F]{{2}})|(.)", re.DOTALL)
        produced = [escape, *self._replacement.values()]
        if (
            len(escape) != 1
            or len(set(produced)) != len(produced)
            or not self._literal.isdisjoint(produced)
            or not self._literal.isdisjoint(self._replacement)
        ):
            raise ValueError("literal, replaced, replacement and escape characters must be disjoint")

    def encode(self, text: str) -> str:
        if not text:
            raise ValueError("an identifier must not be empty")
        out = []
        for i, ch in enumerate(text):
            if ch in self._literal and not (i == 0 and ch in self._not_first):
                out.append(ch)
            elif ch in self._replacement:
                out.append(self._replacement[ch])
            else:
                out.extend(f"{self._escape}{b:02X}" for b in ch.encode("utf-8", "surrogatepass"))
        return "".join(out)

    def decode(self, part: str) -> str:
        raw = bytearray()
        for hex_byte, ch in self._token.findall(part):
            if hex_byte:
                raw.append(int(hex_byte, 16))
            else:
                raw += self._original.get(ch, ch).encode("utf-8", "surrogatepass")
        try:
            text = raw.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            text = ""
        # The scan above accepts more than ``encode`` produces (foreign characters, an escape
        # character without two upper-case hex digits, escapes of literal characters). Encoding
        # again rejects all of them in one place, instead of a second copy of the rules.
        if not text or self.encode(text) != part:
            raise ValueError(f"{part!r} is not an encoded identifier")
        return text


# FD40. Output alphabet: [A-Za-z0-9_.-@{}%]; never "/", NUL, whitespace, ":" or a glob character;
# never starts with "." or "-". "{" and "}" are not glob characters; bash and zsh expand braces only
# if they contain "," or "..", and the codec never emits ",".
_FILE_CODEC = _Codec(_ALNUM + "_.-", {":": "@", "[": "{", "]": "}"}, "%", not_first=".-")

# PI50. Output alphabet: [A-Za-z0-9_.:[]-+], a subset of the characters of a ``-k`` identifier
# (``(\w|:|\+|-|\.|\[|\]|\\|/)+`` in pytest >= 7.0). It is printable ASCII without "\", so pytest's
# ``ascii_escaped`` does not change it, and it never contains "/", the separator after the
# configuration name. A case identifier that follows CP30 and CP50 does not change.
_TEST_ID_CODEC = _Codec(_ALNUM + "_.-:[]", {}, "+")


def path_id_to_file_part(identifier: str) -> str:
    """Encode a case or call identifier for a basename (FD40)."""
    return _FILE_CODEC.encode(identifier)


def path_id_from_file_part(part: str) -> str:
    return _FILE_CODEC.decode(part)


def check_basename_length(basename: str) -> str:
    """Return ``basename`` if it has at most ``NAME_MAX`` bytes; raise ``ValueError`` otherwise."""
    if len(basename.encode()) > NAME_MAX:
        raise ValueError(f"basename is longer than {NAME_MAX} bytes: {basename!r}")
    return basename


# ---------------------------------------------------------------------------------------------
# Configuration names (TD80).

_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def check_config_name(name: str) -> str:
    """Return ``name`` unchanged if it follows TD80; raise ``ValueError`` otherwise.

    A TD80 name is already a safe directory name and a safe ``-k`` identifier: ASCII only, no
    ``/``, never ``.`` or ``..``, no leading ``.`` or ``-``. So the configuration directory name and
    the first part of a test case ID are the configuration name itself; this function only checks
    the preconditions of that identity.
    """
    if not isinstance(name, str) or len(name) > CONFIG_NAME_MAX or not _CONFIG_NAME_RE.fullmatch(name):
        raise ValueError(f"{name!r} is not a valid configuration name (TD80)")
    return name


# ---------------------------------------------------------------------------------------------
# Test case IDs (PI50).

TEST_ID_SEP = "/"


def test_case_id(config_name: str, case_id: str) -> str:
    """``<config_name>/<T(case_id)>``, the only parametrization ID of the case's pytest item."""
    return f"{check_config_name(config_name)}{TEST_ID_SEP}{_TEST_ID_CODEC.encode(case_id)}"


def parse_test_case_id(test_id: str) -> tuple[str, str]:
    """Return ``(config_name, case_id)``. Neither part can contain "/", so the split is unique."""
    config_name, sep, part = test_id.partition(TEST_ID_SEP)
    if not sep:
        raise ValueError(f"{test_id!r} is not a test case ID")
    return check_config_name(config_name), _TEST_ID_CODEC.decode(part)


# ---------------------------------------------------------------------------------------------
# Paths.

INSTANCES = ("A", "B")
_RANK_RE = re.compile(r"0|[1-9][0-9]*")
# No "_" in a content id: the last "_" of a data file name ends the encoded call id (TD75).
_CONTENT_ID_RE = re.compile(r"[A-Za-z0-9.-]+")
_RANK_LOG_RE = re.compile(r"(?P<instance>[AB])_rank_(?P<rank>0|[1-9][0-9]*)\.jsonl")
_METRICS_SUFFIX = ".jsonl"


def _check_instance_rank(instance: str, rank: int) -> None:
    if instance not in INSTANCES or not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError(f"bad instance or rank: {instance!r}, {rank!r}")


def check_roots(data_dir: str, metrics_dir: str) -> tuple[str, str]:
    """FD10: return both roots resolved; raise if they are equal or one is inside the other.

    A fresh run deletes configuration directories under the data root (PI60.3), and TM60 forbids
    deleting a metrics file. A data root inside the metrics root can be the metrics subdirectory of
    a configuration (FD20).
    """
    data, metrics = os.path.realpath(data_dir), os.path.realpath(metrics_dir)
    if os.path.commonpath([data, metrics]) in (data, metrics):
        raise ValueError(f"--xdit_data_dir {data!r} and --xdit_metrics_dir {metrics!r} must not overlap")
    return data, metrics


def config_dir(root: str, config_name: str) -> str:
    """``<root>/<config_name>``, where ``root`` is a resolved data or metrics root (FD20)."""
    return os.path.join(root, check_config_name(config_name))


def rank_log_name(instance: str, rank: int) -> str:
    _check_instance_rank(instance, rank)
    return f"{instance}_rank_{rank}.jsonl"


def parse_rank_log_name(name: str) -> tuple[str, int]:
    """Also the launcher's filter for rank logs (RL10); it never matches a data file name."""
    m = _RANK_LOG_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"{name!r} is not a rank log name")
    return m["instance"], int(m["rank"])


def data_file_name(instance: str, rank: int, call_id: str, content_id: str) -> str:
    """``<instance>_<rank>_<F(call_id)>_<content_id>``; ``content_id`` includes the extension.

    The result can be longer than ``NAME_MAX``: the rank logs it in ``files=`` (RL60, step 1), and
    a ``check_basename_length`` failure before the write makes the call UNSUPPORTED (RL60, step 2).
    """
    _check_instance_rank(instance, rank)
    if not _CONTENT_ID_RE.fullmatch(content_id):
        raise ValueError(f"bad content id: {content_id!r}")
    return f"{instance}_{rank}_{path_id_to_file_part(call_id)}_{content_id}"


def parse_data_file_name(name: str) -> tuple[str, int, str, str]:
    # Instance and rank contain no "_", so the first two "_" end them; the content id contains no
    # "_", so the last "_" starts it. The encoded call id is everything between and may contain "_".
    fields = name.split("_", 2)
    if len(fields) == 3 and "_" in fields[2]:
        instance, rank, rest = fields
        call_part, content_id = rest.rsplit("_", 1)
        if instance in INSTANCES and _RANK_RE.fullmatch(rank) and _CONTENT_ID_RE.fullmatch(content_id):
            return instance, int(rank), path_id_from_file_part(call_part), content_id
    raise ValueError(f"{name!r} is not a data file name")


def metrics_file_path(metrics_root: str, config_name: str, case_id: str) -> str:
    """``<metrics_root>/<config_name>/<F(case_id)>.jsonl`` (PI20.2.2).

    Raise ``ValueError`` for a basename longer than ``NAME_MAX``; the harness calls this during
    collection, so that is a collection error (FD40).
    """
    basename = check_basename_length(path_id_to_file_part(case_id) + _METRICS_SUFFIX)
    return os.path.join(config_dir(metrics_root, config_name), basename)


def parse_metrics_file_path(path: str) -> tuple[str, str]:
    """Return ``(config_name, case_id)`` from the last two components of ``path``."""
    parent, basename = os.path.split(path)
    if not basename.endswith(_METRICS_SUFFIX):
        raise ValueError(f"{path!r} is not a metrics file path")
    case_id = path_id_from_file_part(basename[: -len(_METRICS_SUFFIX)])
    return check_config_name(os.path.basename(parent)), case_id
```
