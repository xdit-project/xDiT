# Requirements and use cases of the component test harness

This document lists the use cases and the requirements of the xDiT component test harness.

Each item has an identifier: `UC` for use cases, `R` for requirements.

## Use cases

**UC1. Check that a component works correctly.** A developer asks: "Does this component of this model
configuration work correctly?" The harness runs the component as production runs it (instance A) and as an
eager reference in high precision (instance B). It then compares the outputs against tolerances. Requirements:
R1–R6, R12–R15, R22–R26, R42–R56.
*Rationale:* some operations of xDiT diffusion models can be flaky (produce wrong numerics, or cause hangs or
crashes). Tests of single components are decoupled from the whole model.

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang.
The report names the instance, the component, the call and the rank. Requirements: R6, R45–R47, R53, R57–R75.

**UC3. Test the configurations of the CI benchmark files.** A test selects an entry of a CI benchmark YAML
file. The test can change any field of the entry. Requirements: R13–R15.
*Rationale:* the user specifies model configurations in the same way as the benchmark configurations.

**UC4. Cover many configurations of one model in one test module.** Requirements: R16–R19.
*Rationale:* there are hundreds of configurations in total, and most of them are very similar.

**UC5. Test sub-components and selected calls.** Examples are a transformer block, an attention module, the
VAE decode method, and one specific call of a transformer. Requirements: R27–R41.

**UC6. Tune tolerances over time.** A test reports its metrics in each run. A missing tolerance makes the test
fail, so the user sets tolerances from the reported values. The metrics history shows how the deviations
change while the implementation changes. Requirements: R23–R25, R87, R88.

**UC7. Run the tests again after failures.** A new run compares the stored data again, for example with new
tolerances, and does not run A and B again. After all cases pass, the next run starts from the beginning
automatically. Requirements: R85, R86.

**UC8. Select single cases.** The user selects cases with pytest, for example with `-k` or a node ID.
Requirements: R41.

**UC9. Keep the tests valid while xDiT changes.** Requirements: R2, R3, R9.
*Rationale:* xDiT changes fast, and its components get new augmentations all the time. Hard-coded tests become
obsolete.

**UC10. Accept nondeterminism that is valid.** An override of one test changes the result of nondeterminism in
that test from a failure to a warning. The determinism checks of all other tests stay active. B should have
its own override. Requirements: R46, R47.
*Rationale:* some operations are validly nondeterministic. A model can have one nondeterministic component
among many, and that component must be testable. Instance B generally use different kernels that might have
other determinism expectations.

**UC11. Later use cases.** Compiled mode, a reference at world size 1 (B1), several input sets, MI300X and
other hardware, and benchmark and determinism scripts that use the foundation of the harness.

## High-level requirements

**R1.** The harness tests individual components of xDiT diffusion models, for example a VAE, a transformer, a
transformer block, an attention module, a text encoder, or a method such as the VAE decode. It uses exactly
the configuration and the weights that the xDiT runner uses for a model configuration.

**R2.** The harness takes each component from the model that the xDiT runner builds. It does not extract,
hard-code or build a component itself.
*Rationale:* the setup of a component depends on the model configuration. For example, the VAE of Wan2.2 can
be the VAE of `diffusers`, a DistVAE, or something else. Also, xDiT changes fast (UC9).

**R3.** The details of a configuration are private to the configuration: Ulysses, ring, classifier-free
guidance, quantized GEMMs and others. The tests do not depend on them. The tests are generic, so that they can
run every configuration. The configuration identifier is one parameter among many.

**R4.** The reference is the same xDiT component, built by the xDiT runner, run with extended precision (fp32
by default). It is not a `diffusers` implementation.

**R5.** One test answers one question: "Does this component, in this regime, with these inputs, work
correctly?" A failed check belongs to one test, so only that test fails.

**R6.** A test fails for a deviation beyond its tolerance, a crash, a hang, nondeterminism, or any other
error. "Flaky" includes a crash and a hang.

**R7.** **Working tests first.** Test coverage has priority over correct reports, for example a correct call
address. A wrong report is acceptable as a documented limitation, possibly with a corresponding future-work
item. A deviation that a working test needs is permitted with the approval of the user.

**R8.** Decisions rest on facts that the code shows directly.

**R9.** Where possible, xDiT gets a refactor that makes components testable, instead of a hack around hidden
coupling. Examples: compile code only compiles, and the warmup is not part of the compile function.
In version 1, xDiT changes only where the first configuration needs a change.

**R10.** Tests and benchmarks are separate from the entry point on. Benchmark scripts and determinism-check
scripts come later. They use the same foundation that the tests use to load and build a model and to get its
components. Version 1 plans nothing for benchmarks, but it identifies the code that benchmarks will use.

**R11.** One model initialization of an instance serves the tests of many components.
*Rationale:* a model can need 5–10 minutes to load and initialize. That is too long for each test.

### Test definition

**R12.** The definition is in Python. One call defines one configuration: its run settings, its list of
components, and the tolerances for each output of each component. Each configuration is a separate call.

**R13.** The run settings come from an entry of a CI benchmark YAML file, in the same format as the benchmark
configurations. A helper reads the entry, builds the command-line flags as CI does, and parses them with the
parser of the xDiT runner.
*Rationale:* the parser applies its defaults and conversions, for example to
`gemm_quantization: low=fp4,high=fp6`.

**R14.** A test can override every field of the YAML entry and it can be sufficient to define the
configuration alone, without a YAML file. `overrides` is a dictionary with the structure of a YAML entry. The
permitted fields are those of the `Experiment` class of CI. Other fields are an error. `args` merges key by
key: a key replaces the value of the entry, and `None` removes the key. Other fields replace the value of the
entry, and `None` removes the field.

**R15.** The overrides apply both to A and to B. B has a few parameters of its own (R55).

**R16.** Each configuration has its own fixture, which runs A and B of that configuration. The tests of the
configuration run before the next configuration starts its instances.

**R17.** The configuration name is the `name` of the entry by default. It is an error if neither a name nor an
entry is set, or if the name is absent, `None`, not a string, or empty. The name selects the data directory
and the metrics file of the configuration.

**R18.** Configuration names are unique. The harness raises an error for two configurations with the same
name. The check runs during collection, and it also covers the configurations that `-k` deselects. A separate
check in the configuration fixture stops a configuration that starts a second time in one session. Unique
names across sessions are the responsibility of the user.

**R19.** A name contains only letters, digits, `_`, `.` and `-`. The literal names `.` and `..` are not
allowed. The `--ff` option of pytest is not supported.

**R20.** The harness forces these settings in A and B: no warmup calls, one iteration, and `use_torch_compile`
off.

**R21.** Deadlines have default values, and a test can override them. The defaults are:
1. A: 10 minutes for the initialization, 20 minutes for the pipeline run;
2. B: 15 minutes for the initialization, 40 minutes for the pipeline run;
3. each replayed call: 2 minutes in A and 8 minutes in B, measured as the time since the last progress record.

### Tolerances and metrics

**R22.** A tolerance applies to one output of one component path. Every tensor output must have a tolerance. A
tensor output without a tolerance is a failure, and so is a tolerance for a name that is not an output. The
failure message lists the actual output names.
*Rationale:* there must be a guarantee that the test itself has no mistake and that it tests each output.

**R23.** A metric without a value gets a very small default tolerance, `1e-10`. The test then fails, and the
user sets a reasonable tolerance from the reported values.

**R24.** The test asserts the maximum absolute deviation and the relative L2 deviation. It reports the cosine
similarity without an assertion.

**R25.** The metrics are computed on the CPU in float64, separately for each rank and call. The test always
reports its metric values, also when it passes.

**R26.** Output names use the path notation relative to the return value: `[0]` for a tuple item, `.name` for
an attribute, and an empty name for a bare tensor. Nested containers flatten to their tensor leaves.

### Components, paths and calls

**R27.** The harness tests only an explicit list of components. The list is a list of strings. Each string is
the path from the runner model object to the component. Only the listed components are captured.

**R28.** The list is known before the run. The harness makes the test cases from the definition, without a
model.

**R29.** Path syntax: dots separate the elements, and `[i]` indexes a container, for example
`pipe.transformer.blocks[0].attn1` and `pipe.transformer.blocks[-1]`. A path has no leading dot. `[key]`
selects an item of an `nn.ModuleDict`.

**R30.** The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
`refiner_pipe`. The harness does not check that the attribute holds a pipeline. An attribute that holds `None`
is a path error.

**R31.** A call number follows the element after a colon, for example `pipe.transformer:3.blocks[0]:3`. A path
without a call number selects all calls, and its one case checks every call. The case fails if one call fails,
and the message names the failing calls.

**R32.** Call numbers are not negative.
*Rationale:* then the harness can save the tensors of a call at once.

**R33.** A path only tells how to reach the component. It does not describe the call stack, and the harness
never uses it to recover the call stack. Each path addresses only one component: the component that the path
reaches with its call numbers stripped. Call numbers only select calls of that component.

**R34.** Each callable element of each listed path gets one global call counter and a set of references to
counters of direct callable descendants (i.e. callable components that directly follow the element in any
path, ignoring non-callable components in between). The harness knows at path resolution which elements
are callable and what are their path descendants.
1. The counter is 0 before the pipeline starts.
2. The element call handler first increments the counter, then walks over its own direct descendant set and
   invoke "reset_counter()" for each of the descendants. Upon getting a "reset_counter()" request, a counter
   sets own call number to 0 and invokes "reset_counter()" on each of counters of its own descendants, so
   counter resetting propagates to each indirect descendant of the original element that started the reset
   chain. Once all descendant counters reset, i.e. counter reset chain finishes, the element makes all other
   checks, for example whether to save the call.
3. So the first call has the number 1, and no recorded call has the number 0. A counter value of 0 means that
   the element was never called since the last call of its parent in some path that was called chronologically
   the last.
4. Every call of the element counts, whatever code makes it.
5. When an element counter is found to belong to more than one set of references to counters of direct
   descendants (which means that the element for which it count calls is shared between several higher-level
   components) a harness issues a warning, explaining that interleaved calls through respective paths might
   not be counted properly due to the counter reset.

**R35.** `:i` refers to the global counter of its element. Example: `a.b:2.c.d[2]:3` selects a call of `d[2]`
when its counter is 3 and the counter of `b` is 2. During the pipeline run of A, the harness saves the inputs
of each selected call, and the outputs after the call returns.

**R36.** An address contains the call numbers of all callable elements that the harness tracks. The first path
element, such as `pipe` `second_pipe` or `refiner_pipe` is usually callable, but version 1 does not track it
due to complications, and assumes it to be a non-callable instead.

**R37.** The harness does not care who calls a component. It uses only the inputs and the outputs of each
call. Calls from another pipeline, from the runner, or from any other code are tested like all other calls.

**R38.** Two entries of the same component with different call numbers are separate cases, for example
`pipe.transformer:1` and `pipe.transformer:2`. Overlapping entries are an error at path resolution. Two
entries on the same component are disjoint only if one object has a call number in both entries and the
numbers differ. Then `pipe.transformer:1` and `second_pipe.transformer:2` in the LTX-2.3 runner are permitted.
*Rationale for the error:* it is better not to support such cases than to make the implementation more
complex.

**R39.** Calls such as `pipe.vae.decode`, where `.vae` isn't callable or not run and `decode` is a method,
also must be testable. A path through a module that the run never calls must be valid.

**R40.** A path must resolve in A and in B. If a path resolves only in A, its case gets FAILED_TO_START with
the path error. Path errors that need a built model show only after the initialization of A; this cost is
accepted.

**R41.** The harness can change the test case names before pytest sees them, if the `-k` grammar needs it.

### Instance A

**R42.** Instance A runs the component as production does: low-precision math, custom kernels, and the other
settings of the configuration. In version 1 it runs eagerly.

**R43.** One instance A does the capture and the replays. There is no separate capture instance. The capture
saves the CPU and CUDA random-number states before each call, and each replay restores them.

**R44.** A runs the pipeline once, captures the calls of the listed components, and replays them with the
captured inputs (a standalone replay).

**R45.** **Fidelity and determinism checks (A and B).** The instance replays a captured call once. If the
output equals the captured output bit for bit, the fidelity check passes. Otherwise it replays the call again:
1. if the two replays differ, the status is NON_DETERMINISTIC;
2. if the two replays are equal but differ from the capture, the status is REPLAY_MISMATCH.

Rare nondeterminism can give a false REPLAY_MISMATCH. This is a false failure, never a hidden one.

**R46.** By default, each test must be deterministic in A and in B. A test is one case: all calls or the
selected calls of one component path (R31, R38). Each test can have two determinism overrides:
1. The general override sets the policy for A. It also sets the policy for B if the test has no B override.
2. The B override sets the policy for B only. If it is set, it takes precedence over the general override.

If the policy requires determinism, NON_DETERMINISTIC fails the test at once. If the policy permits
nondeterminism, the harness prints a warning and continues to the tolerance checks. The bitwise checks always
run, whatever the policy is.

**R47.** REPLAY_MISMATCH always fails the test at once.

**R48.** If the harness cannot handle an argument (fp8 or MX tensors, types that it cannot serialize), only
the affected case gets UNSUPPORTED. The message names the argument.

### Instance B

**R49.** Instance B is an eager reference. It uses the same configuration and the same world size and parallel
layout as A, with these changes:
1. the reference dtype: fp32 by default; a parameter, because some models might need a smaller dtype; If uses
   fp32, B forces full fp32 matrix precision: `torch.backends.cuda.matmul.allow_tf32 = False`,
   `torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`.
2. quantization off, so B loads normal weights;
3. dense attention;
4. vendor packages blocked;
5. no compile, no step cache, no FP8 communication, no hybrid schedules.

The classification is: the trusted eager path replaces compile, the attention kernel, fused kernels, FP8
communication and the hybrid precision schedules. B keeps the parallel degrees and layout, the parallel VAE,
VAE tiling and its tile shape, distilled weights, LoRA and channels-last.

**R50.** A `TorchDispatchMode` audit checks the dtypes in B. It warns about custom calls and does not fail.

**R51.** B loads in bf16 and casts up to the reference dtype. It does not load an fp32 checkpoint. B converts
all components at once.

**R52.** B runs the pipeline in the reference dtype with the same run arguments as A (changes of R49 applied).
It captures its own calls and runs its own fidelity check on them.

**R53.** B replays the inputs of A at least two times and compares the replays bit for bit. Nondeterminism in
B gives NON_DETERMINISTIC_B.

**R54.** If the fidelity check of B fails for a case, only that case fails. The other cases continue.

**R55.** B has three parameters for the user: the reference dtype, the self-attention backend and the
cross-attention backend.

**R56.** B refuses an attention backend that declares a fallback. A backend that needs a blocked vendor
package fails at the start of B. A cross-attention parameter for a model without a separate cross-attention
backend is an error. The default cross-attention backend is `SDPA_MATH`.

### Failures, crashes and hangs

**R57.** The harness must detect each crash and each hang, wherever it occurs, and attribute it to the code
that ran last: the instance (A or B) and the case and component that it ran.

**R58.** Each instance computes the results of many cases in one process and saves them to files. A crash or a
hang ends the instance. Version 1 does not recover.
1. A case with complete data (A and B) is checked as usual.
2. A case without data gets FAILED_TO_START.
3. The case whose replay was running gets a status that shows what happened and in which instance, for example
   CRASHED_A, CRASHED_B, HUNG_A or HUNG_B.
4. A failure outside the replays gives FAILED_TO_START to every case without data, while propagating the
failure cause description with detailed attribution back to pytest and shown to the user.
5. After a failure of A, B does not start.

**R59.** A failure of B gets its own status, separate from a failure of A.

**R60.** An exception ends the instance with ERROR_A or ERROR_B and the traceback.

**R61.** A synchronization and a barrier at the end of each case, and progress records of each rank, attribute
crashes. A fault that silently damages memory and shows only in a later case is an accepted limit.

**R62.** Statuses can be extended later. UNSUPPORTED marks unsupported configurations. There is no
NOT_EVALUATED status. Users do not need to see status identifiers: the harness raises verbose exceptions with
explanations that a person can read.

**R63.** One launcher runs each instance, between the fixture and torchrun. It exits with 0 when it worked;
the detailed statuses go through another channel. A nonzero exit means that the launcher failed and that no
data exists, so every case of the configuration fails. The launcher enforces the deadlines.

**R64.** **Lock step.** When an instance has more than one rank, all ranks run every execution that can
contain a collective in the same order: the initialization, the pipeline run, each replay, and the collectives
of the harness. Work without collectives does not need lock step. Each per-rank decision that changes this
sequence is shared, or it ends the instance on all ranks through an exception.
*Rationale:* almost every execution of a pipeline or a component can contain a collective, and the harness
does not know this in advance.

**R65.** These decisions are shared across all ranks of the instance:
1. the support check of the configuration, in A and B;
2. the skip of a case before the replays: a path error, UNSUPPORTED from the capture of A, NOT_REACHED_B,
   inputs of A that B cannot use, and call lists that differ between ranks;
3. the second replay of a call;
4. the stop of a case after a call;
5. the number of replays of the inputs of A in B.

**R66.** If the call addresses of a case differ between the ranks, the case gets FAILED_TO_START on all ranks.
"Participating ranks" means all ranks of the instance. "The case fails at once" means that all ranks stop the
remaining replays of the case and go to the end of the case. B makes no replays of a case that it skips before
the replays.

**R67.** If a rank fails, torchrun stops the other ranks, and the status is the failure of the failed rank (or failed ranks, if several failed before torchrun reacted), not HUNG.

**R68.** If a rank of A or B does not do its work because a configuration or a case is UNSUPPORTED, the cause
and its details reach pytest as the skip reason of each affected case. For an unsupported configuration, that
is every selected case. The state keeps the reason for later reuse runs. The details are:
1. the instance and the ranks;
2. for a configuration: the feature, and the run argument with its value;
3. for a case: the call address, the argument name, and its type or dtype.

**R69.** Every run shows the UNSUPPORTED reasons without extra pytest options. A `pytest_terminal_summary`
hook adds a section that lists each UNSUPPORTED case with its reason.

**R70.** When several ranks report a skip, the reason lists each distinct reason with the ranks that found it.
The most severe status wins, in this order: FAILED_TO_START, NOT_REACHED_B, UNSUPPORTED.

**R71.** Each result that is not a pass names the rank that caused it. If a correct attribution is not
possible, the result shows the status of every rank, labeled with the rank number. In detail:
1. support and skip decisions: each rank that found the condition is a cause;
2. call lists that differ: the report lists the call addresses of every rank;
3. NON_DETERMINISTIC and REPLAY_MISMATCH of a call: the report gives the status of every rank, also of the
   passing ranks, and says that the source rank is unknown;
4. an exception or a fatal signal: one rank is named only if exactly that rank failed by itself; otherwise the
   report lists the failure of every rank with its time, kind and traceback;
5. a hang: one rank is named only if exactly that rank's last progress record differs from the others;
   otherwise the report lists the last progress record of every rank;
6. ranks with the same content share one line.

**R72.** One result path carries all results of the workers. Each worker appends result records to its own
progress file. After torchrun ends, the launcher reads the records of all ranks, merges them by case, call and
rank, and writes each state file. While an instance runs, the launcher is the only writer of state files.

**R73.** Worker records plus a small supervisor for each rank attribute crashes. An exception gives an error
record. A fatal signal that Python can catch gives a fault record through `faulthandler`. The rank supervisor
starts the worker as its child, records the time and the kind of the end of the worker, also after SIGKILL of the worker alone,
and records when torchrun sent SIGTERM. So the launcher can tell the ranks that torchrun stopped from the rank
that failed by itself.

**R74.** The status of a rank does not show which rank causes it, because collectives move data between ranks.
The specification states this limitation, and future work addresses it.

**R75.** A process that survives SIGKILL (uninterruptible sleep) must not hang the pytest session. After
SIGKILL, the supervisor waits only for a limited time, and then it reports the surviving processes. The
harness then reports the results of this configuration and stops the session: B does not start, the cases keep
the outcomes that the launcher wrote, cases without data get FAILED_TO_START with the list of surviving
processes, and the fixture stops the session before the next configuration. The stop message names the
configuration, the instance and the surviving processes, and says that the GPUs may need a reset.

**R76.** The harness itself must have a test set. The tests make sure that the instances and the launcher
handle each behavior of a component correctly, as the specification requires: a failed determinism check, a
failed fidelity check, a hang and a crash. The specification must list these tests.
*Rationale:* the instances have complex rules for correct behavior across ranks.

### pytest integration

**R77.** The harness is a set of pytest tests. The `pytest_integration` prototype is an implementation hint,
not a binding rule. The harness takes the best of it and adapts it.

**R78.** pytest outcomes: tolerance failures, NON_DETERMINISTIC, REPLAY_MISMATCH, CRASHED, HUNG and ERROR give
"failed"; FAILED_TO_START gives "error"; UNSUPPORTED gives "skipped". `record_property` stores the status.

**R79.** The test checks each rank independently against the same tolerance; compares A with the output of the
first replay of B.

**R80.** The test runs as many checks as possible before it fails, like the `EXPECT_` macros of GoogleTest. It
uses the installed `pytest-check` package. NON_DETERMINISTIC without the override and REPLAY_MISMATCH fail the
test at once. The failure message lists every call and rank with these statuses, and the case writes one
metrics line with its outcome only.

### Files, re-runs and metrics

**R81.** Each configuration writes its data into its own directory. Only the listed components are saved. A
test that passes deletes its files; only the files of failed tests stay. A failed test keeps its inputs and
both outputs. A crashed, hung or not-started case keeps the files that exist.

**R82.** One entity, in the launcher, defines the paths of the data files. The test functions get the paths
from it, read the files, and delete them.

**R83.** Tensors go into safetensors files. Structure and plain values go into JSON. Other types give
UNSUPPORTED for the affected case.

**R84.** The data goes into `_data/` and the metrics go into `_metrics/`. Both are in the directory of the
running test module. The pytest process resolves the paths one time. The launchers and the workers get
absolute paths.

**R85.** **Re-runs.** Each case has its own state. A fresh run gives each case the state ABOUT_TO_START, and
the state changes during the run.
1. If every recorded case passed (UNSUPPORTED counts as passed), the harness cleans the directory
   automatically and runs everything from the beginning.
   *Rationale:* a new run of tests that all passed gives no new information.
2. Otherwise, the harness prints a warning, does not run A and B, and runs the tests only on the data that
   still exists.
3. Each case acts by its stored state: a case that passed earlier is skipped with "passed in a previous run";
   a failed case with data is compared again; a case without data shows its stored status again; a case that
   is still ABOUT_TO_START is an error ("the previous run did not finish"); a case without a state is an error
   ("not part of the run that produced the data").
4. The user cleans the directory after a change of the settings. There is no fingerprint check.

**R86.** A marker file and a lock protect the data directory. Two pytest sessions must never use one
configuration directory at the same time, also during the cleanup.

**R87.** **Metrics file.** It lives in its own directory, apart from the dumps. It only grows, and cleaning
never deletes it. There is one file for each configuration name, in JSON Lines format, with UTC timestamps in
ISO 8601. Each append takes an `fcntl.flock` lock on the file.
*Rationale:* a metrics file that survives many runs shows how the deviations and the tolerances change while
the implementation changes.

**R88.** Each metrics line has at least these fields:
1. the timestamp and the run identifier;
2. the identifier of the run that produced the compared data;
3. the xDiT commit, and whether the working tree had uncommitted changes;
4. the GPU architecture;
5. the configuration name, the case path, the call address and the output name;
6. the metric name, and its values as a list with one value for each rank;
7. the tolerance in force;
8. the outcome.

A case without metrics writes one line with its outcome. A case that is skipped as "passed in a previous run"
writes nothing.
*Rationale:* a timestamp alone does not show which implementation gave a value; the commit does. The GPU
architecture matters because values differ between MI355X and MI300X. One line for each rank would make the
file too large to read.
