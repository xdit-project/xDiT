# Requirements and use cases of the component test harness

This document lists the use cases and the requirements of the xDiT component test harness.

Each item has an identifier. Use cases use `UC` and a sequence number. A requirement uses letters for the title of
its section and its position in the section multiplied by 10, for example `HL10` for the first requirement of
"High-level requirements". The steps of 10 leave space for new requirements.

## Use cases

**UC1. Check that a component works correctly.** A developer asks: "Does this component of this model
configuration work correctly?" The harness runs the component as production runs it (instance A) and as an
eager reference in high precision (instance B). It then compares the outputs against tolerances. Requirements:
HL10–HL60, TD10–TD40, TM10–TM50, IA10–IA80, IB10–IB80.
*Rationale:* some operations of xDiT diffusion models can be flaky (produce wrong numerics, or cause hangs or
crashes). Tests of single components are decoupled from the whole model.

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang.
The report names the instance and the case. At least for a deviation or nondeterminism, it also names the calls.
Requirements: HL60, IA40–IA60, IB50, FC10–FC160.

**UC3. Test the configurations of the CI benchmark files.** A test selects an entry of a CI benchmark YAML
file. The test can change any field of the entry. Requirements: TD20–TD40.
*Rationale:* the user specifies model configurations in the same way as the benchmark configurations.

**UC4. Cover many configurations of one model in one test module.** Requirements: TD50–TD80.
*Rationale:* there are hundreds of configurations in total, and most of them are very similar.

**UC5. Test sub-components and selected calls.** Examples are a transformer block, an attention module, the
VAE decode method, and one specific call of a transformer. Requirements: CP10–CP150.

**UC6. Tune tolerances over time.** A test reports its metrics in each run. A missing tolerance makes the test
fail, so the user sets tolerances from the reported values. The metrics history shows how the deviations
change while the implementation changes. Requirements: TM20–TM40, FR70, FR80.

**UC7. Run the tests again after failures.** A new run compares the stored data again, for example with new
tolerances, and does not run A and B again. After all cases pass, the next run starts from the beginning
automatically. Requirements: FR50, FR60.

**UC8. Select single cases.** The user selects cases with pytest, for example with `-k` or a node ID.
Requirements: CP150.

**UC9. Keep the tests valid while xDiT changes.** Requirements: HL20, HL30, HL90.
*Rationale:* xDiT changes fast, and its components get new augmentations all the time. Hard-coded tests become
obsolete.

**UC10. Accept nondeterminism that is valid.** An override of one test changes the result of nondeterminism in
that test from a failure to a warning. The determinism checks of all other tests stay active. B should have
its own override. Requirements: IA50, IA60.
*Rationale:* some operations are validly nondeterministic. A model can have one nondeterministic component
among many, and that component must be testable. Instance B generally uses different kernels that might have
other determinism expectations.

**UC11. Later use cases.** Compiled mode, a reference at world size 1 (B1), several input sets, MI300X and
other hardware, and benchmark and determinism scripts that use the foundation of the harness.

## High-level requirements

**HL10.** The harness tests individual components of xDiT diffusion models, for example a VAE, a transformer, a
transformer block, an attention module, a text encoder, or a method such as the VAE decode. It uses exactly
the configuration and the weights that the xDiT runner uses for a model configuration.

**HL20.** The harness takes each component from the model that the xDiT runner builds. It does not extract,
hard-code or build a component itself.
*Rationale:* the setup of a component depends on the model configuration. For example, the VAE of Wan2.2 can
be the VAE of `diffusers`, a DistVAE, or something else. Also, xDiT changes fast (UC9).

**HL30.** The details of a configuration are private to the configuration: Ulysses, ring, classifier-free
guidance, quantized GEMMs and others. The tests do not depend on them. The tests are generic, so that they can
run every configuration. The configuration identifier is one parameter among many.

**HL40.** The reference is the same xDiT component, built by the xDiT runner, run with extended precision (fp32
by default). It is not a `diffusers` implementation.

**HL50.** One test answers one question: "Does this component, in this regime, with these inputs, work
correctly?" A failed check belongs to one test, so only that test fails.

**HL60.** A test fails for a deviation beyond its tolerance, a crash, a hang, nondeterminism, or any other
detectable issue. "Flaky" includes a crash and a hang.

**HL70.** **Working tests first.** Test coverage has priority over correct reports, for example a correct call
address. A wrong report is acceptable as a documented limitation, possibly with a corresponding future-work
item. A deviation that a working test needs is permitted with the approval of the user.

**HL80.** Decisions rest on facts that the code shows directly.

**HL90.** Where possible, xDiT gets a refactor that makes components testable, instead of a hack around hidden
coupling. Examples: compile code only compiles, and the warmup is not part of the compile function.
In version 1, xDiT changes only where the first configuration needs a change.

**HL100.** Tests and benchmarks are separate from the entry point on. Benchmark scripts and determinism-check
scripts come later. They use the same foundation that the tests use to load and build a model and to get its
components. Version 1 plans nothing for benchmarks, but it identifies the code that benchmarks will use.

**HL110.** One model initialization of an instance serves the tests of many components.
*Rationale:* a model can need 5–10 minutes to load and initialize. That is too long for each test.

### Test definition

**TD10.** The definition is in Python. One call defines one configuration: its run settings, its list of
components, and the tolerances for each output of each component. Each configuration is a separate call.

**TD20.** The run settings come from an entry of a CI benchmark YAML file, in the same format as the benchmark
configurations. A helper reads the entry, builds the command-line flags as CI does, and parses them with the
parser of the xDiT runner.
*Rationale:* the parser applies its defaults and conversions, for example to
`gemm_quantization: low=fp4,high=fp6`.

**TD30.** A test can override every field of the YAML entry and it can be sufficient to define the
configuration alone, without a YAML file. `overrides` is a dictionary with the structure of a YAML entry. The
permitted fields are those of the `Experiment` class of CI. Other fields are an error. `args` merges key by
key: a key replaces the value of the entry, and `None` removes the key. Other fields replace the value of the
entry, and `None` removes the field.

**TD40.** The overrides apply both to A and to B. B has a few parameters of its own (IB70).

**TD50.** Each configuration has its own fixture, which runs A and B of that configuration. The tests of the
configuration run before the next configuration starts its instances.

**TD60.** The configuration name is the `name` of the entry by default. It is an error if neither a name nor an
entry is set, or if the name is absent, `None`, not a string, or empty. The name selects the data directory
and the metrics file of the configuration.

**TD70.** Configuration names are unique. The harness raises an error for two configurations with the same
name. The check runs during collection, and it also covers the configurations that `-k` deselects. A separate
check in the configuration fixture stops a configuration that starts a second time in one session. Unique
names across sessions are the responsibility of the user.

**TD80.** A name contains only letters, digits, `_`, `.` and `-`. The literal names `.` and `..` are not
allowed. The `--ff` option of pytest is not supported.

**TD90.** The harness forces these settings in A and B: no warmup calls, one iteration, and `use_torch_compile`
off.

**TD100.** Deadlines have default values, and a test can override them. The defaults are:
1. A: 10 minutes for the initialization, 20 minutes for the pipeline run;
2. B: 15 minutes for the initialization, 40 minutes for the pipeline run;
3. each replayed call: 2 minutes in A and 8 minutes in B, measured as the time since the last progress record.

### Tolerances and metrics

**TM10.** A tolerance applies to one output of one component path. Every tensor output must have a tolerance. A
tensor output without a tolerance is a failure, and so is a tolerance for a name that is not an output. The
failure message lists the actual output names.
*Rationale:* there must be a guarantee that the test itself has no mistake and that it tests each output.

**TM20.** A metric without a value gets a very small default tolerance, `1e-10`. The test then fails, and the
user sets a reasonable tolerance from the reported values.

**TM30.** The test asserts the maximum absolute deviation and the relative L2 deviation. It reports the cosine
similarity without an assertion.

**TM40.** The metrics are computed on the CPU in float64, separately for each rank and call. The test always
reports its metric values, also when it passes.

**TM50.** Output names use the path notation relative to the return value: `[0]` for a tuple item, `.name` for
an attribute, and an empty name for a bare tensor. Nested containers flatten to their tensor leaves.

### Components, paths and calls

**CP10.** The harness tests only an explicit list of components. The list is a list of strings. Each string is
the path from the runner model object to the callable component or a method. Only the listed components are captured.

**CP20.** The list is known before the run. The harness makes the test cases from the definition, without a
model.

**CP30.** Path syntax: dots separate the elements, and `[i]` indexes a container, for example
`pipe.transformer.blocks[0].attn1` and `pipe.transformer.blocks[-1]`. A path has no leading dot. `[key]`
selects an item of an `nn.ModuleDict`.

**CP40.** The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
`refiner_pipe`. The harness does not check that the attribute holds a pipeline. An attribute that holds `None`
is a path error.

**CP50.** A call number follows the element after a colon, for example `pipe.transformer:3.blocks[0]:3`. A path
without a call number selects all calls, and its one case checks every call. The case fails if one call fails,
and the message names the failing calls.

**CP60.** Call numbers are not negative.
*Rationale:* then the harness can save the tensors of a call at once.

**CP70.** A path only tells how to reach the component. It does not describe the call stack, and the harness
never uses it to recover the call stack. Each path addresses only one component: the component that the path
reaches with its call numbers stripped. Call numbers only select calls of that component.

**CP80.** Each callable element of each listed path gets one global call counter and a set of references to
counters of direct callable descendants (i.e. callable components that directly follow the element in any
path, ignoring non-callable components in between). The harness knows at path resolution which elements
are callable and what their path descendants are.
1. The counter is 0 before the pipeline starts.
2. The element call handler first increments the counter, then walks over its own direct descendant set and
   invokes "reset_counter()" for each of the descendants. Upon getting a "reset_counter()" request, a counter
   sets own call number to 0 and invokes "reset_counter()" on each of counters of its own descendants, so
   counter resetting propagates to each indirect descendant of the original element that started the reset
   chain. Once all descendant counters reset, i.e. counter reset chain finishes, the element makes all other
   checks, for example whether to save the call.
3. So the first call has the number 1, and no recorded call has the number 0. A counter value of 0 means that
   the element was never called since the last call of its parent in some path that was called chronologically
   the last.
4. Every call of the element counts, whatever code makes it.
5. When an element counter is found to belong to more than one set of references to counters of direct
   descendants (which means that the element whose calls it counts is shared between several higher-level
   components) the harness issues a warning, explaining that interleaved calls through respective paths might
   not be counted properly due to the counter reset.

**CP90.** `:i` refers to the global counter of its element. Example: `a.b:2.c.d[2]:3` selects a call of `d[2]`
when its counter is 3 and the counter of `b` is 2. During the pipeline run of A, the harness saves the inputs
of each selected call, and the outputs after the call returns.

**CP100.** An address contains the call numbers of all callable elements that the harness tracks. The first path
element, such as `pipe`, `second_pipe` or `refiner_pipe` is usually callable, but version 1 does not track it
due to complications, and assumes it to be a non-callable instead.

**CP110.** The harness does not care who calls a component. It uses only the inputs and the outputs of each
call. Calls from another pipeline, from the runner, or from any other code are tested like all other calls.

**CP120.** Two entries of the same component with different call numbers are separate cases, for example
`pipe.transformer:1` and `pipe.transformer:2`. Overlapping entries are an error at path resolution. Two
entries on the same component are disjoint only if one object has a call number in both entries and the
numbers differ. Then `pipe.transformer:1` and `second_pipe.transformer:2` in the LTX-2.3 runner are permitted.
*Rationale for the error:* it is better not to support such cases than to make the implementation more
complex.

**CP130.** Calls such as `pipe.vae.decode`, where `.vae` isn't callable or doesn't run in the pipeline,
and `decode` is a method, also must be testable. A path through a module that the run never calls must be valid.

**CP140.** A path must resolve in A and in B. If a path resolves only in A, its case gets FAILED_TO_START_B with
the path error. Path errors that need a built model show only after the initialization of A; this cost is
accepted.

**CP150.** The harness can change the test case names before pytest sees them, if the `-k` grammar needs it.

### Instance A

**IA10.** Instance A runs the component as production does: low-precision math, custom kernels, and the other
settings of the configuration. In version 1 it runs eagerly.

**IA20.** One instance A does the capture and the replays. There is no separate capture instance. The capture
saves the CPU and CUDA random-number states before each call, and each replay restores them.

**IA30.** A runs the pipeline once, captures the calls of the listed components, and replays them with the
captured inputs (a standalone replay).

**IA40.** **Fidelity and determinism checks (A and B).** The instance replays a captured call once. If the
output equals the captured output bit for bit, the fidelity check passes. Otherwise it replays the call again:
1. if the two replays differ, the status is NON_DETERMINISTIC_A or NON_DETERMINISTIC_B (depending on
   the instance), collectively called NON_DETERMINISTIC;
2. if the two replays are equal but differ from the capture, the status is REPLAY_MISMATCH_A or
   REPLAY_MISMATCH_B (depending on the instance), collectively called REPLAY_MISMATCH.

Rare nondeterminism can give a false REPLAY_MISMATCH. This is a false failure, never a hidden one.

**IA50.** By default, each test must be deterministic in A and in B. A test is one case: all calls or the
selected calls of one component path (CP50, CP120). Each test can have two determinism overrides:
1. The general override sets the policy for A. It also sets the policy for B if the test has no B override.
2. The B override sets the policy for B only. If it is set, it takes precedence over the general override.

If the policy of an instance requires determinism, the NON_DETERMINISTIC status of that instance fails the
test at once. If it permits nondeterminism, the harness prints a warning and continues to the tolerance checks.
The bitwise checks always run, whatever the policy is.

**IA60.** REPLAY_MISMATCH always fails the test at once.

**IA70.** UNSUPPORTED marks a known limit of the harness:
1. If the harness cannot handle an argument (fp8 or MX tensors, types that it cannot serialize), only the
   affected case gets UNSUPPORTED. The message names the argument.
2. If the harness cannot handle a feature of a configuration, every case of the configuration gets
   UNSUPPORTED_A if A cannot handle the feature, otherwise UNSUPPORTED_B. The message names the feature and the
   run argument with its value. The harness finds this at collection, from the run settings (TD20), or when A or
   B loads the configuration. After that, no instance of the configuration starts or continues. If no case gets
   SUCCESS_A, B does not start (FC20 item 5), so a feature that only B can find stays unknown in that run.

**IA80.** After passing the end-of-case barrier of FC50, a case replaces its status with (the only) successful status
SUCCESS_A or SUCCESS_B (depending on the instance), collectively called SUCCESS, if it passed
all the checks (including getting NON_DETERMINISTIC under a policy that permits it), has done all
the replays of the case, and successfully dumped all the necessary data.

### Instance B

**IB10.** Instance B is an eager reference. It uses the same configuration and the same world size and parallel
layout as A, with these changes:
1. the reference dtype: fp32 by default; a parameter, because some models might need a smaller dtype; If it uses
   fp32, B forces full fp32 matrix precision: `torch.backends.cuda.matmul.allow_tf32 = False`,
   `torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`.
2. quantization off, so B loads normal weights;
3. dense attention;
4. vendor packages blocked;
5. no compile, no step cache, no FP8 communication, no hybrid schedules.

The classification is: the trusted eager path replaces compile, the attention kernel, fused kernels, FP8
communication and the hybrid precision schedules. B keeps the parallel degrees and layout, the parallel VAE,
VAE tiling and its tile shape, distilled weights, LoRA and channels-last.

**IB20.** A `TorchDispatchMode` audit checks the dtypes in B. It warns about custom calls and does not fail.

**IB30.** B loads in bf16 and casts up to the reference dtype. It does not load an fp32 checkpoint. B converts
all components at once.

**IB40.** B runs the pipeline in the reference dtype with the same run arguments as A (changes of IB10 applied).
It captures its own calls and runs its own fidelity check on them.

**IB50.** B replays the inputs of A at least two times and compares the replays bit for bit. Nondeterminism in
B gives NON_DETERMINISTIC_B.

**IB60.** If a case gets REPLAY_MISMATCH_B, or NON_DETERMINISTIC_B under a policy that requires determinism, only
that case fails. The other cases continue.

**IB70.** B has three parameters for the user: the reference dtype, the self-attention backend and the
cross-attention backend.

**IB80.** An attention backend that declares a fallback, or that needs a blocked vendor package, makes the
configuration UNSUPPORTED_B (IA70 item 2). A cross-attention parameter for a model without a separate
cross-attention backend is an error. The default cross-attention backend is `SDPA_MATH`.

### Failures, crashes and hangs

**FC10.** The harness must detect each crash and each hang, wherever it occurs, and attribute it to the
instance (A or B) and, for a failure during a replay, to the case, the component and the call (FC20.3).

**FC20.** Each instance computes the results of many cases in one process and saves them to files. A crash or a
hang ends the instance. Version 1 does not recover.
1. A case with complete data (A and B) is checked as usual.
2. A case that an instance had to run but did not start gets FAILED_TO_START.
3. The case whose replay was running gets a status that shows what happened and in which instance, for example
   CRASHED_A, CRASHED_B, HUNG_A or HUNG_B. The message names the last call of each rank, if available (such a
   call is not a proven cause; a documented limitation, see HL70).
4. A failure outside the replays gives FAILED_TO_START to every case that the instance did not start.
   The message names the instance and the kind of failure (crash or hang).
5. B runs only on cases with SUCCESS_A (IA80). If there are no such cases, or if processes of A survive
   SIGKILL (FC160), B does not start.

**FC30.** Each status that an instance gives has a form for A and a form for B, for example CRASHED_A and
CRASHED_B, collectively called CRASHED. NOT_REACHED_IN_B exists only for B.

**FC40.** An exception that ends the instance counts as a crash (FC20).

**FC50.** A synchronization and a barrier at the end of each case, and progress records of each rank, attribute
crashes. Before each replayed call, each rank writes a progress record with the call address. A fault that
silently damages memory and shows only in a later case is an accepted limit.

**FC60.** Statuses can be extended later. There is no
NOT_EVALUATED status. Users do not need to see status identifiers: the harness raises verbose exceptions with
explanations that a person can read.

**FC70.** One launcher runs each instance, between the fixture and torchrun. It exits with 0 when it worked;
the detailed statuses go through another channel. A nonzero exit means that the launcher failed and that no
data exists, so every case of the configuration fails triggering a pytest error. The launcher enforces the deadlines.

**FC80.** **Lock step.** When an instance has more than one rank, all ranks run every execution that can
contain a collective in the same order: the initialization, the pipeline run, each replay, and the collectives
of the harness. Work without collectives does not need lock step. Each per-rank decision that changes this
sequence is shared, or it ends the instance on all ranks through an exception.
*Rationale:* almost every execution of a pipeline or a component can contain a collective, and the harness
does not know this in advance.

**FC90.** These decisions are shared across all ranks of the instance:
1. the support check of the configuration, in A and B;
2. the skip of a case before the replays: a path error, UNSUPPORTED_A or UNSUPPORTED_B from the capture of A
   or B, UNSUPPORTED_B for inputs of A that B cannot use, and call lists that differ between ranks;
3. the second replay of a call;
4. the stop of a case after a call;
5. the number of replays of the inputs of A in B;
6. the failure of a case in B when the pipeline run of B on any rank did not reach a call that A captured
   (NOT_REACHED_IN_B). B makes no replays of the case (FC100).

**FC100.** If the call addresses of a case differ between the ranks, the case gets FAILED_TO_START on all ranks.
B makes no replays of a case that it skips before the replays.

**FC110.** If a rank fails, torchrun stops the other ranks, and the failure is a crash, not a hang.

**FC120.** If a rank of A or B does not do its work because a configuration or a case is UNSUPPORTED, the cause
and its details reach pytest as the skip reason (UNSUPPORTED_A) or the error message (UNSUPPORTED_B) of each
affected case. If a single rank gets UNSUPPORTED status all other ranks must also get UNSUPPORTED (shared
decision, FC90).
The state keeps the reason for later reuse runs. The details are:
1. the instance and the ranks;
2. for a configuration: the feature, and the run argument with its value;
3. for a case: the call address, the argument name, and its type or dtype.

**FC130.** Every run shows the UNSUPPORTED reasons without extra pytest options. A `pytest_terminal_summary`
hook adds a section that lists each UNSUPPORTED case with its reason.

**FC140.** If the ranks report different statuses for one case, a single status that is not a SUCCESS decides
the outcome of the whole case (PI30). The message still shows all different statuses of ranks along with ranks
that found them.

**FC150.** One result path carries all results of the workers. Each worker appends result records to its own
progress file. After torchrun ends, the launcher reads the records of all ranks, merges them by case, call and
rank, and writes each state file. While an instance runs, the launcher is the only writer of state files.

**FC160.** A process that survives SIGKILL (uninterruptible sleep) must not hang the pytest session. After
SIGKILL, the harness waits only for a limited time, and then it reports the surviving processes. The
harness then reports the results of this configuration and stops the session: if the processes belong
to A, B does not start, and each case with SUCCESS_A gets FAILED_TO_START_B with the list of surviving
processes, and the fixture stops the session before the next configuration. The stop message names the
configuration, the instance and the surviving processes, and says that the GPUs may need a reset.

**FC170.** The harness itself must have a test set. The tests make sure that the instances and the launcher
handle each behavior of a component correctly, as the specification requires: a failed determinism check, a
failed fidelity check, a hang and a crash. The specification must list these tests.
*Rationale:* the instances have complex rules for correct behavior across ranks.

### pytest integration

**PI10.** The harness is a set of pytest tests. The `pytest_integration` prototype is an implementation hint,
not a binding rule. The harness takes the best of it and adapts it.

**PI20.** **General meaning of the pytest outcomes.** (This requirement exist to alleviate decisions
for new statuses. Actual mapping is provided by PI30) "error" means that the test has nothing to check, for example
because an instance that had to replay the case did not start its replays. "failed" means that an instance
replayed the case, and the instance or the test found a problem. Documented exceptions are possible:
1. UNSUPPORTED_A gives "skipped".
2. A configuration that is UNSUPPORTED_B gives "error" to every case, whatever its statuses from A are.

*Rationale:* UNSUPPORTED_A is a known limit of the harness, not a problem of the component. B is the reference
and must handle everything A handles, so UNSUPPORTED_B is not a reason to skip. If B cannot handle the configuration, no
case of the configuration can have a meaningful test.

**PI30.** pytest outcomes (PI20; for re-runs FR50 takes precedence):
1. A failure of the configuration fixture (TD70, FC70, FR60), or a configuration that is UNSUPPORTED_B (IA70 item 2),
   gives "error" to every case of the configuration, whatever its other statuses are. For UNSUPPORTED_B, the
   message of each case also lists its statuses from A.
2. Otherwise, each case gets its own outcome decided in the following order:
   1. "error" if any rank has FAILED_TO_START, UNSUPPORTED_B, or NOT_REACHED_IN_B. Also if a rank of A
      has UNSUPPORTED_A, but some other rank of A doesn't have it. The case fixture (function scope) raises
      the exception before the test function starts, because pytest gives "error" only for an exception in
      setup, teardown or a fixture.
   2. "skipped" for UNSUPPORTED_A on all ranks of A.
   3. "failed" if not all ranks of A and B have SUCCESS (PI50), or for a failed check.
   4. "passed" otherwise
3. `record_property` stores the statuses.
4. If a pytest outcome mandates an exception (i.e. it's a "failure" or "error"), the exception message must contain
   statuses of all ranks and all other applicable information mandated by other requirements.

**PI40.** The test checks each rank independently against the same tolerance; compares A with the output of the
first replay of B.

**PI50.** If a status gives "failed" (PI30.2.3), the test fails at once. The failure message
lists every call and rank with its status, and the case writes one metrics line with its outcome only.
Otherwise, the test runs as many checks as possible before it fails, like the `EXPECT_` macros of
GoogleTest. It uses the installed `pytest-check` package. After the checks, it writes the metrics
lines with the values and the outcome.

### Files, re-runs and metrics

**FR10.** Each configuration writes its data into its own directory. Only the listed components are saved. A
test that passes deletes its files; the files of tests that did not pass stay.

**FR20.** One entity, in the launcher, defines the paths of the data files. The test functions get the paths
from it, read the files, and delete them.

**FR30.** Tensors go into safetensors files. Structure and plain values go into JSON. Other types give
UNSUPPORTED for the affected case.

**FR40.** The data goes into `_data/` and the metrics go into `_metrics/`. Both are in the directory of the
running test module. The pytest process resolves the paths one time. The launchers and the workers get
absolute paths.

**FR50.** **Re-runs.** Each case has its own state. A fresh run gives each case the state ABOUT_TO_START, and
the state changes during the run.
1. If every recorded case passed (a case that is UNSUPPORTED_A counts as passed), the harness
   cleans the directory automatically and runs everything from the beginning.
   *Rationale:* a new run of tests that all passed gives no new information.
2. Otherwise, the harness prints a warning, does not run A and B, and runs the tests only on the data that
   still exists.
3. Each case acts by its stored state: a case that passed earlier is skipped with "passed in a previous run";
   a failed case with data is compared again; a case without data shows its stored statuses again; a case that
   is still ABOUT_TO_START is an error ("the previous run did not finish"); a case without a state is an error
   ("not part of the run that produced the data").
4. The user cleans the directory after a change of the settings. There is no fingerprint check.

**FR60.** A marker file and a lock protect the data directory. Two pytest sessions must never use one
configuration directory at the same time, also during the cleanup.

**FR70.** **Metrics file.** It lives in its own directory, apart from the dumps. It only grows, and cleaning
never deletes it. There is one file for each configuration name, in JSON Lines format, with UTC timestamps in
ISO 8601. Each append takes an `fcntl.flock` lock on the file.
*Rationale:* a metrics file that survives many runs shows how the deviations and the tolerances change while
the implementation changes.

**FR80.** Each metrics line has at least these fields:
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
