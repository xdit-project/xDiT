# Requirements and use cases of the component test harness

This document lists the use cases and the requirements of the xDiT component test harness.

Each item has an identifier. Use cases use `UC` and a sequence number. A requirement uses letters for the title of
its section and its position in the section multiplied by 10, for example `HL10` for the first requirement of
"High-level requirements". The steps of 10 leave space for new requirements.

## Use cases

**UC1. Check that a component works correctly.** A developer asks: "Does this component of this model
configuration work correctly?" The harness runs the component as production runs it (instance A) and as an
eager reference in potentially extended precision (instance B). It then compares the outputs against tolerances. Requirements:
HL10–HL60, TD10–TD40, TM10–TM50, IA10–IA80, IB10–IB80.
*Rationale:* some operations of xDiT diffusion models can be flaky (produce wrong numerics, or cause hangs or
crashes). Component-wise testing could help with QA of the implementation.

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang.
The report names the instance and the case. At least for a deviation or nondeterminism, it also names the calls.
Requirements: HL60, IA40–IA60, IB50, FC10–FC160.

**UC3. Test the configurations of the CI benchmark files.** Tests run on selected components
of a model that is configured by an entry in a CI benchmark YAML file, or directly.
Requirements: TD20–TD40.
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

## Requirements

### High-level requirements

**HL10.** The harness tests individual components of xDiT diffusion models, for example a VAE, a transformer, a
transformer block, an attention module, a text encoder, or a method such as the VAE decode. It uses exactly
the configuration and the weights that the xDiT runner uses for a model configuration.

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
correctly?" A failed check belongs to one test, so only that test fails (wider failures due to
crashes or hangs are an accepted limitation of version 1).

**HL60.** A test fails for a deviation beyond its tolerance, a crash, a hang, nondeterminism, or any other
detectable issue. "Flaky" includes a crash and a hang.

**HL70.** **Working tests first.** Test coverage has priority over correct reports, for example a correct call
identifier. A wrong report is acceptable as a documented limitation, possibly with a corresponding future-work
item. A deviation from some requirements that a working test needs is permitted with a due documentation.

**HL80.** **Traceability and debuggability.** Each rank executing a model and its components
should save a structured log of all events and situations related to execution decisions and
defined by this document.

**HL90.** Where possible, xDiT gets a refactor that makes components testable, instead of a hack around hidden
coupling. Examples: compile code only compiles, and the warmup is not part of the compile function.
In version 1, xDiT changes only where the first configuration needs a change.

**HL100.** Tests and benchmarks are separate from the entry point on. Benchmark scripts and determinism-check
scripts come later. They use the same foundation that the tests use to load and build a model and to get its
components. Version 1 plans nothing for benchmarks, but it identifies the code that benchmarks will use.

**HL110.** One model load and initialization serves the tests of many, preferably all, components.
A process that loads and initializes a model, and dumps components execution data for testing later
is called "an instance". There are two instances: an instance A works with the model to test, and
an instance B works with the same model as A, but configured to perform computations in a higher
precision to obtain reference values to compare against.
*Rationale:* it may take 5–10 minutes to fully load and initialize a model, so we have to perform
test data acquisition in batches: one batch gathers data to test (A), and the other batch gather
reference data (B).

**HL120.** Multi-node model execution isn't supported, but single node multi-GPU is supported,
hence a term "rank" is an alias to "local rank". A term "worker" is an alias to "local rank" too.

#### Statuses

**HLS10.** When a rank executes a model or does something else mandated by this document, it gets a status label that identifies a certain event, situation, state or a condition, that the rank seen. Statues are written in CAPITAL_LETTERS. Status only describe what the rank observed last. When this document say "status applies to a case" or "something gets status X", it describes the state the rank was in during or after execution of the case or that something.

**HLS20.** This document follows a convention: when a certain status X has _A or _B suffix, that means
that status X applies to a rank of A or B correspondingly. There are no distinct statuses
X_A or X_B, there's only a status X and a rank of instance A or B on which it appears.
It's always known to which instance a rank belongs, so instance specific statuses aren't
generally needed.

**HLS30.** Status is written to the rank's log file immediately when it happens (FC150) as `status=X` record (the record should have other fields providing additional information). In the rank's log, when `status=` field is recorded with `case=` entry (setting a current test case identifier), the status applies to the test case also. It's an error to have a `case=` field without `status=` field.

**HLS40.** When the harness processes data obtained from instances A and B, it loads individual case data
from each of the instances and by aggregating statuses from all ranks decides how it's going
to handle case testing. During that, rank statuses might trigger standard pytest
states: error (an exception is thrown from setup/teardown or fixture code),
failure (an exception is thrown from a test function), test skip or test pass. If an exception
is raised, it's message contain all important information to explain the reason for that.

**HLS50.** In some cases a rank getting a certain status may cause a "shared decision", making after
a synchronization point all other rank to not do what they were going to do (for example,
to skip case execution, or a replay). These decisions don't affect these other rank statuses (they
keep their own); they only affect on what the ranks will do next. When a shared decision is made,
if it is different from what a rank was going to do, the rank logs it into their own log file.

### Tests definition

**TD10.** The definition is in Python. One call defines one configuration: its xDiT settings,
its list of components to test, the tolerances for each output of each component, special parameters
of B instance, per component path determinism overrides, deadlines and other things the
requirements mandate to be user-overridable.
Each entry (a component path) of the list of components defines a separate test case,
selectable with `-k` pytest parameter.
Each configuration is a separate call.

**TD20.** The xDiT settings usually come from an entry of a CI benchmark YAML file. The `yaml_path` configuration (TD10) parameter defines path to the file (in the same format as the benchmark configurations), and the `entry` field defines name of the entry in the file. There's a helper that reads an entry from a file.  If both `yaml_path` and `entry` are `None` (default), helper returns an empty dictionary. If `yaml_path` can be read and it has an entry with `name` field equal to the `entry`, - the helper returns the read dictionary of xDiT settings. Otherwise the helper raise an exception.

**TD30.** Test configuration call (TD10) can override every field of the xDiT settings, and overrides can be sufficient to define these settings alone, without a YAML file. `overrides` is a dictionary with the structure of a YAML entry. The permitted fields are only those of the `Experiment` class of CI; other fields are an error. After the helper of TD20 returns a dictionary read from `yaml_path` and `entry`, the `overrides` is used to override some fields: `args` key is a dictionary and it is merged key by key: if its key is `None`, the corresponding key is remove from final dictionary, otherwise its value replaces the value in the final xDiT settings dictionary. Similarly, other fields replace the value of the entry, and `None` removes the field.

**TD40.** The resulting xDiT settings dictionary of TD30 apply both to A and to B. Each configuration has its own fixture, which applies instance-specific overrides (TD90, IB10, IB70) to the xDiT settings, launches instances A and B, processes results and lets individual test fixtures + test cases of the configuration run later.

**TD50.** After the configuration fixture (TD40) prepares instance-specific xDiT settings, it passes them to the instance. The instance uses these settings to build the command-line flags as CI does, and then to parse them with the parser of the xDiT runner.

*Rationale:* the xDiT parser applies its defaults and conversions, for example to `gemm_quantization: low=fp4,high=fp6`, so it must not be skipped.

**TD60.** The configuration call has no explicit `name` parameter. The configuration name is the `name` field of the merged xDiT settings (TD30). Hence, if `yaml_path` and `entry` aren't used (i.e. have default `None` value), the `overrides` parameter must have a valid `name` value. The `name` value of the merged xDiT settings (TD30) becomes the configuration name. The name selects the data directory and the metrics file of the configuration.

**TD70.** Configuration names are unique. The harness raises an error for two configurations with the same
name. The check runs during collection, and it also covers the configurations that `-k` deselects. A separate
check in the configuration fixture stops a configuration that starts a second time in one session. Unique
names across sessions are the responsibility of the user.

**TD80.** A name contains only letters, digits, `_`, `.` and `-`. The literal names `.` and `..` are not
allowed. The `--ff` option of pytest, `pytest-xdist` and random-order pytest plugins are not supported.

**TD90.** The harness forces these settings in A and B: no warmup calls, one execution ("iteration"
in terms of xDiT runner), and `use_torch_compile` off.

**TD100.** Deadlines have default values, and a configuration can override them. The defaults are:
1. A: 10 minutes for the initialization, 20 minutes for the pipeline run;
2. B: 15 minutes for the initialization, 40 minutes for the pipeline run;
3. each replayed call (measured between pre-call and post-call barriers): 2 minutes in A and
   5 minutes in B.

**TD110.** **Deadlines enforcement.** The configuration fixture (TD40) launches each instance sequentially using a supplementary launcher process that takes care of deadlines and child processes cleanup. Each rank of the instance writes its deadline -related progress and timestamps to a certain rank-specific log file known to the launcher. The launcher then periodically reads these files and uses progress records and timestamps to enforce deadlines. A missed deadline leads to a warning message and killing the instance and all its descendant processes.

**TD120.** Since the launcher is the only entity that knows of missed deadlines, and also
only it can detect crashes or other premature exits, it's responsible for maintaining proper
rank statuses for the cases that were executed last according to the rank progress reports,
upon the instance finishing execution (whatever the reasons for instance process ending were).

**TD130.** If `HSA_DISABLE_COREDUMP_ON_EXCEPTION` environment variable isn't defined for the
launcher process, it must set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` for self and all child
processes it runs.

**TD140.** For instance B only (to reduce memory pressure), if none of `PYTORCH_ALLOC_CONF`,
`PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` environment variables are defined,
the launcher sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for self and all child
processes it runs.

**TD200.** Configurations with step caching or hybrid schedules aren't supported.

### Tolerances and metrics

**TM10.** Each output of each component path for instance A is compared to that of instance B
taken as a reference (see "Instance A" and "Instance B" below). Two comparison metrics are
computed and asserted:
1. maximum absolute deviation (requires absolute tolerance value, default is `1e-10`) and
2. the relative L2 deviation (requires relative tolerance value, default is `1e-10`)

For tensors, the cosine similarity is also computed but reported without an assertion.

**TM20.** Each output of each component path must be named in the configuration, and may have
a tolerance override. An output of a component that doesn't have a set name is an error, so
is a name that is not an output.
*Rationale:* there must be a guarantee that the test itself has no mistake and that it tests each output.

**TM30.** The metrics are computed on the CPU in float64, separately for each rank and call. The test always
reports its metric values, also when it passes.

**TM40.** Output names use the path notation relative to the return value: `[0]` for a tuple item, `.name` for
an attribute, and an empty name for a bare tensor. Nested containers flatten to their tensor leaves.

**TM50.** A proposed way to specify a list of component paths, outputs and tolerances, to a
configuration call (TD10) is as a python dictionary where keys are components paths and values are
dictionaries mapping component output names to a named tuple (abs, rel) of tolerance overrides
to apply.

### Components, paths and calls

**CP10.** The harness tests only components mentioned in an explicit list of component path strings. Each path string defines a component to test and is the path from the runner model object to the callable component or a method. Only the listed components are captured. A listed component must run at least once during pipeline run.

**CP20.** The list is known before the run. The harness makes the test cases from the list only (without a
model) and list's items (component paths) serve as test case identifiers, so there is an exact 1-to-1 relationship between a component path and a test case.

**CP30.** Component path syntax: dots separate the path
elements, and `[i]` indexes a container, for example
`pipe.transformer.blocks[0].attn1` and `pipe.transformer.blocks[-1]`. A path has no leading dot. `[key]`
selects an item of an `nn.ModuleDict`.

**CP40.** The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
`refiner_pipe`. The harness does not check that the attribute holds a pipeline. An attribute that holds `None`
is a path error.

**CP50.** A call number follows the element after a colon, for example `pipe.transformer:3.blocks[0]:3`. A path
without a call number selects all calls, and its one case checks every call. The case fails if one call fails,
and the message names the failing calls. Version 1 has accepted limitation: the first path element
ignores its call number.

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
1. The counter is 0 before the pipeline starts and can only be modified during pipeline run (replays excluded).
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
to data files (it doesn't store them in memory). These data files are read later for replays and
checks. Only the corresponding test case function can delete these data files.

**CP100.** An address contains the call numbers of all callable elements of a path that the
harness tracks. The first path element, such as `pipe`, `second_pipe` or `refiner_pipe` is
usually callable, but version 1 does not track it due to complications, and assumes it to
be a non-callable instead.

**CP110.** The harness does not care who calls a component. It uses only the inputs and the outputs of each
call. Calls from another pipeline, from the runner, or from any other code are tested like all other calls.

**CP120.** Two entries of the same component with different call numbers are separate cases, for example
`pipe.transformer:1` and `pipe.transformer:2`. Overlapping entries are an error at path resolution. Two
entries on the same component are disjoint only if one object has a call number in both entries and the
numbers differ. Then `pipe.transformer:1` and `second_pipe.transformer:2` in the LTX-2.3 runner
are permitted.
*Rationale for the error:* it is better not to support such cases than to make the implementation more
complex.

**CP130.** Calls such as `pipe.vae.decode`, where `.vae` isn't callable or doesn't run in the pipeline,
and `decode` is a method, also must be testable. A path through a module that the run never calls must be valid.

**CP140.** A path must resolve in A and in B. If a path resolves only in A, its case gets FAILED_TO_START_B with
the path error. Path errors that need a built model show only after the initialization of A; this cost is
accepted.

**CP150.** The harness can change the test case names before pytest sees them, if the `-k` grammar needs it.

### Rank lifecycle

**RL10.** In its lifetime a rank comes through a sequence of phases and cross-rank synchronization points (in which all ranks must participate), described in order below. Getting to the next phase is reflected in the rank's log with a corresponding entry `phase=`.

**RL20.** When a rank just starts, it gets a status STARTED.

**RL30.** **Configuration parsing phase.** This is where the rank parses its xDiT setting (TD50) and determines if it can run the configuration. If the configuration gets `status=UNSUPPORTED` record with a `reason=` field describing what exactly isn't supported, and then rank exits with an error (no special synchronization points are needed). This automatically forces torchrun to kill all other ranks. By reading the log files the launcher is able to correctly identify the cause and not render rank statuses as CRASH.

**RL40.** **Model loading phase.** Doesn't have specific statuses and synchronization points.

**RL50.** **Model run phase.** During this phase rank runs the model pipeline and captures data.

4. Replays phase

1. During pipeline execution ranks doesn't communicate with each other, but only:
   1. if rank 

2. After the pipeline finishes, 

### Instance A

**IA10.** Instance A runs the component as production does: low-precision math, custom kernels, and the other
settings of the configuration. In version 1 it runs eagerly. 

**IA20.** One instance A does the capture and the replays. There is no separate capture instance. The capture
saves the CPU and CUDA random-number states before each call, and each replay restores them.

**IA30.** A runs the pipeline once, captures the calls of the listed components, and replays them with the
captured inputs afterwards (a standalone replay). If a component is listed, but never run on a rank, the rank get
NOT_REACHED_IN_A status for the component.
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

**IA40.** **Fidelity and determinism checks (A and B).** The instance replays a captured call once (replay 1). If
the output equals the captured output bit for bit, the fidelity check passes. Otherwise it replays the call
again (replay 2):
1. if the two replays differ, the status is NON_DETERMINISTIC_A or NON_DETERMINISTIC_B (depending on
   the instance), collectively called NON_DETERMINISTIC;
2. if the two replays are equal but differ from the capture, the status is REPLAY_MISMATCH_A or
   REPLAY_MISMATCH_B (depending on the instance), collectively called REPLAY_MISMATCH.

Rare nondeterminism can give a false REPLAY_MISMATCH. This is a false failure, never a hidden one.

**IA50.** By default, each test must be deterministic in A and in B. A test is one case: all calls or the
selected calls of one component path (CP50, CP120). Each test can have two determinism overrides:
1. The general override sets the policy for A. It also sets the policy for B if the test has no B override.
2. The B override sets the policy for B only. If it is set, it takes precedence over the general override.

If the policy of an instance requires determinism, NON_DETERMINISTIC stops the case on all ranks (FC90.4)
and fails the test at once. If the policy permits nondeterminism, NON_DETERMINISTIC does not stop the case, and
SUCCESS can replace it (IA80). The bitwise checks always run, whatever the policy is.

Each rank also has a nondeterminism status flag for each case. Each NON_DETERMINISTIC status that the rank gets sets
the flag, and nothing clears it; the rank also records the call identifier of each nondeterministic execution.
The flag of a case is the OR of the flags of all ranks of A and B. If it is set, the test issues a pytest
warning that names the instances, the ranks with the flag and their call identifiers, whatever the policy and
the outcome are.

**IA60.** REPLAY_MISMATCH always stops the case on all ranks (FC90.4) and fails the test at once.

**IA61.** It's an accepted limitation of UC10 that under a policy that permits non-determinism, fidelity
check failures (IA40) might be hidden while false REPLAY_MISMATCH might appear.
*Rationale:* Without invoking HL70 we can't make tolerance tests possible for such a component.
a. Suppose a component depends on hidden state, so replay 1 differs from the capture, and
it is also nondeterministic, so replay 2 differs from replay 1. Then the status is
NON_DETERMINISTIC, and IA80 counts it as passed. The fidelity failure is hidden.
b. The opposite error also happens. For a validly nondeterministic component, replay 1
and replay 2 can be equal and differ from the capture. IA60 then always fails the test.
So the override does not change every nondeterminism result "from a failure to a
warning", as UC10 says.

**IA70.** UNSUPPORTED marks a known limit of the harness:
1. If the harness cannot handle an argument (fp8 or MX tensors, types that it cannot serialize), only the
   affected case gets UNSUPPORTED. The message names the argument.
2. If the harness cannot handle a feature of a configuration, every case of the configuration gets
   UNSUPPORTED. The message names the feature and the
   run argument with its value. The harness finds this at collection, from the xDiT settings (TD20), or when A or
   B loads the configuration. After that, no instance of the configuration starts or continues. If no case gets
   SUCCESS_A, B does not start (FC20.5), so a feature that only B can find stays unknown in that run.

**IA80.** After passing the end-of-case barrier of FC50, each rank replaces its status with SUCCESS, if it passed all the checks (a
NON_DETERMINISTIC under a policy that permits it counts as passed), has done all the replays of the case, and
successfully dumped all the necessary data. SUCCESS is the only successful status. Otherwise the rank keeps its
last status; a rank whose case another rank stopped (FC90.4) keeps ABOUT_TO_START (FR50).

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

The classification is: the trusted eager path replaces compile, the attention kernel, fused kernels, FP8
communication and the hybrid precision schedules. B keeps the parallel degrees and layout, the parallel VAE,
VAE tiling and its tile shape, distilled weights, LoRA and channels-last.

**IB20.** B always loads in bf16 and casts to the reference dtype if the data dtype offers less
precision (less mantissa bits) than the reference dtype (so if the reference dtype is bf16
and there's fp32 data, that higher precision data stays unmodified). It does not load a
checkpoint in the reference dtype. B converts all components at once.

**IB30.** After B loads a model, a `TorchDispatchMode` audit checks all dtypes used to
run the model (including replays). It warns about dtypes with less precision than the reference dtype, and about
custom calls. The check does not fail, only prints a warning informing about an imprecise dtype use,
or a custom call. (it's a known limitation of version 1 needs to be addressed in future
versions that these warnings aren't recorded and are produced only during instance B runs)

**IB40.** B runs the pipeline in the reference dtype with the same run arguments as A (changes of IB10 applied).
It captures its own calls and runs its own fidelity check on them.
*Rationale:* if the replay is faithful with B's inputs, it is also faithful with A's inputs. That
holds if faithfulness depends on the context and state of the replay and not on the input values.
A component whose behavior depends on its inputs and on hidden state could break this assumption,
so a fidelity check on own inputs is needed.

**IB50.** B replays the inputs of A at least two times (using A's random-number states, but
restoring the original state after a replay) and compares each replay with the first one bit for
bit. Nondeterminism in B gives NON_DETERMINISTIC_B.

**IB60.** If a case gets REPLAY_MISMATCH_B, or NON_DETERMINISTIC_B under a policy that requires determinism, only
that case fails. The other cases continue.

**IB70.** B has three run-related parameters for the user: the reference dtype, the self-attention backend and the
cross-attention backend, and also own per-path determinism override setting, and deadlines.

**IB80.** An attention backend that declares a fallback, or that needs a blocked vendor package, makes the
configuration UNSUPPORTED_B (IA70.2). A cross-attention parameter for a model without a separate
cross-attention backend is an error. The default attention backend is `sdpa_efficient`.

### Failures, crashes and hangs

**FC10.** The harness must detect each crash and each hang, wherever it occurs, and attribute it to the
instance (A or B) and, for a failure during a replay, to the case, the component and the call identifier (FC20.3)

**FC20.** Each instance computes the results of many cases in one go and saves them to files. A crash or a
hang ends the instance. Version 1 does not recover.
1. A case with complete data (A and B) is checked as usual.
2. A case that an instance had to run but did not start gets FAILED_TO_START.
3. The case whose replay was running gets a status that shows what happened and in which instance, for example
   CRASHED_A, CRASHED_B, HUNG_A or HUNG_B. The message names the instance, the kind of failure
   (crash or hang) and, if available, the call identifier of the last progress record of each rank
   (not a proven cause, HL70).
4. A failure outside the replays gives FAILED_TO_START to every case that the instance did not start.
   The message names the instance and the kind of failure (crash or hang).
5. B runs only on cases with SUCCESS_A (IA80). If there are no such cases, or if processes of A survive
   SIGKILL (FC160), B does not start.

**FC30.** Each status that an instance gives has a form for A and a form for B, for example CRASHED_A and
CRASHED_B, collectively called CRASHED, or NOT_REACHED_IN_A and NOT_REACHED_IN_B, collectively
called NOT_REACHED_IN.

**FC40.** An exception that ends the instance counts as a crash (FC20).

**FC50.** A synchronization and a barrier at the end of each case, and progress records of each
rank, attribute crashes. Before each call of a listed component in a pipeline run, and before each
replayed call, each rank writes a progress record with the call identifier. A fault that silently
damages memory and shows only in a later case is an accepted limit.

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
A rank causing a decision and a rank following the decision must write the details to their log files.

**FC100.** If the call addresses of a case differ between the ranks, the case gets FAILED_TO_START on all ranks.
B makes no replays of a case that it skips before the replays.

**FC110.** If a rank fails, torchrun stops the other ranks, and the failure is a crash, not a hang.

**FC120.** If a rank of A or B does not do its work because a configuration or a case is UNSUPPORTED, the cause
and its details reach pytest as the skip reason (UNSUPPORTED_A) or the error message (UNSUPPORTED_B) of each
affected case. If a single rank gets UNSUPPORTED status all other ranks must also get UNSUPPORTED (shared
decision, FC90).
The state keeps the reason for later re-runs. The details are:
1. the instance and the ranks;
2. for a configuration: the feature, and the run argument with its value;
3. for a case: the call identifier, the argument name, and its type or dtype.

**FC130.** Every run shows the UNSUPPORTED reasons without extra pytest options. A `pytest_terminal_summary`
hook adds a section that lists each UNSUPPORTED case with its reason.

**FC140.** If the ranks report different statuses for one case, PI30 decides the outcome from the statuses of
all ranks. The message shows each status with the ranks that have it.

**FC150.** Each worker (a worker is one rank of an instance) has a structured log file that only the worker
can write to (the launcher can read from the file, TD110) during instance execution. On a worker startup,
it gets from the launcher a path to data directory to store the dumps and the status file.
The status file name is `rank_<rank_number>.jsonl`. It's an error if the file existed when
the worker started. Each worker appends (and flushes immediately) their own file
with a progress record (TD110, TD120), statuses change and all other information that
requirements of this document mandate to keep. Each record must have
at least one common field: a timestamp. Records related to a case must have another common field
the case identifier, so one can use it to find entries related to the case.
The file hence serves as a structured timestamped log of all important events the worker observed.
A worker never deletes anything from the file, only appends to it.
After an instance ends, the ownership of the log files of all
workers is passed to the launcher: it reads the records of all ranks (grepping for files with a
known pattern and inferring participating ranks from it) and appends final statuses
for cases that must run, but didn't finish according to the file records. This enables crashes/hangs detection as per
TD110, and sets FAILED_TO_START for cases that didn't run (FC20.2).

** * vvv TODO vvv * **

 After the launcher
reconstructs the final states of all cases on all ranks from logs, it aggregates them by case and call,
and writes final case state report to a corresponding `instance.json` file, where for
each case, it writes all information a case results reader in pytest would need
for further work (applicable rank statuses, determinism flags, call identifiers and
everything else the reader would need). While aggregating, it must not assume that for
a certain case all ranks necessary always ran the same calls; it must identify the
discrepancies between ranks whenever possible, issue a diagnostics message and fail the execution.

** * ^^^ TODO ^^^ * **

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

**PI20.** **General meaning of the pytest outcomes.** (This requirement exists to alleviate decisions
for new statuses. Actual mapping is provided by PI30.) "error" means that the test has nothing to check, for example
because an instance that had to replay the case did not start its replays. "failed" means that an instance
replayed the case, and the instance or the test found a problem. Documented exceptions are possible:
1. UNSUPPORTED_A gives "skipped".
2. A configuration that is UNSUPPORTED_B gives "error" to every case, whatever its statuses from A are.

*Rationale:* UNSUPPORTED_A is a known limit of the harness, not a problem of the component. B is the reference
and must handle everything A handles, so UNSUPPORTED_B is not a reason to skip. If B cannot handle the configuration, no
case of the configuration can have a meaningful test.

**PI30.** pytest outcomes (PI20; for re-runs FR50 takes precedence):
1. A failure of the configuration fixture (TD70, FC70, FR60), or a configuration that is UNSUPPORTED_B (IA70.2),
   gives "error" to every case of the configuration, whatever its other statuses are. For UNSUPPORTED_B, the
   message of each case also lists its statuses from A.
2. Otherwise, each case gets its own outcome decided in the following order:
   1. "error" if any rank has FAILED_TO_START, UNSUPPORTED_B, or NOT_REACHED_IN.
      Also if a rank of A has UNSUPPORTED_A, but some other rank of A doesn't have it.
      Also if all ranks of an instance that had to run the case still have ABOUT_TO_START
      ("the previous run did not finish"): A always has to run a case, B only if all ranks
      of A have SUCCESS (FC20.5). The case fixture (function scope) raises
      the exception before the test function starts, because pytest gives "error" only for an exception in
      setup, teardown or a fixture.
   2. "skipped" for UNSUPPORTED_A on all ranks of A.
   3. "failed" if not all ranks of A and B have SUCCESS (PI50), or for a failed check.
   4. "passed" otherwise
3. `record_property` stores the statuses, the nondeterminism flags and applicable information mandated by other
   requirements such as call identifiers of all ranks.
4. If a pytest outcome mandates an exception (i.e. it's a "failure" or "error"), the exception message must contain
   the statuses and the nondeterminism flags of all ranks and all other applicable information mandated by other
   requirements.

**PI40.** The test checks each rank independently against the same tolerance. It compares the output that A
captured with the output of replay 1 of B with the inputs of A.

**PI50.** If the statuses give "failed" (PI30.2.3), the test fails at once. The failure message lists every
rank with its status and, if known, the call identifier at which the rank got it; the case writes one metrics line
with its outcome only.
Otherwise, the test runs as many checks as possible before it fails, like the `EXPECT_` macros of
GoogleTest. It uses the installed `pytest-check` package. After the checks, it writes the metrics
lines with the values and the outcome.

### Files, re-runs and metrics

**FR10.** Each configuration writes its data into its own directory. Only the listed components are saved. A
test that passes marks its state as passed and deletes its data files; the state stays, so that re-runs can
skip the case (FR50). The files of tests that did not pass stay.

**FR20.** One entity, in the pytest process, defines the paths of the data files. The launcher,
instances and test functions all get the paths from it. Test functions read the files, and
might delete some of them during the work, following other requirements to save the storage space.

**FR30.** Tensors go into safetensors files. Structure and plain values go into JSON. Other types give
UNSUPPORTED for the affected case.

**FR40.** The data goes into `_data/` and the metrics go into `_metrics/`. Both are in the directory of the
running test module. The pytest process resolves the paths one time. The launchers and the workers get
absolute paths.

**FR50.** **Re-runs.** Each case has its own state: the status, the nondeterminism flag and supplemental
information of each rank of A and B (for example, see IA50, IA80, or PI50; other requirements might
add to the list), and the mark of a passed test (FR10). A fresh run gives every rank the status
ABOUT_TO_START in each case.
1. If every recorded case passed (a case that is UNSUPPORTED_A counts as passed), the harness cleans the
   directory automatically and runs everything from the beginning.
   *Rationale:* a new run of tests that all passed gives no new information.
2. Otherwise, the harness prints a warning and does not run A and B. Each case gets its outcome from its stored
   state by the same rules as in a fresh run (IA50, PI30), and the checks use the stored data. Two rules come
   first in this order: a case that passed in a previous run is skipped with "passed in a previous run",
   and a case without a state or data expected by the test function from the rank statuses
   is an error ("not part of the run that produced the data" without a state, "the stored data is
   missing" without data).
3. The user cleans the directory after a change of the settings or configurations.
   There is no fingerprint check.

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
