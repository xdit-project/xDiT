# Requirements and use cases of the component test harness

This document lists the use cases and the requirements of the xDiT component test harness.

Each item has an identifier: `UC` for use cases, `R` for requirements.

## Use cases

**UC1. Check that a component works correctly.** A developer asks: "Does this component of this model
configuration work correctly?" The harness runs the component as production runs it (instance A) and as an eager
reference in high precision (instance B). It then compares the outputs against tolerances. Requirements: R1–R6,
R23–R28, R36–R40, R60–R75.
*Rationale:* some operations of xDiT diffusion models can be flaky (produce wrong numerics, or cause hangs or crashes). Tests of single components are decoupled from the whole model.

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang. The
report names the instance, the component, the call and the rank. Requirements: R6, R63–R65, R72, R76–R94.

**UC3. Test the configurations of the CI benchmark files.** A test selects an entry of a CI benchmark YAML file.
The test can change any field of the entry. Requirements: R25–R28.
*Rationale:* the user specifies model configurations in the same way as the benchmark configurations.

**UC4. Cover many configurations of one model in one test module.** Requirements: R29–R33.
*Rationale:* there are hundreds of configurations in total, and most of them are very similar.

**UC5. Test sub-components and selected calls.** Examples are a transformer block, an attention module, the VAE
decode method, and one specific call of a transformer. Requirements: R41–R59.

**UC6. Tune tolerances over time.** A test reports its metrics in each run. A missing tolerance makes the test fail,
so the user sets tolerances from the reported values. The metrics history shows how the deviations change while
the implementation changes. Requirements: R37–R39, R106, R107.

**UC7. Run the tests again after failures.** A new run compares the stored data again, for example with new
tolerances, and does not run A and B again. After all cases pass, the next run starts from the beginning
automatically. Requirements: R104, R105.

**UC8. Select single cases.** The user selects cases with pytest, for example with `-k` or a node ID.
Requirements: R45, R58.

**UC9. Keep the tests valid while xDiT changes.** Requirements: R2, R3, R10.
*Rationale:* xDiT changes fast, and its components get new augmentations all the time. Hard-coded tests become
obsolete.

**UC10. Accept nondeterminism that is valid.** An override of one test changes the result of nondeterminism in that
test from a failure to a warning. The determinism checks of all other tests stay active. B should have its own
override. Requirements: R64, R65.
*Rationale:* some operations are validly nondeterministic. A model can have one nondeterministic component among
many, and that component must be testable. Instance B generally use different kernels that might have other determinism expectations.

**UC11. Later use cases.** Compiled mode, a reference at world size 1 (B1), several input sets, MI300X and other
hardware, and benchmark and determinism scripts that use the foundation of the harness. Requirements: R11, R13,
R17, R18, R108–R114.

### Requirements

**R1.** The harness tests individual components of xDiT diffusion models, for example a VAE, a transformer, a
transformer block, an attention module, a text encoder, or a method such as the VAE decode. It uses exactly the
configuration and the weights that the xDiT runner uses for a model configuration.

**R2.** The harness takes each component from the model that the xDiT runner builds. It does not extract,
hard-code or build a component itself.
*Rationale:* the setup of a component depends on the model configuration. For example, the VAE of Wan2.2 can be
the VAE of `diffusers`, a DistVAE, or something else. Also, xDiT changes fast (UC9).

**R3.** The details of a configuration are private to the configuration: Ulysses, ring, classifier-free guidance,
quantized GEMMs and others. The tests do not depend on them. The tests are generic, so that they can run every
configuration. The configuration identifier is one parameter among many.

**R4.** The reference is the same xDiT component, built by the xDiT runner, run with extended precision (fp32 by default). It is not a `diffusers` implementation.
*Rationale:* a comparison with `diffusers` forbids all customizations of xDiT. xDiT has reasons for its own
implementations.

**R5.** One test answers one question: "Does this component, in this regime, with these inputs, work correctly?"
A failed check belongs to one test, so only that test fails.

**R6.** A test fails for a deviation beyond its tolerance, a crash, a hang, nondeterminism, or any other error.
"Flaky" includes a crash and a hang.

**R7.** **Building principle of version 1.** Version 1 is a simple solution. It must run end to end in the best
case. It does not handle special cases. This principle applies to all other decisions. Rejected alternatives that
are worth an implementation go to future work.
*Rationale:* the idea of the harness rests on many risky assumptions. They must be validated first, before work
on special cases starts.

**R8.** **Working tests first.** Test coverage has priority over correct reports, for example a correct call
address. A wrong report is acceptable as a known limitation with a future-work item. A deviation that a working
test needs is permitted with the approval of the user.

**R9.** Decisions rest on facts that the code shows directly.

**R10.** Where possible, xDiT gets a refactor that makes components testable, instead of a hack around hidden
coupling. Examples: compile code only compiles, and the warmup is not part of the compile function. In version 1,
xDiT changes only where the first configuration needs a change.
*Rationale:* separation of concerns makes the code testable.

**R11.** Tests and benchmarks are separate from the entry point on. Benchmark scripts and determinism-check scripts
come later. They use the same foundation that the tests use to load and build a model and to get its components.
Version 1 plans nothing for benchmarks, but it identifies the code that benchmarks will use.

**R12.** One model initialization of an instance serves the tests of many components.
*Rationale:* a model can need 5–10 minutes to load and initialize. That is too long for each test.

### 3. Scope of version 1

**R13.** Version 1 tests eager mode only. Compiled mode comes later, after the xDiT refactors. Eager mode and
compiled mode are separate regimes with separate expectations: an eager result counts only for eager mode. The
harness ignores the `use_torch_compile` value of the configuration and forces compile off (R34).
*Rationale:* compiled code can fuse two low-precision operations with one wider accumulator. Eager code moves the
data through memory in the low-precision type, so it does not get this precision.

**R14.** Multi-GPU configurations are required from version 1 on. Runs use one node only, with torchrun or an
equivalent.

**R15.** The first configuration is `wan2_2.quantgemm_quantattn.gfx950`. It must run end to end.

**R16.** The harness runs other models as they are. It reports their failures. Version 1 does not fix them.

**R17.** Version 1 is built on this MI355X machine. Later, the tests must work on at least MI300X. Other hardware
(for example, entries for other GPU types) is future work.

**R18.** Version 1 uses one input set: the run arguments of the configuration. Several input sets are future work.

**R19.** The configuration defines all dtypes. The tests do not change dtypes.

**R20.** Version 1 does not support configurations with per-step schedules (hybrid attention schedules, hybrid GEMM
schedules), VSA, PipeFusion, step caching or FSDP. A guard gives UNSUPPORTED to such a configuration. The
refactors that remove the hidden state of per-step schedules and PipeFusion are future work.

**R21.** All other work on step caching is far-future work.

**R22.** Version 1 has no pytest markers and no CI integration; they come later. Version 1 does not reduce the
number of transformer blocks. Section 12 lists the future work.

### 4. Test definition

**R23.** The definition is in Python. One call defines one configuration: its run settings, its list of
components, and the tolerances for each output of each component. Each configuration is a separate call. The
definition can move to a YAML file later.

**R24.** The interface of a test module is:
1. `harness.Configuration(yaml_path, entry, overrides, reference, components)` for one configuration;
2. `harness.Reference(dtype, self_attention_backend, cross_attention_backend)` for the parameters of B;
3. `harness.Tolerance(max_abs, rel_l2)` for the tolerance of one output;
4. `components`: a dictionary from a component path to a dictionary from an output name to a `Tolerance`;
5. `test_component = harness.make_test(...)` with the configurations of the module.

**R25.** The run settings come from an entry of a CI benchmark YAML file, in the same format as the benchmark
configurations. A helper reads the entry, builds the command-line flags as CI does, and parses them with the
parser of the xDiT runner.
*Rationale:* the parser applies its defaults and conversions, for example to `gemm_quantization: low=fp4,high=fp6`.

**R26.** A test can override every field of the YAML entry. `overrides` is a dictionary with the structure of a
YAML entry: the key `model` refers to the model, and the key `args` holds the overrides of the arguments.
*Rationale:* tests need specific overrides, for example `num_inference_steps`.

**R27.** The overrides apply to A and to B. B has a few parameters of its own (R74).

**R28.** A test can reduce `num_inference_steps`, for example to 5 for Wan2.2. The value depends on the model.
*Rationale:* the transformer blocks are the same in each step, so fewer steps test the same components in less
time.

**R29.** One test module covers many or all configurations of one xDiT model.

**R30.** Each configuration has its own fixture, which runs A and B of that configuration. The tests of the
configuration run before the next configuration starts its instances.
*Rationale:* the instances of one configuration take long and make large files. Runs that alternate with the
tests need less storage than one fixture that runs all instances first.

**R31.** The configuration name is the `name` of the entry by default. It is an error if neither a name nor an
entry is set, or if the name is absent, `None`, not a string, or empty. The name selects the data directory and
the metrics file of the configuration.

**R32.** Configuration names are unique. The harness raises an error for two configurations with the same name.
The check runs during collection, and it also covers the configurations that `-k` deselects. A separate check in
the configuration fixture stops a configuration that starts a second time in one session. Unique names across
sessions are the responsibility of the user.

**R33.** A name contains only letters, digits, `_`, `.` and `-`. The literal names `.` and `..` are not allowed. The
`--ff` option of pytest is not supported.
*Rationale for `.` and `..`:* each configuration needs its own directory. `_data/.` is the shared `_data/`
directory, and `_data/..` is the directory of the test module.

**R34.** The harness forces these settings in A and B: no warmup calls, one iteration, and `use_torch_compile`
off.

**R35.** Deadlines have default values, and a test can override them. The defaults are:
1. A: 10 minutes for the initialization, 20 minutes for the pipeline run;
2. B: 15 minutes for the initialization, 40 minutes for the pipeline run;
3. each replayed call: 2 minutes in A and 8 minutes in B, measured as the time since the last progress record.

*Rationale:* the number of calls of a case is known only after the pipeline run of A, so a limit per case does not
fit. The limits per call are below the 10-minute default timeout of the process group, so a rank that stops in a
collective gets HUNG, not CRASHED.

### 5. Tolerances and metrics

**R36.** A tolerance applies to one output of one component path. Every tensor output must have a tolerance. A
tensor output without a tolerance is a failure, and so is a tolerance for a name that is not an output. The
failure message lists the actual output names.
*Rationale:* there must be a guarantee that the test itself has no mistake and that it tests each output.

**R37.** A metric without a value gets a very small default tolerance, `1e-10`. The test then fails, and the user
sets a reasonable tolerance from the reported values.

**R38.** The test asserts the maximum absolute deviation and the relative L2 deviation. It reports the cosine
similarity without an assertion.

**R39.** The test always reports its metric values, also when it passes.

**R40.** Output names use the path notation relative to the return value: `[0]` for a tuple item, `.name` for an
attribute, and an empty name for a bare tensor. Nested containers flatten to their tensor leaves. Outputs that are
not tensors must be exactly equal in A and B.

### 6. Components, paths and calls

**R41.** The harness tests only an explicit list of components. The list is a list of strings. Each string is the
path from the runner model object to the component. Only the listed components are captured.
*Rationale:* there are too many components. Each transformer has 40 blocks, and each block has its attention
modules. There is no space to store the inputs and outputs of all of them.

**R42.** The list is known before the run. The harness makes the test cases from the definition, without a model.

**R43.** Path syntax: dots separate the elements, and `[i]` indexes a container, for example
`pipe.transformer.blocks[0].attn1` and `pipe.transformer.blocks[-1]`. A path has no leading dot. `[key]` selects
an item of an `nn.ModuleDict`.
*Rationale:* the dots show the boundaries between the ancestors, and the brackets show indices. This is easy to
parse.

**R44.** The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
`refiner_pipe`. The harness does not check that the attribute holds a pipeline. An attribute that holds `None` is
a path error.

**R45.** A call number follows the element after a colon, for example `pipe.transformer:3.blocks[0]:3`. A path
without a call number selects all calls, and its one case checks every call. The case fails if one call fails,
and the message names the failing calls.
*Rationale for the colon:* `-k` and the shell accept it. Parentheses and braces fail in `-k`. The user accepts
that `-k` matches substrings, so `-k "transformer:3"` also selects `transformer:30`.
*Rationale for one case:* it does not follow "one test, one input", but it serves the purpose and keeps the code
simple.

**R46.** Call numbers are not negative.
*Rationale:* then the harness can save the tensors of a call at once.

**R47.** A path only tells how to reach the component. It does not describe the call stack, and the harness never
uses it to recover the call stack. A path without call numbers addresses exactly one component.

**R48.** Each callable element of each listed path has one global call counter. The harness knows at path
resolution which elements are callable.
1. The counter is 0 before the pipeline starts. It does not depend on other counters.
2. The call handler first increments the counter, and then makes all other checks, for example whether to save
   the call.
3. So the first call has the number 1, and no call has the number 0. A counter value of 0 means that the element
   was never called.
4. Every call of the element counts, whatever code makes it.

**R49.** `:i` refers to the global counter of its element. Example: `a.b:2.c.d[2]:3` selects a call of `d[2]`
when its counter is 3 and the counter of `b` is 2. During the pipeline run of A, the harness saves the inputs of
each selected call, and the outputs after the call returns.

**R50.** An address contains the call numbers of all callable elements that the harness tracks. `pipe` is
callable, but version 1 does not track it.

**R51.** The harness does not care who calls a component. It uses only the inputs and the outputs of each call.
Calls from another pipeline, from the runner, or from any other code are tested like all other calls.

**R52.** Two entries of the same component with different call numbers are separate cases, for example
`pipe.transformer:1` and `pipe.transformer:2`. Overlapping entries are an error at path resolution. Two entries on
the same component are disjoint only if one object has a call number in both entries and the numbers differ.
Then `pipe.transformer:1` and `second_pipe.transformer:2` in the LTX-2.3 runner are permitted.
*Rationale for the error:* it is better not to support such cases than to make the implementation more complex.

**R53.** `pipe.vae.decode` must be testable. A path through a module that the run never calls must be valid.
*Rationale:* the VAE decode is an important component to test.

**R54.** Version 1 has no special handling and no rules for other pipelines or for shared components. The harness
does not detect, warn about, or report calls from outside the first element. A misleading address is a known
limitation and the responsibility of the user. The limitation covers three causes:
1. another pipeline calls an object that the first element also holds;
2. the runner calls the first element more than once in one run;
3. the runner calls a component directly.

*Rationale:* the test results stay valid. Test coverage is more important than a correct address (R8).

**R55.** The comparison of A and B assumes that both take the same execution path. A selection of more calls than
the user wants, because of a limitation, is acceptable if the component gets tested.

**R56.** A processor cannot be listed. The list names the attention module that calls it, for example
`pipe.transformer.blocks[0].attn1`.

**R57.** A path must resolve in A and in B. If a path resolves only in A, its case gets FAILED_TO_START with the
path error. Path errors that need a built model show only after the initialization of A; this cost is accepted.

**R58.** The harness can change the test case names before pytest sees them, if the `-k` grammar needs it.

**R59.** Version 1 needs no replay of an ancestor before a replay. For several input sets, the user can give an
optional component that the harness replays first to set the stage (future work).

### 7. Instance A

**R60.** Instance A runs the component as production does: low-precision math, custom kernels, and the other
settings of the configuration. In version 1 it runs eagerly.

**R61.** One instance A does the capture and the replays. There is no separate capture instance.
*Rationale:* this saves one model initialization.

**R62.** A runs the pipeline once, captures the calls of the listed components, and replays them with the captured
inputs (a standalone replay).

**R63.** **Fidelity and determinism checks (A and B).** The instance replays a captured call once. If the output
equals the captured output bit for bit, the fidelity check passes. Otherwise it replays the call again:
1. if the two replays differ, the status is NON_DETERMINISTIC;
2. if the two replays are equal but differ from the capture, the status is REPLAY_MISMATCH.

Rare nondeterminism can give a false REPLAY_MISMATCH. This is a false failure, never a hidden one.

**R64.** By default, each test must be deterministic in A and in B. A test is one case: all calls or the selected
calls of one component path (R45, R52). Each test can have two determinism overrides:
1. The general override sets the policy for A. It also sets the policy for B if the test has no B override.
2. The B override sets the policy for B only. If it is set, it takes precedence over the general override.

The policy that applies to an instance decides the result of nondeterminism in that instance:
1. If the policy requires determinism, NON_DETERMINISTIC fails the test at once.
2. If the policy permits nondeterminism, the harness prints a warning and continues to the tolerance checks.
3. The bitwise checks always run, whatever the policy is.

*Rationale:* a model can have one nondeterministic component among many. That component must be testable, and
the determinism checks must stay active for all other tests. B has its own override because B generally runs
other kernels than A.

**R65.** REPLAY_MISMATCH always fails the test at once.
*Rationale for R64 and R65:* nondeterminism and a replay mismatch show that something is seriously wrong. Without
the override, a check of the data content tells nothing: the next run of a component with an unknown source of
nondeterminism can give very large deviations.

**R66.** The test compares the output that A captured during its pipeline run.

**R67.** If the harness cannot handle an argument (fp8 or MX tensors, types that it cannot serialize), only the
affected case gets UNSUPPORTED. The message names the argument.

### 8. Instance B

**R68.** Instance B is an eager reference. It uses the same configuration and the same world size and parallel
layout as A, with these changes:
1. the reference dtype: fp32 by default; a parameter, because some models need a smaller dtype;
2. quantization off, so B loads normal weights;
3. dense attention;
4. vendor packages blocked;
5. no compile, no step cache, no FP8 communication, no hybrid schedules.

The classification is: the trusted eager path replaces compile, the attention kernel, fused kernels, FP8
communication and the hybrid precision schedules. B keeps the parallel degrees and layout, the parallel VAE, VAE
tiling and its tile shape, distilled weights, LoRA and channels-last.

**R69.** A `TorchDispatchMode` audit checks the dtypes in B. It warns about custom calls and does not fail.

**R70.** B loads in bf16 and casts up to the reference dtype. It does not load an fp32 checkpoint. B converts all
components at once.
*Rationale:* casting up from bf16 is enough. About 140 GB of fp32 weights per rank fits on MI300X.

**R71.** B runs the pipeline in the reference dtype with the same run arguments as A. It captures its own calls
and runs its own fidelity check on them.
*Rationale:* a run in bf16 proves only bf16. fp32 uses other kernels and needs its own proof.

**R72.** B replays the inputs of A at least two times and compares the replays bit for bit. Nondeterminism in B
gives NON_DETERMINISTIC_B.
*Rationale:* B runs its own kernels, which can also have problems.

**R73.** If the fidelity check of B fails for a case, only that case fails. The other cases continue.

**R74.** B has three parameters for the user: the reference dtype, the self-attention backend and the
cross-attention backend.
*Rationale:* there are several fp32 backends, and the reference dtype can be other than fp32. The user selects
the backends together with the dtype.

**R75.** B refuses a backend that declares a fallback. A backend that needs a blocked vendor package fails at the
start of B. A cross-attention parameter for a model without a separate cross-attention backend is an error. The
default cross-attention backend is `SDPA_MATH`. A chunked exact attention backend is future work.

### 9. Failures, crashes and hangs

**R76.** The harness must detect each crash and each hang, wherever it occurs, and attribute it to the code that
ran last: the instance (A or B) and the case and component that it ran.

**R77.** Each instance computes the results of many cases in one process and saves them to files. A crash or a
hang ends the instance. Version 1 does not recover.
1. A case with complete data (A and B) is checked as usual.
2. A case without data gets FAILED_TO_START.
3. The case whose replay was running gets a status that shows what happened and in which instance, for example
   CRASHED_A, CRASHED_B, HUNG_A or HUNG_B.
4. A failure outside the replays gives FAILED_TO_START to every case without data.
5. After a failure of A, B does not start.

*Rationale:* a model initialization is expensive, so one process must serve many cases (R12).

**R78.** A failure of B gets its own status, separate from a failure of A.

**R79.** An exception ends the instance with ERROR_A or ERROR_B and the traceback.

**R80.** A synchronization and a barrier at the end of each case, and progress records of each rank, attribute
crashes. A fault that silently damages memory and shows only in a later case is an accepted limit.

**R81.** Statuses can be extended later. UNSUPPORTED marks unsupported configurations. There is no
NOT_EVALUATED status. Users do not need to see status identifiers: the harness raises verbose exceptions with
explanations that a person can read.

**R82.** One launcher runs each instance, between the fixture and torchrun. It exits with 0 when it worked; the
detailed statuses go through another channel. A nonzero exit means that the launcher failed and that no data
exists, so every case of the configuration fails. The launcher enforces the deadlines.

**R83.** **Lock step.** When an instance has more than one rank, all ranks run every execution that can contain a
collective in the same order: the initialization, the pipeline run, each replay, and the collectives of the
harness. Work without collectives does not need lock step. Each per-rank decision that changes this sequence is
shared, or it ends the instance on all ranks through an exception.
*Rationale:* almost every execution of a pipeline or a component can contain a collective, and the harness does
not know this in advance.

**R84.** These decisions are shared across all ranks of the instance:
1. the support check of the configuration, in A and B;
2. the skip of a case before the replays: a path error, UNSUPPORTED from the capture of A, NOT_REACHED_B, inputs of
   A that B cannot use, and call lists that differ between ranks;
3. the second replay of a call;
4. the stop of a case after a call;
5. the number of replays of the inputs of A in B.

**R85.** If the call addresses of a case differ between the ranks, the case gets FAILED_TO_START on all ranks.
"Participating ranks" means all ranks of the instance. "The case fails at once" means that all ranks stop the
remaining replays of the case and go to the end of the case. B makes no replays of a case that it skips before the
replays.

**R86.** If a rank fails, torchrun stops the other ranks, and the status is the failure of the failed rank, not
HUNG.

**R87.** If a rank of A or B does not do its work because a configuration or a case is UNSUPPORTED, the cause and
its details reach pytest as the skip reason of each affected case. For an unsupported configuration, that is
every selected case. The state keeps the reason for later reuse runs. The details are:
1. the instance and the ranks;
2. for a configuration: the feature, and the run argument with its value;
3. for a case: the call address, the argument name, and its type or dtype.

**R88.** Every run shows the UNSUPPORTED reasons without extra pytest options. A `pytest_terminal_summary` hook
adds a section that lists each UNSUPPORTED case with its reason.

**R89.** When several ranks report a skip, the reason lists each distinct reason with the ranks that found it.
The most severe status wins, in this order: FAILED_TO_START, NOT_REACHED_B, UNSUPPORTED.

**R90.** Each result that is not a pass names the rank that caused it. If a correct attribution is not possible,
the result shows the status of every rank, labeled with the rank number. In detail:
1. support and skip decisions: each rank that found the condition is a cause;
2. call lists that differ: the report lists the call addresses of every rank;
3. NON_DETERMINISTIC and REPLAY_MISMATCH of a call: the report gives the status of every rank, also of the
   passing ranks, and says that the source rank is unknown;
4. an exception or a fatal signal: one rank is named only if exactly that rank failed by itself; otherwise the
   report lists the failure of every rank with its time, kind and traceback;
5. a hang: one rank is named only if exactly that rank's last progress record differs from the others; otherwise
   the report lists the last progress record of every rank;
6. ranks with the same content share one line.

**R91.** One result path carries all results of the workers. Each worker appends result records to its own
progress file. After torchrun ends, the launcher reads the records of all ranks, merges them by case, call and
rank, and writes each state file. While an instance runs, the launcher is the only writer of state files.

**R92.** Worker records plus a small supervisor for each rank attribute crashes. An exception gives an error
record. A fatal signal that Python can catch gives a fault record through `faulthandler`. The rank supervisor
starts the worker as its child, records the time and the kind of the end of the worker, also after SIGKILL, and
records when torchrun sent SIGTERM. So the launcher can tell the ranks that torchrun stopped from the rank that
failed by itself.

**R93.** The status of a rank does not show which rank causes it, because collectives move data between ranks.
The specification states this limitation, and future work addresses it.

**R94.** A process that survives SIGKILL (uninterruptible sleep) must not hang the pytest session. After SIGKILL,
the supervisor waits only for a limited time, and then it reports the surviving processes. The harness then
reports the results of this configuration and stops the session: B does not start, the cases keep the outcomes
that the launcher wrote, cases without data get FAILED_TO_START with the list of surviving processes, and the
fixture stops the session before the next configuration. The stop message names the configuration, the instance
and the surviving processes, and says that the GPUs may need a reset.

**R95.** The harness itself must have a test set. The tests make sure that the instances and the launcher handle
each behavior of a component correctly, as the specification requires: a failed determinism check, a failed
fidelity check, a hang and a crash. The specification must list these tests.
*Rationale:* the instances have complex rules for correct behavior across ranks.

### 10. pytest integration

**R96.** The harness is a set of pytest tests. The `pytest_integration` prototype is an implementation hint, not a
binding rule. The harness takes the best of it and adapts it.

**R97.** pytest outcomes: tolerance failures, NON_DETERMINISTIC, REPLAY_MISMATCH, CRASHED, HUNG and ERROR give
"failed"; FAILED_TO_START gives "error"; UNSUPPORTED gives "skipped". `record_property` stores the status.

**R98.** The test checks each rank independently against the same tolerance.
*Rationale:* a failure then shows the failed rank. It also permits tolerances for each rank later.

**R99.** The test runs as many checks as possible before it fails, like the `EXPECT_` macros of GoogleTest. It
uses the installed `pytest-check` package. NON_DETERMINISTIC without the override and REPLAY_MISMATCH fail the
test at once. The failure message lists every call and rank with these statuses, and the case writes one metrics
line with its outcome only.
*Rationale:* a failure at the first mismatch tells nothing about the later checks.

### 11. Files, re-runs and metrics

**R100.** Each configuration writes its data into its own directory. Only the listed components are saved. A test
that passes deletes its files; only the files of failed tests stay. A failed test keeps its inputs and both
outputs. A crashed, hung or not-started case keeps the files that exist.

**R101.** One entity, in the launcher, defines the paths of the data files. The test functions get the paths from
it, read the files, and delete them.

**R102.** Tensors go into safetensors files. Structure and plain values go into JSON. Other types give
UNSUPPORTED for the affected case.

**R103.** The data goes into `_data/` and the metrics go into `_metrics/`. Both are in the directory of the running
test module.

**R104.** **Re-runs.** Each case has its own state. A fresh run gives each case the state ABOUT_TO_START, and the
state changes during the run.
1. If every recorded case passed (UNSUPPORTED counts as passed), the harness cleans the directory automatically and
   runs everything from the beginning.
   *Rationale:* a new run of tests that all passed gives no new information.
2. Otherwise, the harness prints a warning, does not run A and B, and runs the tests only on the data that still
   exists.
3. Each case acts by its stored state: a case that passed earlier is skipped with "passed in a previous run"; a
   failed case with data is compared again; a case without data shows its stored status again; a case that is
   still ABOUT_TO_START is an error ("the previous run did not finish"); a case without a state is an error ("not
   part of the run that produced the data").
4. The user cleans the directory after a change of the settings. There is no fingerprint check.
   *Rationale:* a fingerprint does not protect against changes of the xDiT code, and it is too much burden.

**R105.** A marker file and a lock protect the data directory. Two pytest sessions must never use one
configuration directory at the same time, also during the cleanup.

**R106.** **Metrics file.** It lives in its own directory, apart from the dumps. It only grows, and cleaning never
deletes it. There is one file for each configuration name, in JSON Lines format, with UTC timestamps in ISO 8601.
Each append takes an `fcntl.flock` lock on the file.
*Rationale:* a metrics file that survives many runs shows how the deviations and the tolerances change while the
implementation changes.

**R107.** Each metrics line has these fields:
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
architecture matters because values differ between MI355X and MI300X. One line for each rank would make the file
too large to read.

### 12. Future work

**R108.** **Compiled mode.**
1. The model loads in two phases: first the initialization without compile, then the compile.
2. The compile code becomes a declarative plan that a simple generic compile step applies. The warmup moves out of
   the compile function.
3. The harness, not the runner, selects what to compile. Sub-components such as transformer blocks and attention
   modules are tested individually in eager mode and compiled, also if the compiled sub-component gets slightly
   different kernels than in the model.
4. Hooks run outside compiled code.
5. Only the tested component must be compiled; its parents can stay eager.

**R109.** **Step schedules.** A step plan reaches the components through a scoped holder. A change of `diffusers`
signatures is not wanted.

**R110.** The future-work part explains in detail what each refactor must deliver.

**R111.** These items are future work:
1. B1 after compiled-mode support, for the models that fit in memory;
2. several input sets, with the challenges that are already known;
3. a full reference run;
4. glue code as components that the harness can capture and replay;
5. shapes that no configuration produces;
6. a run of B for the cases that A finished, and continuation after a failure of B;
7. FSDP: B loads the FSDP components directly in the reference dtype, through an override of the strategy dtype;
8. dequantization of fp8 and MX inputs;
9. a comparison of arguments that a component changes in place;
10. refactors that remove hidden state from components;
11. a tolerance-based fidelity check for nondeterministic components;
12. a full cleanup on request;
13. step caching (far future);
14. special handling of components that several pipelines share.

**R112.** A brief future-work note: configurations are often alike, and many could probably share at least one
instance B, which could speed up the runs significantly. An automatic grouping algorithm that does not depend on
the model could be tricky.

**R113.** Future work lists the possible ways to identify the calls of each pipeline. The class swap is the
probable choice. The future-work item describes the concerns about it and what its implementation takes.

**R114.** A future-work item addresses the source rank of a replay failure (R93).

---

## Approved through "do it"

**D1.** The harness package goes into `xDiT/tests/component/harness/`. Test modules go directly into
`xDiT/tests/component/`. So `_data/` and `_metrics/` are in `xDiT/tests/component/`.

**D2.** The YAML path is a parameter of the configuration call. The first test module gives
`/app/.ci/benchmark_configs/wan2_2.yaml` as a literal.

**D3.** B forces full fp32 matrix precision: `torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`.
*Rationale:* without these settings, fp32 matrix operations can run at a lower precision, and the dtype audit
cannot see this.

**D4.** The pytest process resolves the paths of `_data/` and `_metrics/` one time. The launchers and the workers
get absolute paths.

**D5.** A test module calls `make_test` one time, with all its configurations.
1. `make_test` parametrizes the test function with (configuration, case) pairs, grouped by configuration.
2. The configuration goes to the fixture as an indirect parameter with module scope.
3. Configuration objects compare by identity, so `==` of pytest never compares the nested settings of two
   configurations.
4. The fixture takes the selected cases of its own configuration from `request.session.items`.

*Rationale:* pytest starts a new fixture instance only when the parameter changes from one test to the next.

**D6.** A case ID has the form `test_component[<name>:<path>]`. The name contains no `:`, so the first `:` always
separates the name from the path.

**D7.** **Merge of `overrides`.** The permitted fields are those of the `Experiment` class of CI. Other fields are
an error. `args` merges key by key: a key replaces the value of the entry, and `None` removes the key. Other fields
replace the value of the entry, and `None` removes the field.

**D8.** The configuration call has no `name` parameter. The name is the `name` of the merged entry.

**D9.** `yaml_path` and `entry` are given together or not at all. Without them, `overrides` must contain `name`
and `model`, and `args` is empty by default.

**D10.** Use of the entry fields: `name`, `model` and `args` as usual; `runner` must be `xdit`; `num_gpus` sets
`--nproc_per_node`, and without it the harness computes the number as the xDiT CLI does; `tags`, `revision` and
`entrypoint` have no effect.

**D11.** Errors in a definition raise an exception during collection. By default, pytest then runs no test.
*Rationale:* the user sees the error at once, before hours of GPU runs.

**D12.** `make_test` registers each configuration name together with its test module. The message for a duplicate
name names both test modules.
*Rationale:* the duplicate shows while the second module is imported, so the message must point to the first
definition too.

**D13.** The second-start check of the configuration fixture uses the configuration name. Its message says that
pytest ran the cases of the configuration in two parts, for example with `--ff`.

**D14.** Batched mode is not UNSUPPORTED. It is one more cause of R54: several calls of the first element in one
run.

**D15.** The limitation text of R54 says that the test results stay valid: B runs the same runner, so it makes the
same calls with the same addresses. Only the address can mislead.

**D16.** **Overlapping entries.** Two entries on the same component that are not disjoint (R52) give the path error
"overlapping entries", and every entry of the pair gets FAILED_TO_START. The check sees only the selected cases.
The capture saves each call into the one case whose selection contains the address of the call. The
specification states that the harness cannot see two equal path strings in one dictionary literal, because
Python keeps only the last value.

**D17.** **Class swap (future work).** After the initialization, in A and in B, the harness sets the class of each
attribute that a path names as its first element to a subclass that wraps `__call__`.
1. Concerns: it is a hack; the subclass must keep the object layout (no `__slots__`); exact type checks after the
   initialization fail; xDiT itself swaps pipeline classes during the load, so the harness swap must come after
   the initialization; nested pipeline calls need a stack; direct calls of the runner stay outside every pipeline
   call.
2. Size: about 5 lines for the subclass, 1 line for the swap, and about 10 lines for a tracker.
3. Harness tests: a fake runner with two pipelines that share a module, one pipeline called two times, and a direct
   call of the runner.
4. Other ways, each with its limits: a proxy on the attribute of the runner model object, a stack inspection in
   each hook, and an xDiT refactor in which one runner method makes all pipeline calls.

**D18.** **Lock file.** Each configuration has one lock file, `_locks/<name>.lock`, outside the configuration
directory.
*Rationale:* `flock` locks an open file, not a path. If the cleanup deletes and creates the lock file again, a
second session can lock the new file. A separate directory keeps the file safe from the cleanup and from a manual
deletion. `_data/<name>.lock` is not used, because a name can end in `.lock`.

**D19.** The configuration fixture starts in this order: it takes the lock; it prepares the directory (it creates
the directory with a marker if the directory is absent or empty, and otherwise checks the marker); it reads the
states; it decides between a fresh run and a reuse run. To clean, it deletes the directory and creates it again
with a new marker.

**D20.** The pytest process holds the lock until it ends. The harness never deletes a lock file.

**D21.** **Accepted limit.** Only the pytest process holds the lock. If the pytest process dies, the supervisor
needs up to the grace period to stop the launchers and the workers. During that time, a new session can clean the
directory while these processes still write into it.

**D22.** The configuration fixture raises an exception only for a second start, a lock that another session
holds, a non-empty configuration directory without a marker file, or a nonzero exit of a launcher. All cases of
the configuration then get "error".

---

## Not confirmed

**N1.** **Default self-attention backend of B.** It is an open question. The known facts:
1. `CUDNN` declares a fallback, so B refuses it.
2. `SDPA_MATH` needs about 22.9 GB per head in fp32 for the first configuration, so about 114 GB per rank for 5
   heads, plus about 138 GB of weights (not verified with a run).
3. `SDPA_EFFICIENT` accepts fp32 by its declaration. Its correctness in fp32 on ROCm is not verified.
4. `SDPA` lets PyTorch select the kernel.

**N2.** A path error in A gives FAILED_TO_START to the affected cases only, and the instance continues.

**N3.** The capture saves the CPU and CUDA random-number states before each call, and each replay restores them.

**N4.** Each replay loads its inputs into new tensors, because `MaskMetaCache` matches entries by tensor
identity. The hooks are off during the replays.

**N5.** The bitwise comparison compares raw bytes, not `torch.equal`, because NaN is not equal to NaN.

**N6.** B blocks these vendor packages: `aiter`, `flash_attn`, `flash_attn_interface`, `sageattention`,
`spas_sage_attn`, `flashinfer`, `flydsl` and `transformer_engine`. B does not block `yunchang`, because xDiT uses
it for the sequence-parallel process groups.

**N7.** The test compares A with the output of the first replay of B.

**N8.** The status NOT_REACHED_B marks a case whose call addresses in B differ from those in A. The case fails at
once.

**N9.** The case fixture raises an exception for the outcome "error" and calls `pytest.skip()` for the outcome
"skipped", because pytest gives "error" only for an exception in a fixture.

**N10.** `tests/component/conftest.py` excludes `_prototypes` from collection, because the prototype module
`harness.py` has the same name as the harness package.

**N11.** The grace period before SIGKILL is 30 seconds.

**N12.** `["key"]` selects the item `key` of a mapping in an output name.

**N13.** The status names ABOUT_TO_START, DATA_COMPLETE, PASSED, FAILED, NOT_REACHED_B and CLEAN.

**N14.** The metrics are computed on the CPU in float64, separately for each rank and call. The metrics line has
the field `yaml_entry`, which is `null` for a configuration without a YAML entry.

**N15.** In compiled mode (future work), the fidelity check compares an eager replay with the eager capture, and
the determinism check compares two compiled replays. A sub-component gets the compile settings of the nearest
compile-plan entry that contains it (the longest prefix of its path). A sub-component without such an entry is
marked "not a production compile path".

**N16.** **Global counters, open points.**
1. A selection uses only the counter values. So `b:2` also selects calls that run after call 2 of `b` ended and
   before call 3 starts. This is a known limitation. The alternative is to also track whether `b` is in a call.
2. `:0` on the last element is a syntax error, because it can never select a call.
3. The first element has no counter, and `:i` on the first element is a syntax error.
4. An element is callable if it is a method, or an `nn.Module` whose class defines its own `forward`.
5. A last element that is not callable is a path error.
6. Counters belong to objects, so two paths to the same object share one counter.
7. The ancestor rules raise no exceptions.
8. Global counters cannot express a selection such as "call 1 of block 0 inside each transformer call".
9. While a call runs, the harness keeps its call number until the call ends, so a nested call of the same
    component cannot change it.
10. An address writes 0 for an ancestor that the run never called, for example `pipe.vae:0.decode:1`.
11. The class swap then only gives each pipeline a real global counter.

**N17.** **Shared decisions across ranks, details.**
1. A rank that makes a second replay only because another rank needs it gets NON_DETERMINISTIC if its two
   replays differ. If its two replays are equal, its fidelity check passes.
2. A shared decision changes what the ranks do. It does not change the status of a rank for a call.
3. One all-gather before the replays exchanges the skip status of each rank and the call addresses that it
   captured. The call-list check runs only if no rank skips the case.
4. Each per-call decision uses one all-reduce with the maximum of an integer flag.
5. B checks the inputs of A for all calls of a case before the replays.
6. One all-gather after the support check exchanges the local results.
7. At world size 1, each shared decision equals the local result, with no special case.
8. Proposed solution for R114: a deterministic checksum of the input of each collective on each rank, in the
   capture and in each replay. The first input that differs shows the rank.

**N18.** **Rank supervisor, details.** The worker runs in its own process group, so the signals of torchrun reach
it directly, and the rank supervisor forwards no signal. The rank supervisor takes SIGTERM with `sigwaitinfo` and
writes its own end-record file. The cleanup relies on the supervisor of the prototype; a record that a process
writes after torchrun ended is lost, and this gap is accepted.

**N19.** **Bounded cleanup, details.**
1. The supervisor waits at most one kill timeout after SIGKILL. The kill timeout equals the grace period.
2. The report of a surviving process gives its PID, its state, its kernel wait function, its command line and its
   rank.
3. `run_in_pty()` raises a dedicated error with the surviving processes and the exit status. The tests of the
   prototype need a seam for this.
4. On a missed deadline, the launcher waits only for torchrun, at most one kill timeout.
5. The fixture calls `pytest.exit(reason)` without a return code, in its teardown, after surviving processes of A
   or of B.
6. Only the specification changes for this fix; the code of the prototype does not change in that task.

**N20.** **Open idea.** Record the ranks on which each call occurs, and replay a call only on these ranks, so that
other ranks continue until a later replay needs them. Its viability is not known.
