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

**UC2. Find flaky kernels.** A test shows a deviation beyond the tolerance, nondeterminism, a crash or a hang. The report names the instance and the case. At least for a deviation or nondeterminism, it also names the calls.
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

**UC10. Accept nondeterminism that is valid.** An override of one test changes the result of nondeterminism in that test from a failure to a warning. The determinism checks of all other tests stay active. B should have its own override. Requirements: IA50, IA60.
*Rationale:* some operations are validly nondeterministic. A model can have one nondeterministic component
among many, and that component must be testable. Instance B generally uses different kernels that might have
other determinism expectations.

**UC11. Later use cases.** Compiled mode, a reference at world size 1 (B1), several input sets, MI300X and other than this machine hardware, and benchmark and determinism scripts that use the foundation of the harness.

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

**HL120.** Multi-node model execution isn't supported, but single node multi-GPU is supported, hence a term "rank" is an alias to "local rank". A term "worker" is an alias to "local rank" too.

**HL130.** The pytest prototype integration in [_prototypes/pytest_integration/](../_prototypes/pytest_integration/) shows how a set of tests can share two expensive processes, select single cases, show live output, and leave no processes behind. The harness takes the parts that fit and changes them as this document requires to adapt to the requirements. The prototype is an implementation hint, not a binding rule, but the final implementation must have no weaker child process handling guarantees and child process console output visibility, than the prototype.

**HL140.** The harness itself must have a test set, ensuring the harness adheres to the requirements. Most important tests are (not exhaustive list):
- random rank fails fidelity check, or crashes, or hangs - correctly changes pytest outcomes for all cases of the configuration
- all possible values for logged statuses in rank log files correctly reflected by pytest outcomes

#### Statuses

**HLS10.** When a rank executes a model or does something else mandated by this document, it gets a status label that identifies a certain event, situation, state or a condition, that the rank seen. Statues are written in CAPITAL_LETTERS. Status only describe what the rank observed last and might apply to a component (i.e. its test case as a set of calls the component made during model execution), or to an individual case call.

**HS11.** Users do not need to see status identifiers: the harness raises verbose exceptions with explanations that a person can read.

**HLS15.** A case call status progression as the document describes it must follow directed acyclic graph rules.

**HLS20.** This document follows a convention: if a certain status X has _A or _B suffix, that only means that status X applies to a rank of instance A or B correspondingly. There are no distinct statuses X_A or X_B, there's only a status X and a rank of instance A or B on which it appears. It's always known to which instance a rank belongs, so instance specific statuses aren't generally needed.

**HLS30.** Status is written to the rank's log file immediately when it happens (FC150) as `status=X` record (the record should have other fields providing additional information). In the rank's log, when `status=` field is recorded with `case=` entry (setting a current test case identifier), the status applies to the test case. It's an error to have a `case=` field without `status=` field. When `status=` is recorded together with `case=` and `call=` identifiers, the status applies to the given call of the given case.

**HLS40.** When the harness processes data obtained from instances A and B, it loads individual case data
from each of the instances and by aggregating statuses from all ranks decides how it's going
to handle case testing. During that, rank statuses might trigger standard pytest
states: error (an exception is thrown from setup/teardown or fixture code),
failure (an exception is thrown from a test function), test skip or test pass. If an exception
is raised, it's message contain all important information to explain the reason for that.

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

**TD100.** Deadlines have default values, and a configuration (TD10) can override them. The defaults are:
1. A: 5 minutes for the `phase=initialization` (RL40), 10 minutes for the `phase=run` (RL60-RL70).
2. B: 10 minutes for the `phase=initialization` (RL40), 20 minutes for the `phase=run` (RL60-RL70).
3. each replayed call (measured between the call's starting replay status (PRE_FIDELITY or PRE_REPLAY) and ending status (FIDELITY_FAILED or FIDELITY_PASSED or REPLAYED), see RL100): 2 minutes in A and 5 minutes in B.
4. 5 (for A) or 10 (for B) minutes since the last record in the log file in all other cases

**TD110.** **Deadlines enforcement.** The configuration fixture (TD40) launches each instance sequentially using a supplementary launcher process that takes care of deadlines, child processes cleanup, actually launching an instance with torchrun and statuses of instance's case in some cases. Each rank of the instance writes its deadline -related progress and timestamps to a certain rank-specific log file known to the launcher. The launcher then periodically reads these files and uses progress records and timestamps to enforce deadlines. A missed deadline leads to a warning message and killing the instance and all its descendant processes.

**TD120.** Since the launcher is the only entity that knows of missed deadlines, and also only it can detect crashes or other premature exits, upon the instance finishing execution (whatever the reasons for instance process ending were) the launcher is responsible for reading rank log files and appending correct statuses to rank log files for the cases that were executed last (these might need `status=CRASHED` or `status=HANGED` if that happened) or don't have records at all (these might need UNSUPPORTED if whole configuration is unsupported, or FAILED_TO_START if the rank crashed or hanged and just didn't have an opportunity to try to run the case).

**TD125.** *Sanity check* The launcher knows which cases the instance should run, so after completing the above it validates that each case is mentioned on the log at least once. It's a harness bug if the launcher detects cases that are still absent.

**TD130.** At the launcher start, if `HSA_DISABLE_COREDUMP_ON_EXCEPTION` environment variable isn't defined for the launcher process, it must set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` for self and all child processes it runs. For instance B only (to reduce memory pressure), if none of `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` environment variables are defined, the launcher sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for self and all child processes it runs.

**TD140.** The launcher exits with 0 when it worked; the detailed statuses and other run information go through rank log files. A nonzero launcher exit means that the launcher failed and that no data exists, so every case of the configuration fails triggering a pytest error.

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

**CP140.** A path must resolve in A and in B. If it doesn't, the case gets (logs) `status=UNRESOLVABLE_PATH case=<case_id>`. Path errors that need a built model show only after the initialization of A; this cost is accepted.

**CP150.** The harness can change the test case names before pytest sees them, if the `-k` grammar needs it.

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

**IA70.** UNSUPPORTED marks a known limit of the harness:
1. If the harness cannot handle an argument (fp8 or MX tensors, types that it cannot serialize), only the affected case call gets UNSUPPORTED. The message names the argument.
2. If the harness cannot handle a feature of a configuration, every case of the configuration gets UNSUPPORTED. The message names the feature and the run argument with its value. The harness finds this at collection, from the xDiT settings (TD20), or when A or B loads the configuration (RL40). After that, no instance of the configuration starts or continues.

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

**IB20.** B always loads in bf16 and casts to the reference dtype if the data dtype offers less
precision (less mantissa bits) than the reference dtype (so if the reference dtype is bf16
and there's fp32 data, that higher precision data stays unmodified). It does not load a
checkpoint in the reference dtype. B converts all components at once.

**IB30.** After B loads a model, a `TorchDispatchMode` audit checks all dtypes used to run the model (including replays). It warns about dtypes with less precision than the reference dtype, and about custom calls. The check does not fail, only prints a warning informing about an imprecise dtype use, or a custom call. (it's a known limitation of version 1 needs to be addressed in future versions that these warnings aren't recorded and are produced only during instance B runs)

**IB40.** B runs the pipeline in the reference dtype with the same run arguments as A (changes of IB10 applied). It captures its own calls and runs its own fidelity check on them.

*Rationale:* if the replay is faithful with B's inputs, it is also faithful with A's inputs. That
holds if faithfulness depends on the context and state of the replay and not on the input values.
A component whose behavior depends on its inputs and on hidden state could break this assumption,
so a fidelity check on own inputs is needed.

**IB50.** After pipeline run and components fidelity check B replays the inputs of A (using A's random-number states, but restoring the original state after a replay).

**IB70.** B has three run-related parameters for the user: the reference dtype, the self-attention backend and the cross-attention backend, and also own per-path determinism override setting, and deadlines.

**IB80.** An attention backend that declares a fallback, or that needs a blocked vendor package, makes the configuration UNSUPPORTED (IA70.2). A cross-attention parameter for a model without a separate cross-attention backend is an error. The default attention backend is `sdpa_efficient`.

### Rank lifecycle

**RL10.** **Logging.** In its lifetime a rank comes through a sequence of phases (in which all ranks must participate), described in the order below. To preserve events happened/observed during rank's lifetime, each rank has a structured log file that only the rank can write to (the launcher can read from the file, TD110) during instance execution. On a rank startup, rank gets from the launcher a path to data directory to store the dumps and the status file. The status file name is `rank_<rank_number>.jsonl`. It's an error if the file existed when the rank started. Each rank appends (and flushes immediately) to their own log file progress records (TD110, TD120), status changes and all other information that requirements of this document mandate to save and pass to pytest/user.

Each log record must have at least one common field: a timestamp. Records related to a case must have another common field `case=<case_id>` - the case identifier, so one can use it to find entries related to the case. A case call related records must also have `call=<call_id>` field set. The file hence serves as a structured timestamped log of all important events the rank observed. Neither rank, nor launcher never deletes anything from the file, only appends to it.

After the instance ends, the ownership of the log files of all ranks is passed to the launcher: it reads the records of all ranks (grepping for files with a known pattern and inferring ranks from it, instead of assuming a certain number of ranks) and appends final statuses for cases that must run, but didn't finish according to the file records. This enables crashes/hangs detection as per TD110, and sets FAILED_TO_START for cases that didn't run (TD120).

**RL20.** Getting to the next phase is recorded in the rank's log with a corresponding field `phase=`. These records are used by the launcher to enforce deadlines (TD110).

**RL30.** When a rank just starts, it gets a status STARTED.

**RL40.** **phase=initialization** This is where the rank parses its xDiT setting (TD50) and determines if it can run the configuration. If the configuration gets `status=UNSUPPORTED` record with an appropriate value of `reason=` field describing what exactly isn't supported, and then rank exits with an error. This automatically forces torchrun to kill all other ranks. By reading the log files the launcher is able to correctly identify the cause and set all cases statuses to UNSUPPORTED too (IA70.2).
*Rationale for not setting case statuses:* In case of a hang or a crash, the launcher have to identify cases that didn't run to assign them appropriate statuses. We simply reuse this machinery for setting case statuses for unsupported configuration too.

**RL50.** **phase=loading** Doesn't have specific statuses.

**RL60.** **phase=run** During this phase rank runs the model pipeline and captures data. From the beginning, the rank has a set of cases it has to run; it makes a copy of this set into "unseen cases set". The rank installs hooks to tracked components and when a hook fires, if the case call conditions are met, the rank:
1. removes the case from "unseen cases set" if it's still there; logs `status=ABOUT_TO_START case=<case_id> call=<call_id> files=<files>` (where `<files>` is a 2-element tuple, each element of which is a list of filepaths to files that will contain dumped data for the call of the case. The first element is for files having data necessary to start the call, such as inputs, rng states & etc, and the second element is for outputs; Filepaths follow a pattern and can be recovered by the rank later from the generic data path it has and a call identifier),
2. saves component inputs and other applicable runtime data to the case call specific dump file it previously allocated (or files, depending on what is more convenient),
3. runs the call and
4. similarly saves outputs on the call return
5. if dumps were successful: the rank writes a `status=DUMPED case=<case_id> call=<call_id>` record to the log and appends the call identifier to an internal ordered list of calls to replay (called "replay list").

If during steps 2 or 4 the rank finds that it can't process the call data correctly (for example, some of the data to dump isn't serializable, or a tensor has quantized dtype, which can't be de-quantized without special knowledge and hence used for B), instead of step 5 that case call records `status=UNSUPPORTED case=<case_id> call=<call_id> reason=<why>`. If that happened on step 2 the rank still runs the call, but don't do steps 4 and 5. Then the pipeline run proceeds further.

**RL70.** **Duration of phase=run for the timeouts of TD110.** Since this phase execute a potentially lengthy code that isn't meant to be accounted for in the phase's deadline timeout calculation, the rank maintains an internal counter `duration_excluded` that is set to 0 when the phase starts. Execution of the code doesn't belong to the pipeline duration deadline threshold (such as dumping input, output & etc) is timed and added to the `duration_excluded`. During the phase, to each log record the rank produces, it automatically adds a  `duration_excluded=` field with the current value of the counter. Hence the launcher can always calculate a true duration of the phase for TD100-TD110 by subtracting from the `now()` timestamp, the timestamp of the latest `phase=` change, and the latest observed `duration_excluded=`.

**RL80.** Once the model finishes running, only these things could have happened to each of the calls of each of the cases that the rank should have run during the model run phase:
1. the call could have obtained UNSUPPORTED status. It makes the call excluded from replays.
2. the call could have completed successfully (this doesn't need a changed status, the original ABOUT_TO_START is ok).
3. the whole case never had a single call; the case is in the "unseen cases set" (RL60)

For each case in the "unseen cases set" the rank logs `status=NOT_REACHED case=<case_id>`.
In version 1 no cross-rank synchronization about which cases to replay is performed, and the user is responsible for choosing components that runs equally successfully (in terms of statuses) on all ranks simultaneously.

*Rationale for the absence of replay synchronization:* Synchronization adds considerable complexity to the code. Instead we assume that each component runs on every rank the same number of calls, and each of the calls is handled by the rank successfully, so ordered "replay lists" are identical across all ranks, therefore letting each rank to use its private "replay list" is harmless. The user is responsible to ensure the component they select meets the assumptions.

**RL90.** **phase=replay** Two kind of replays are needed:
- both instance A and B should replay each call for fidelity check IA40-IA60;
- instance B also replays all calls that got status=DUMPED on A and then either passed B's fidelity check or the check was disabled for a call by instance-specific override flag, using A's data casted to reference dtype to obtain reference outputs of the component.

Instance B runs both replays: first it runs fidelity check replays on its own inputs and outputs, then it runs replays on A's inputs.

**RL100.** Call replays run in "replay list" order (RL60), and in a lock step, with a stream synchronization and a barrier before and after replay. This helps to time execution properly to ensure correct call timeouts (TD100.3), and help to correctly attribute crashes, should they happen. For each call in ordered "replay list", the rank does:
1. if this is a fidelity check replay and call's case has the instance-specific fidelity check override flag set: the rank logs `status=FIDELITY_SKIPPED case=<case_id> call=<call_id>` and proceeds to the next call in the list (since all ranks obtain the same "replay list" (RL80) and have the same flag value, the same case calls will be skipped on all ranks simultaneously and that can't lead to a hang).
2. the rank reads inputs, rng states and outputs (only if it's a fidelity check), from the call replay files (the paths are recoverable from data directory path and the call identifier).
3. the rank logs `status=<status> case=<case_id> call=<call_id> file=<file>` where the `<status>` is `PRE_FIDELITY` if this is fidelity check, or `PRE_REPLAY` if this is B replaying inputs from A; `<file>` is a file path that will contain outputs of the call.
4. The rank executes a stream synchronization and a barrier.
5. the rank saves current CPU and CUDA rng states into a temporary variable; restores rng states from loaded from the input file values, execute the call with loaded inputs (B must have casted them to the reference dtype first), obtains the replayed call outputs and saves them to the `<file>`. Finally it restores the original CPU and CUDA rng states from the temporary variable.
6. The rank executes a stream synchronization and a barrier.
7. if this is a fidelity check, the rank perform bitwise comparison of each of the elements of obtained output with expected (read in point 2). Any mismatch in the outputs structure or content gives `status=FIDELITY_FAILED case=<case_id> call=<call_id>`. If the check passes, the rank logs instead `status=FIDELITY_PASSED case=<case_id> call=<call_id>`.
8. Otherwise if this isn't a fidelity check, the rank logs `status=REPLAYED case=<case_id> call=<call_id>`.

*Rationale:* strict call execution order and synchronization with barriers are necessary to ensure proper collectives functioning and along with detailed logging help to correctly attribute crashes. A fault that silently damages memory and shows only in a later case is an accepted limit.

### pytest integration

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
