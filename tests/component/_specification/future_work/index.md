# Future work

This document lists the work after version 1. It is for coding agents and for humans. Each item describes the
limitation of version 1 and how the future work removes it.

## 1. Order

1. Version 1 must first run end to end and validate the risky assumptions ([../index.md](../index.md),
   section 3.1).
2. Compiled mode comes next. It needs refactors of xDiT ([compiled_mode.md](compiled_mode.md)).
3. B1 comes after compiled mode (section 2.2).
4. The order of the other items is not decided.

## 2. Items

### 2.1 Compiled mode

1. **Limitation.** Version 1 runs only in eager mode. It forces `use_torch_compile` off. Thus it does not test the
   compiled code that production runs.
2. **Solution.** See [compiled_mode.md](compiled_mode.md).

### 2.2 Instance B1: a reference at world size 1

1. **Limitation.** B keeps the parallel layout of A: the world size, the Ulysses degree and the parallel VAE. An
   error of the parallel split that A and B share gives the same wrong result in both instances. The comparison of
   A with B cannot show it. Version 1 finds a multi-GPU problem only if it occurs only in the code path of A, for
   example in FP8 communication or in quantized attention with Ulysses.
2. **Solution.**
   1. A third instance, B1, runs the reference profile of B at world size 1.
   2. B1 replays only whole components, for example `pipe.transformer`. Inputs inside a component, such as the
      inputs of a block, are split by sequence across the ranks, so B1 cannot replay them.
   3. B1 needs the full inputs. The harness joins the inputs of the ranks of A, or takes them from the rank that
      holds the full tensor. This mapping must be defined for each kind of split.
   4. Comparisons:
      1. B against B1 shows the error of the split alone. Both are in the reference dtype, so a tight tolerance
         fits.
      2. A against B1 shows the total error.
   5. **Memory.** The whole model must fit on one GPU in the reference dtype. For the first configuration, the fp32
      weights need about 138 GB. The activations of the full sequence (75,600 tokens) come in addition.
3. **Dependency.** After compiled mode.

### 2.3 Multiple input sets

1. **Limitation.** Version 1 uses one input set: the run arguments of the configuration.
2. **Solution.** See [multiple_input_sets.md](multiple_input_sets.md).

### 2.4 Chunked exact attention backend

1. **Limitation.** B has no confirmed default for self-attention ([../instance_b.md](../instance_b.md),
   section 3.2). The backends without a vendor package either need too much memory (`SDPA_MATH`), or their fp32
   correctness on ROCm is not verified (`SDPA_EFFICIENT`), or they declare a fallback (`CUDNN`).
2. **Solution.** A new backend type in the attention backend registry of xDiT:
   1. It computes exact attention over blocks of queries, with plain matmul and softmax in the reference dtype.
   2. Memory stays bounded. Example: a block of 4,096 queries against 75,600 keys needs about 1.24 GB for each head
      in fp32. The full score matrix needs about 22.9 GB for each head.
   3. The reference then depends only on matmul and softmax, not on a fused kernel whose fp32 correctness nobody
      checked. An example of the risk: the comment in the SDPA backend says that the cuDNN kernel returns NaN for
      fp32 ([spec.py:35-38](../../../../xfuser/core/attention/backends/sdpa/spec.py#L35-L38)).
   4. It declares no fallback, because B refuses backends with a fallback.
   5. **Ring attention (later step).** Ring attention merges the partial results of the ranks with the
      log-sum-exp of the softmax. A backend returns `(output, softmax_lse)`, and `lse` is `None` when the backend
      does not produce it ([spec.py:131-132](../../../../xfuser/core/attention/spec.py#L131-L132)). Such a backend
      declares `ring=NEVER` ([spec.py:174-178](../../../../xfuser/core/attention/spec.py#L174-L178)). The first
      version of the new backend does not need the log-sum-exp, because the first configuration does not use
      ring attention (`ring_degree` has the default 1). A later version returns it, so that B can run ring
      configurations.

### 2.5 Results after a crash

1. **Limitation.** Version 1 does not recover from crashes and hangs. After a failure of A, B does not start, so
   the cases that A finished get FAILED_TO_START. After a failure of B, the remaining cases get FAILED_TO_START.
2. **Solution.**
   1. Run B for the cases whose data A finished.
   2. After a failure of B, start B again without the failed case.
   3. After a failure of A, start A again without the failed case. This needs a new pipeline run, because the
      capture happens in the pipeline run.

### 2.6 Hardware tags

1. **Limitation.** A YAML entry can be meant for one GPU architecture. Example: the first configuration has the tag
   `gfx950`. On other hardware, the run fails deep inside a kernel, which looks like a defect.
2. **Solution.** Compare the tags of the entry with the GPU architecture (`gcnArchName` of the device properties).
   On a mismatch, give UNSUPPORTED with a clear reason.

### 2.7 Fidelity check with a tolerance for nondeterministic components

1. **Limitation.** Under the determinism override, two different replays cannot show whether hidden state is also
   missing. The fidelity of such a call is unknown ([../instance_a.md](../instance_a.md), section 4.3).
2. **Solution.** Measure the spread between replays. Then compare the replay with the capture against a tolerance
   derived from that spread. A deviation much larger than the spread shows missing hidden state.

### 2.8 Arguments that a component changes in place

1. **Limitation.** Version 1 compares only the return value. Some components also write into their arguments. Example:
   the Wan VAE calls `self.decoder` once for each latent frame and passes the list `feat_cache`, which the decoder
   changes
   ([autoencoder_kl_wan.py:1197-1205](../../../../../external/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1197-L1205)).
   A wrong value in such a list stays hidden until a later call.
2. **Solution.** Save the arguments again after the call, and compare the changed arguments as extra outputs. The
   output names then need a prefix for arguments.

### 2.9 Refactors that remove hidden state

1. **Limitation.** A replay is correct only when everything that a call reads is in its arguments, the weights and
   the run configuration. xDiT has three kinds of state that change from call to call (counts are approximate):
   1. **Step schedules.** Eight transformer forwards call `get_runtime_state().increment_step_counter()`, for
      example the Wan forward
      ([transformer_wan.py:477](../../../../xfuser/model_executor/models/transformers/transformer_wan.py#L477-L477)).
      When a hybrid schedule is active, the counter selects the attention backend and the GEMM precision
      ([runtime_state.py:465-475](../../../../xfuser/core/distributed/runtime_state.py#L465-L475)). VSA also changes
      its state for each step.
   2. **PipeFusion patch state.** About 37 reads of `patch_mode`, `pipeline_patch_idx`, `num_pipeline_patch` and
      `pp_patches_*` in transformers, layers and attention code.
   3. **Context in shared dictionaries.** About 41 writes in 11 transformer files. In Wan: `attention_kwargs`,
      `_self_attn_kwargs` with `thw` and `valid_kv_len`, and `_sp_pad["local_rows"]`. The parent forward writes these
      values, and processors and hooks read them.

   Version 1 works without these refactors: it does not support schedules, VSA or PipeFusion, and the shared
   dictionaries hold the right values because each instance runs its own pipeline at the same shapes. The fidelity
   checks detect a violation.
2. **Solution: four refactors, one for each feature.**
   1. **Step plan through a scoped holder.**
      1. A schedule becomes a pure lookup: `plan_for(step_index)` returns an immutable value, for example
         `StepPlan(attention_backend, high_precision_gemm, vsa_drop_rate, vsa_use_dense)`.
      2. The denoising loop computes the plan once for each step. The loop knows the step index: `i` in
         `for i, t in enumerate(timesteps):`
         ([pipeline_wan_i2v.py:179](../../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L179-L179)).
      3. The loop sets the plan for the duration of the step with a context manager (`with step_plan(plan): ...`).
         The holder is a plain module-level object that eager code sets. It is not a `ContextVar`, and the
         signatures of diffusers do not change. Consumers read the plan as they read the runtime state today, but no
         callee changes it.
      4. The plan replaces `increment_step_counter()` in the forwards. Both guidance branches of a step get the same
         plan. Thus the step multiplier of the Wan runner disappears
         ([wan.py:167](../../../../xfuser/model_executor/models/runner_models/wan.py#L167-L167)).
      5. The capture stores the plan (a few scalars), and a replay sets it. A repeated call then changes nothing.
      6. Compiled code guards on the value that it reads. This is the same as today with
         `get_runtime_state().attention_backend`.
   2. **Explicit PipeFusion patch state.** The patch state becomes an explicit value that the caller gives, not a
      global value that changes during the run.
   3. **Explicit per-call context.** The parent forward passes its per-call context as explicit arguments,
      preferably as one context object, instead of shared dictionaries.
   4. **Glue code as components.** Code between components moves into methods, so that a path can list it.
      Examples: the guidance combination
      ([pipeline_wan_i2v.py:234](../../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L234-L234)) and the
      de-normalization of the latents before the decode
      ([pipeline_wan_i2v.py:271](../../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L271-L271)).

### 2.10 Per-step schedules and VSA

1. **Limitation.** UNSUPPORTED in version 1 ([../unsupported.md](../unsupported.md)).
2. **Solution.** The step plan (section 2.9, item 2.1).

### 2.11 PipeFusion

1. **Limitation.** UNSUPPORTED in version 1.
2. **Solution.** The explicit patch state (section 2.9, item 2.2). PipeFusion also has its own warmup pass, which
   compiled mode must move out of the compile step ([compiled_mode.md](compiled_mode.md), section 3.2).

### 2.12 Step caching (far future)

1. **Limitation.** UNSUPPORTED in version 1. A step cache keeps state across steps, so a single call depends on
   earlier calls.
2. **Solution.** Not designed. It needs a way to capture and restore the cache state for each call.

### 2.13 FSDP

1. **Limitation.** UNSUPPORTED in version 1. With FSDP, xDiT casts a component to the dtype of the strategy before
   it wraps the component, for example bf16 for `transformer_2` of Wan2.2 I2V
   ([wan.py:342-345](../../../../xfuser/model_executor/models/runner_models/wan.py#L342-L345)). FSDP holds the
   parameters of one unit in one flat allocation and rejects a mixture of dtypes in it
   ([sharding.py:395-402](../../../../xfuser/core/distributed/sharding.py#L395-L402)). Thus B cannot convert the
   parameters after the load.
2. **Solution.**
   1. B overrides the dtype of the FSDP strategy with the reference dtype. Then the cast before the wrap produces
      flat parameters in the reference dtype.
   2. B must keep the values of the weights of A. A uses weights rounded to the production dtype. If B loads fp32
      values directly from the checkpoint, its weights differ from those of A, and the comparison includes the
      rounding of the weights. Some checkpoint files store fp32 values (for the first configuration, the VAE file
      and at least one transformer shard). Thus B must first round each weight to the production dtype and then
      cast it to the reference dtype. It must be verified whether the loader already does this.
   3. The modules that the loader keeps in fp32 stay out of the FSDP units today
      ([sharding.py:395-400](../../../../xfuser/core/distributed/sharding.py#L395-L400)). This must still work
      with the reference dtype.

### 2.14 Dequantization of fp8 and MX inputs

1. **Limitation.** A case whose component receives an fp8 or MX-format tensor is UNSUPPORTED in version 1.
2. **Why it is not trivial.** The conversion of the numbers is exact (fp8 to fp32 is exact). The meaning is the
   problem:
   1. A quantized tensor is usually one half of a pair: the data and a scale tensor (per tensor, or per block of 32
      values for MX formats). The real value is the data times the scale. An upcast of the data alone gives wrong
      numbers.
   2. The component of A can take data and scale as two arguments, while the reference path of B expects one tensor
      in high precision. The harness must map one argument list to the other.
   3. fp8 tensors have their own dtypes and can be detected. MX data is usually packed into `uint8` (two 4-bit values
      in one byte), so the dtype alone cannot show it.
3. **Solution.** Per-format knowledge of the pairing of data and scales, and a map from the arguments of A to the
   arguments of B.

### 2.15 Several pipelines on one runner model object

1. **Limitation.** UNSUPPORTED in version 1 ([../component_paths.md](../component_paths.md), section 2.1). The
   harness cannot tell which pipeline calls a shared component.
2. **Solution.** An xDiT refactor sends every pipeline call through one method of the runner that the harness can
   observe. Then any pipeline attribute can be the first path element, and the call address gets a stage for the
   pipeline.
3. **Rejected alternative.** A change of the class of the pipeline instance can hook `__call__`. This is a hack, so
   it is not used.

### 2.16 Full cleanup on request

1. **Limitation.** The harness cleans a configuration directory only when every recorded case passed. Otherwise the
   user must delete the directory.
2. **Solution.** A command-line option or an environment variable that cleans the configuration directories before
   the run. The safety guards of [../files.md](../files.md), section 6 stay.
3. **Rejected alternative.** pytest `--basetemp` deletes its directory at the start of each session
   ([tmpdir.py:149-153](../../../../../../venv/lib/python3.12/site-packages/_pytest/tmpdir.py#L149-L153)). Without
   `--basetemp`, pytest makes a new numbered directory for each session. Both behaviors stop the reuse of data
   across sessions, which the harness needs ([../files.md](../files.md), section 5).

### 2.17 Markers and CI

1. **Limitation.** The component tests have no pytest markers, and CI does not run them.
2. **Solution.** Not designed yet. Markers that depend on the configuration, for example `multi_gpu` from the world
   size, are one option.

### 2.18 Tolerances for each rank

1. **Limitation.** One tolerance applies to all ranks.
2. **Solution.** The checks already run for each rank separately ([../checks.md](../checks.md), section 4.2). A
   tolerance can thus take one value for each rank without other changes.

### 2.19 Models with sparse attention in B

1. **Limitation.** B uses dense attention. A model whose capabilities include `supports_sparse_attention_backends`
   rejects a dense backend at init
   ([base_model.py:842-847](../../../../xfuser/model_executor/models/runner_models/base_model.py#L842-L847)). B then
   ends with ERROR_B. Wan2.2 I2V does not set this capability, so the first configuration is not affected.
2. **Solution.** A way for reference builds to skip this check.

### 2.20 Batched mode (Not confirmed)

1. **Limitation.** With `batch_size` set, the runner calls the pipeline several times in one run, so the count of
   `pipe` calls is wrong. The harness gives UNSUPPORTED.
2. **Solution.** Count the pipeline calls through the observable entry point of section 2.15.

### 2.21 Run-level determinism checks and benchmarks

1. **Limitation.** The tests check determinism only through replays of single calls. They do not measure time.
2. **Solution.** Separate tools that reuse the harness:
   1. checksums of the inputs and outputs of every call in several pipeline runs. The first call whose inputs match
      but whose outputs differ is itself nondeterministic. The checksum must be deterministic itself, for example a
      position-weighted integer sum over the bytes;
   2. benchmarks of single components: warmup replays, then timed replays with CUDA events, on an exclusive node.

### 2.22 Shapes that no configuration produces

Arbitrary shape grids belong to kernel-level tests. They are outside this design. A sub-component can get a new
shape only through a new whole-model shape ([multiple_input_sets.md](multiple_input_sets.md), section 3).
