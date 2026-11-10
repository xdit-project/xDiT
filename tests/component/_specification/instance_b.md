# Instance B

Instance B is the eager reference. It runs the components in the reference dtype with trusted, plain kernels.

## 1. Purpose

1. B gives the reference outputs for the inputs that A captured.
2. B must not depend on vendor kernels, quantization, compiled code or low-precision arithmetic. Then a difference
   between A and B shows the error of the production path.
3. B keeps the parallel layout of A: the same world size, Ulysses degree and parallel VAE. Thus B does not find
   errors that the parallel split causes in A and B in the same way. Instance B1 finds them in the future
   ([future_work/index.md](future_work/index.md), section 2.2).

## 2. Reference profile

### 2.1 Vendor packages

1. Before the worker imports `xfuser`, it sets `sys.modules[name] = None` for each vendor kernel package. An import
   of such a package then raises `ImportError`.
2. The initial list: `aiter`, `flash_attn`, `flash_attn_interface`, `sageattention`, `spas_sage_attn`,
   `flashinfer`, `flydsl`, `transformer_engine`. The harness keeps this list in one place.
3. `yunchang` must not be blocked. xDiT uses it for the sequence-parallel process groups
   ([parallel_state.py:514-516](../../../xfuser/core/distributed/parallel_state.py#L514-L516)), and
   [usp.py:11-12](../../../xfuser/model_executor/layers/usp.py#L11-L12) imports it when a GPU is present.
4. If xDiT cannot start with these packages blocked, B ends with ERROR_B. The fix is then an xDiT change, which
   the harness reports but does not make.

### 2.2 Run-argument overrides of the profile

B adds these overrides after all other layers ([test_definition.md](test_definition.md), section 3.2):

1. no GEMM quantization: `gemm_quantization` `none`; `use_fp8_gemms`, `use_fp4_gemms`, `use_int8_gemms` and
   `use_fp8_text_encoder` off;
2. no FP8 communication: `use_fp8_comms` off;
3. no step cache: `cache_method` removed; `use_teacache` and `use_fbcache` off;
4. `attention_backend` set to the self-attention backend of B;
5. `cross_attention_backend` set to the cross-attention backend of B.

Compile is already off through the forced overrides.

### 2.3 Full float32 matmul precision

The worker sets these values before the model loads:

1. `torch.backends.cuda.matmul.allow_tf32 = False`;
2. `torch.backends.cudnn.allow_tf32 = False`;
3. `torch.set_float32_matmul_precision("highest")`.

Without these values, fp32 matmuls can run at reduced precision.

### 2.4 Dtype audit

1. During the pipeline run and the replays, B runs under a `TorchDispatchMode` audit.
2. The audit warns when an operation returns a floating-point tensor with a dtype below the reference dtype.
3. The audit warns when an operation is not in the `aten` namespace. Such an operation can be a vendor kernel.
4. The audit does not fail a case. The case state records the warnings, and the test function prints them.
5. Limit: the audit sees only the outputs of operations. A kernel that computes in lower precision inside one
   `aten` operation does not show up.

## 3. Attention backends

### 3.1 Validation

B validates its backend parameters when it starts:

1. B refuses a backend whose `Spec` declares a fallback. Reason: `Spec.run` sends every call that the backend
   rejects to the fallback, without a log message
   ([spec.py:316-320](../../../xfuser/core/attention/spec.py#L316-L320)). Example: `CUDNN` accepts only half
   precision and falls back to `SDPA_EFFICIENT`
   ([spec.py:39-40](../../../xfuser/core/attention/backends/sdpa/spec.py#L39-L40)). In fp32, every call would thus
   run `SDPA_EFFICIENT`.
2. A backend that needs a vendor package fails at the start of B, because the package is blocked.
3. If the model does not have the `cross_attention_backend` capability, a cross-attention parameter is an error.
4. A validation failure ends B with ERROR_B and a message that names the parameter.

### 3.2 Default self-attention backend (open question, Not confirmed)

Version 1 has no confirmed default for the self-attention backend of B. A new chunked exact backend is future work
([future_work/index.md](future_work/index.md), section 2.4). The backends without a vendor package are in
[backends/sdpa/spec.py:8-43](../../../xfuser/core/attention/backends/sdpa/spec.py#L8-L43). The facts for a decision:

1. `CUDNN`: refused, because it declares a fallback (section 3.1).
2. `SDPA_MATH` computes the full score matrix. For the first configuration, one head needs about 22.9 GB in fp32
   (75,600 × 75,600 × 4 bytes). With 40 heads and Ulysses degree 8, each rank has 5 heads. If the kernel computes
   all heads at once (not verified), the scores need about 114 GB per rank, in addition to about 138 GB of fp32
   weights. This does not fit on MI300X (192 GB).
3. `SDPA_EFFICIENT`: its `Spec` accepts fp32. It is not verified that the ROCm kernel computes fp32 correctly at the
   shapes of the first configuration (head dimension 128, 75,600 tokens).
4. `SDPA`: the PyTorch dispatcher selects the kernel itself. The reference would then depend on a choice that the
   harness does not control.
5. `SDPA_FLASH`: not examined for fp32.

### 3.3 Default cross-attention backend

The default is `sdpa_math`. Cross-attention in Wan has 512 text keys, so the full score matrix is small.

## 4. Flow of one worker

1. **Start with the profile.** Block the vendor packages (section 2.1) and set the matmul precision (section 2.3).
2. **Build and initialize** as A does ([instance_a.md](instance_a.md), section 2, item 1), with the run arguments
   of the profile. The weights load in the production dtype. For the first configuration, this is bf16
   ([wan.py:414](../../../xfuser/model_executor/models/runner_models/wan.py#L414-L414)).
3. **Validate the backend parameters** (section 3.1).
4. **Check support** ([unsupported.md](unsupported.md)).
5. **Convert all components at once** to the reference dtype:
   1. Convert each floating-point parameter and buffer whose dtype is below the reference dtype. Keep tensors whose
      dtype is equal to or above the reference dtype.
   2. Convert in place: `tensor.data = tensor.data.to(reference_dtype)`. A probe confirmed that this works for
      parameters created under `torch.inference_mode()` (local evidence, one torch version).
   3. Upcasting the bf16 weights is sufficient. B does not load an fp32 checkpoint.
6. **Resolve paths and install hooks**, as A does.
7. **Pipeline run of B.** Call `runner.run(input_args)` one time with the same run arguments as A. The pipeline
   then runs in the reference dtype, because the Wan pipeline casts its inputs to the dtype of the transformer
   ([pipeline_wan_i2v.py:114-115](../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L114-L115)). The
   hooks capture the calls of the listed components, as in A.
8. **Fidelity check of B.** Replay the calls that B captured, with the algorithm of A
   ([instance_a.md](instance_a.md), section 4.3). The statuses are NON_DETERMINISTIC_B and REPLAY_MISMATCH_B. The
   determinism override of B applies.
9. **Replays of the inputs of A.** For each selected case, call and rank:
   1. load the inputs that A captured on the same rank;
   2. convert floating-point tensors with a dtype below the reference dtype to the reference dtype; keep integer and
      boolean tensors;
   3. restore the random number generator states that A captured;
   4. replay the call at least two times;
   5. compare the replay outputs bitwise. If they differ, the status is NON_DETERMINISTIC_B;
   6. dump the output of the first replay. The test compares this output with A.
10. **End of each case.** Synchronize, barrier and progress record, as in A.
11. **State files.** Write the results of B into the state files.

## 5. Calls that B does not reach

A and B use the same run arguments, so B must produce the same call addresses as A. A difference means that the
pipeline run of B did not reach a call that A captured, or the reverse. Then the fidelity of B for that call is not
checked. The status is NOT_REACHED_B (the name is Not confirmed), and the case fails at once.

## 6. NaN and Inf

NaN or Inf in an output of B is a problem of the reference. The test function reports it as a failure with the
text "reference problem" ([checks.md](checks.md), section 4).

## 7. Memory

1. The fp32 weights of the first configuration need about 138 GB on each rank, because Ulysses replicates the
   weights:

   | Component | Parameters | fp32 size |
   |---|---|---|
   | text encoder | 5.68 B | 22.7 GB |
   | each transformer | 14.29 B | 57.2 GB |
   | VAE | 0.13 B | 0.5 GB |

2. This fits on MI300X (192 GB). The activation memory in fp32 at 720p is not measured, especially for the VAE
   decode.

## 8. Statuses that B can set

1. FAILED_TO_START, for a path that does not resolve in B, or for every case without data after a failure outside
   the replays.
2. NON_DETERMINISTIC_B, REPLAY_MISMATCH_B and NOT_REACHED_B.
3. CRASHED_B, HUNG_B and ERROR_B.
4. UNSUPPORTED, for one case whose inputs B cannot use.
