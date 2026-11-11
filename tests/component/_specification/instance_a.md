# Instance A

Instance A runs the components as production runs them. It runs in eager mode.

## 1. Profile

1. The run arguments are the merged entry (the YAML entry with the overrides of the configuration) and the forced
   overrides ([test_definition.md](test_definition.md), section 3.2).
2. A uses the production kernels, quantization and communication of the configuration. The only differences from
   production are the forced overrides and the step override of the configuration.

## 2. Flow of one worker

1. **Build and initialize.** The worker does the same steps as `main` in
   [runner.py:145-161](../../../xfuser/runner.py#L145-L161):
   1. parse the flags with `xFuserArgs.add_runner_args(parser).parse_args(flags)`;
   2. create `runner = xFuserModelRunner(vars(args))`;
   3. compute `input_args = runner.preprocess_args(vars(args))`;
   4. call `runner.initialize(input_args)`.

   The worker runs these steps and all later steps under `torch.inference_mode()`, as
   [runner.py:166-168](../../../xfuser/runner.py#L166-L168) does.
2. **Check support.** The worker checks the conditions in [unsupported.md](unsupported.md). If one is true, every
   case gets UNSUPPORTED with the reason, and the instance ends.
3. **Resolve paths.** The worker resolves the paths of the selected cases
   ([component_paths.md](component_paths.md), section 3).
4. **Install hooks** ([component_paths.md](component_paths.md), section 4).
5. **Pipeline run.** The worker calls `runner.run(input_args)` one time. The hooks capture each selected call
   (section 3).
6. **Replays.** For each selected case and each captured call, the worker replays the call and does the fidelity
   and determinism checks (section 4).
7. **End of each case.** After the replays of one case, the worker calls `torch.cuda.synchronize()` and a barrier.
   Then it writes a progress record. An asynchronous GPU fault thus shows up inside the case that caused it.
8. **State files.** The worker writes the result of each case into its state file ([files.md](files.md),
   section 4).

## 3. Capture

1. The capture saves each selected call at once. Call numbers are non-negative, so the hooks know at once whether
   a call is selected.
2. For each call and rank, the capture saves:
   1. the positional and keyword arguments, before the component runs;
   2. the CPU and CUDA random number generator states, before the component runs;
   3. the output, after the component runs.
3. The capture copies tensors to the host and writes them with the serializer ([files.md](files.md), section 7).
4. If a call has an argument that the serializer does not support, or an fp8 or MX-format tensor argument, the
   case gets UNSUPPORTED with a message that names the argument. The capture stops for that case. The other cases
   continue.
5. The capture records the address of every call of a listed component, also of calls that the selection does not
   include. The test function needs these addresses to report calls that did not occur.

## 4. Replays, fidelity and determinism

### 4.1 One replay

1. The worker loads the inputs from disk into new tensors on the device. It must not refill old tensors in place.
   The reason: `MaskMetaCache` in xDiT matches entries by tensor identity
   ([attention_mask.py:75](../../../xfuser/model_executor/layers/attention_mask.py#L75-L75)). A tensor that changes in
   place can thus return stale metadata.
2. The worker restores the captured random number generator states.
3. The worker calls the component directly, with the capture hooks disabled:
   1. a module: `module(*args, **kwargs)`;
   2. a method: the original bound method, without the wrapper of the harness.
4. Each replay writes a progress record before it starts. The launcher uses these records for the per-call deadline
   ([processes.md](processes.md), section 4).

### 4.2 Bitwise comparison

1. Two outputs are bitwise equal if the structure, the non-tensor values, and each tensor leaf are equal.
2. Two tensor leaves are equal if they have the same dtype and shape, and the same bytes in a contiguous layout.
3. The comparison uses the bytes, not `torch.equal`. Reason: `torch.equal` treats NaN as unequal to NaN, so equal
   outputs with NaN would not compare as equal.

### 4.3 Algorithm for one captured call

1. Replay the call one time. Compare the replay output with the captured output.
2. If they are equal, the fidelity check passes.
3. If they are not equal, replay the call a second time. Compare the two replay outputs.
   1. If the two replays differ, the status is NON_DETERMINISTIC_A.
      1. Without the determinism override of A, the case fails at once.
      2. With the override, the case continues with a warning. The warning also says that the fidelity of this call
         is unknown. Two different replays cannot show whether hidden state is also missing.
   2. If the two replays are equal, but differ from the capture, the status is REPLAY_MISMATCH_A. The case fails at
      once. The determinism override does not change this.
4. Edge case: if the nondeterminism of a component is rare, two replays can be equal by chance. The result is then a
   REPLAY_MISMATCH_A that is really nondeterminism. The error is a false failure, never a hidden failure.

### 4.4 Meaning of REPLAY_MISMATCH

1. The component is deterministic in the replay, but the replay does not reproduce the captured call.
2. The usual cause is hidden state that the replay does not restore, for example a global step counter or a cache.
3. Then the replays do not test the production behavior, so the test cannot trust any comparison of this call.

## 5. The output that the test compares

The test compares the captured output of A with the output of B. When the fidelity check passes, the replay output
of A is bitwise equal to the captured output. Under the determinism override, the captured output is what
production actually computed.

## 6. Statuses that A can set

1. UNSUPPORTED, for the configuration or for one case.
2. FAILED_TO_START, for a path error, or for every case without data after a failure outside the replays.
3. NON_DETERMINISTIC_A and REPLAY_MISMATCH_A, for one call of a case.
4. CRASHED_A, HUNG_A and ERROR_A, for the case whose replay was running.
5. For a case whose capture and replays finished, A records that its data is complete. The case gets
   DATA_COMPLETE only when B also finishes it.

[statuses.md](statuses.md) lists all statuses. [processes.md](processes.md), section 5 tells how the launcher
attributes crashes and hangs.

## 7. B after A

The configuration fixture starts B only if A ended clean. "Clean" means that the launcher of A recorded no crash, no
hang, no error, and no UNSUPPORTED configuration.
