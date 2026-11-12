I found **nine issues** in version 1. The most serious are distributed replay coordination, conflicting component-path rules, and the unresolved reference backend. I excluded future work.

4. **High — Configuration-name validation permits paths outside the configuration directory.**

   [test_definition.md:237](/app/xDiT/tests/component/_specification/test_definition.md:237) uses the name directly in `_data/<name>/` and permits names containing only letters, digits, `_`, `.` and `-`.

   Both `.` and `..` satisfy those rules. They resolve respectively to the shared `_data` directory and the test-module directory, rather than separate configuration directories. This breaks the isolation assumed by cleanup and locking. The marker check can reject some situations, but it does not make these names valid.

   Explicitly rejecting `.` and `..`, and checking resolved-path containment, would close this gap.

5. **Medium — B disables GEMM quantization but retains settings that require it.**

   The reference overrides in [instance_b.md:30](/app/xDiT/tests/component/_specification/instance_b.md:30) set `gemm_quantization` to `none`, but do not remove `gemm_config` or reset advanced precision-target settings.

   xDiT loads those settings afterward in [args.py:549](/app/xDiT/xfuser/config/args.py:549). Its validation rejects advanced targets without a tiered quantization profile in [args.py:1600](/app/xDiT/xfuser/config/args.py:1600). For example, a valid production configuration with `gemm_high_precision_targets: none` becomes invalid under B’s prescribed overrides.

   The first configuration does not use this option, but the configuration API permits it and the unsupported list does not exclude it. B’s profile needs to normalize the dependent quantization settings together.

7. **Medium — Reuse has no rule for `DATA_COMPLETE`.**

   [statuses.md:20](/app/xDiT/tests/component/_specification/statuses.md:20) defines `DATA_COMPLETE`, but the reuse table at [statuses.md:27](/app/xDiT/tests/component/_specification/statuses.md:27) omits it.

   This state can remain without any worker failure. For example, both instances finish, the first test fails, and pytest stops because of `-x`. Later cases retain complete data but have not been checked. The next invocation must reuse the directory, yet their outcome is undefined.

   These cases need an explicit reuse rule, presumably comparison using their stored data.

8. **Medium — Skipped and error cases cannot reach their designated metrics writer.**

   [pytest_integration.md:69](/app/xDiT/tests/component/_specification/pytest_integration.md:69) makes the case fixture raise or skip for error and unsupported states. Therefore, their test functions do not run.

   However, [checks.md:26](/app/xDiT/tests/component/_specification/checks.md:26) assigns the no-metrics record to the test function, and [files.md:117](/app/xDiT/tests/component/_specification/files.md:117) explicitly requires records for unsupported and failed-to-start cases.

   The reporting responsibility needs to move to a fixture or reporting hook for outcomes that stop during fixture setup.

9. **Medium — Required metric values cannot be represented directly in JSON.**

   [checks.md:76](/app/xDiT/tests/component/_specification/checks.md:76) defines infinity for some relative errors and NaN for cosine similarity when a norm is zero. [files.md:97](/app/xDiT/tests/component/_specification/files.md:97) requires JSON Lines, but specifies no encoding for these values.

   NaN and infinity are not valid JSON numbers. A strict writer raises an exception; Python’s permissive default writes nonstandard JSON. Even equal, finite, all-zero outputs trigger this problem through the reported cosine value.

   The format needs an explicit representation for nonfinite metric values.

These findings have high confidence as specification contradictions or omissions. End-to-end GPU feasibility remains unverified. I used source inspection and small CPU-only probes; I did not run model/GPU code or change repository files.


1. **No owner for the state files.** [instance_a.md:34-35](xDiT/tests/component/_specification/instance_a.md#L34-L36) says that the worker writes the state files. [files.md:39-41](xDiT/tests/component/_specification/files.md#L39-L42) names only the configuration fixture, the launchers and the test function as writers. Neither document says which process combines the results of the 8 ranks into one file. An implementer must guess, and if several ranks write one file, results can be lost.

2. **Missing future-work entry.** [unsupported.md:3-4](xDiT/tests/component/_specification/unsupported.md#L3-L5) says that each item has a future-work entry. The serializer item (section 2, item 2) has none: grep for "serializ" in future_work/index.md finds nothing.

3. **Unclear order of B's steps.** [instance_b.md:116-127](xDiT/tests/component/_specification/instance_b.md#L116-L128) does not say whether B runs the fidelity check and the replays of A's inputs one case at a time, or each step over all cases. "End of each case" (item 10) suggests one case at a time.


-----------------

20. **Low — The self-attention reference remains unresolved, and the math-backend memory estimate is incomplete.**

   [instance_b.md:76](/app/xDiT/tests/component/_specification/instance_b.md:76) leaves B’s default self-attention backend unconfirmed. This directly affects the required first configuration.

   At the specified 75,600-token shape, five fp32 heads require **114.3 GB** for one score tensor. The math implementation creates the scores and then a separate softmax output: see [attention.cpp:893](/app/_my_src/pytorch/aten/src/ATen/native/transformers/attention.cpp:893) and [attention.cpp:672](/app/_my_src/pytorch/aten/src/ATen/native/transformers/attention.cpp:672). Both tensors are live during softmax.

   Adding the specification’s approximately 138 GB of weights gives **at least 366.6 GB per rank**, before other activations. That exceeds the MI355X’s 288 GB. [AMD specifications](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf)

   This is a code-based estimate for the inspected math implementation, not a measured peak. Version 1 may still be feasible with a suitable memory-efficient backend, but that path remains unvalidated.

   =========

   ## In progress

1. **High — Replay decisions are not coordinated across ranks.**

   [instance_a.md:78](/app/xDiT/tests/component/_specification/instance_a.md:78) makes each worker run a second replay only if its first replay differs from its capture. B uses the same algorithm for its fidelity check.

   Wan attention performs collective communication through [usp.py:142](/app/xDiT/xfuser/model_executor/layers/usp.py:142). If one rank observes a mismatch and another does not, the ranks can execute different replay counts. One rank can enter the second replay while another enters the next call or the case-ending barrier. This can hang or compare calls with mixed inputs.

   The specification needs a shared decision across participating ranks before an optional replay or early case termination.

3. **High — The ancestor rules conflict with the required VAE case.**

   [component_paths.md:187](/app/xDiT/tests/component/_specification/component_paths.md:187) explicitly allows uncalled ancestors, including `vae` when the pipeline calls `vae.decode`. But [component_paths.md:210](/app/xDiT/tests/component/_specification/component_paths.md:210) requires an exception whenever a component runs outside its path parent’s call.

   The actual pipeline statement is `video = self.vae.decode(latents, return_dict=False)[0]` at [pipeline_wan_i2v.py:272](/app/xDiT/xfuser/model_executor/pipelines/pipeline_wan_i2v.py:272). The `vae` module itself is not called. Applied literally, the ancestor rule rejects the specification’s own `pipe.vae.decode` case.

   The rule needs to distinguish the immediate path parent from the nearest ancestor that actually participates in call counting.

6. **Medium — Directory cleanup can invalidate the locking guarantee.**

   [files.md:53](/app/xDiT/tests/component/_specification/files.md:53) requires cleanup before a fresh run. [files.md:65](/app/xDiT/tests/component/_specification/files.md:65) permits directory deletion while placing the session’s lock file inside that directory.

   If cleanup removes and recreates the lock file, the first session retains a lock on the old file, while another session can lock the replacement. I confirmed this behavior with a small CPU-only probe: both exclusive locks succeeded.

   The specification needs to define lock acquisition before state inspection and preserve the locked file throughout cleanup, or place the lock outside the cleaned directory.