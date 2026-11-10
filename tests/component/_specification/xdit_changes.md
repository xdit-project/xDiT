# Changes in xDiT

## 1. Changes for version 1

Version 1 needs no change in xDiT.

1. The harness takes the model from the xDiT runner and changes nothing in the xDiT code. It adds hooks and
   instance-attribute wrappers at run time only.
2. The default self-attention backend of B is an open question ([instance_b.md](instance_b.md), section 3.2). If
   the answer is a new backend, that backend is an xDiT change. The chunked exact backend is described in
   [future_work/index.md](future_work/index.md), section 2.4.
3. If xDiT cannot start with the vendor packages blocked ([instance_b.md](instance_b.md), section 2.1), B ends
   with ERROR_B. The fix is then an xDiT change. It is not known whether this occurs.

The changes that future versions need are in [future_work/index.md](future_work/index.md).

## 2. xDiT interfaces that the harness uses

The harness depends on these interfaces. A change of one of them needs a change of the harness.

1. `xFuserArgs.add_runner_args(parser)` and its flags
   ([runner.py:148](../../../xfuser/runner.py#L148-L148)).
2. `xFuserModelRunner(args)`, `preprocess_args`, `initialize` and `run`
   ([runner.py:57-95](../../../xfuser/runner.py#L57-L95)).
3. `runner.model`, the runner model object, and its attribute `pipe`.
4. The run loop that calls the pipeline once for each warmup call and once for each iteration
   ([base_model.py:1239-1262](../../../xfuser/model_executor/models/runner_models/base_model.py#L1239-L1262)).
5. The attention backend registry: `AttentionBackendType`, `Spec` with `fallback`, `sparsity` and the vendor
   requirements ([spec.py](../../../xfuser/core/attention/spec.py)).
6. `AttentionBackendType[name.upper()]` as the parser of backend names (`_parse_attention_backend` in
   [base_model.py](../../../xfuser/model_executor/models/runner_models/base_model.py)).
7. The run-argument names in [unsupported.md](unsupported.md) and in [instance_b.md](instance_b.md),
   section 2.2.
