# Non-GPU Quantization Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix hybrid text-encoder routing and reduce remaining full-precision
load peaks using general, component-level loading plans that can be verified
without accelerator kernels.

**Architecture:** Backend adapters produce exact ownership plans: streamed
linear leaves, residual leaves, or a blockwise fallback. The runner consumes
those plans without backend-specific branches. The existing checkpoint filler
gains a local transport and an optional key-mapping source so eager,
distributed, and remapped loads share one implementation.

**Tech Stack:** Python, PyTorch meta modules, Diffusers/TorchAO configuration
adapters, safetensors, pytest, Gloo/NCCL-compatible loader contracts.

---

## File Map

- `xfuser/model_executor/models/runner_models/base_model.py`: select component
  adapters, consume ownership plans, trigger local blockwise fill.
- `xfuser/model_executor/models/runner_models/loading/fp8_backends.py`: validate
  TorchAO FP8 accelerator eligibility.
- `xfuser/model_executor/models/runner_models/loading/format_backends.py`:
  derive exact native/residual linear ownership.
- `xfuser/model_executor/models/runner_models/loading/meta_load.py`: share block
  filling between local, replicated, and FSDP paths.
- `xfuser/model_executor/models/runner_models/loading/checkpoint.py`: represent
  checkpoint tensor locations whose source and live keys differ.
- `xfuser/model_executor/models/runner_models/wan.py`: route distilled Wan
  through the generic mapped checkpoint source.
- `tests/core/test_replicated_load_decision.py`: hybrid text-encoder regression.
- `tests/core/test_fp8_backend_adapter_contract.py`: CUDA FP8 preflight.
- `tests/core/test_format_backend_adapters.py`: partial NVFP4 ownership.
- `tests/core/test_meta_load_safety.py`: local blockwise fill and bounded source
  lifetime.
- `tests/core/test_checkpoint_request_contract.py`: mapped tensor locations.
- `tests/core/test_roadmap6_model_coverage.py`: distilled Wan declaration.
- `docs/runner/runner.md`: implemented behavior and remaining GPU validation.

### Task 1: Preserve the formatting and documentation cleanup

**Files:**
- Modify: the currently dirty Black-formatted Python files
- Modify: `docs/runner/runner.md`
- Modify: `docs/runner/gpu_validation_handoff.md`

- [ ] **Step 1: Verify the cleanup**

Run:

```bash
base=$(git merge-base HEAD origin/main)
git diff --diff-filter=A --name-only -z "$base"...HEAD -- '*.py' |
  xargs -0 black --check
git diff --check
python3 -m pytest -q \
  tests/core/test_checkpoint_request_contract.py \
  tests/core/test_loading_contracts.py \
  tests/core/test_roadmap3_runner_declarations.py \
  tests/core/test_roadmap6_model_coverage.py \
  tests/core/test_fp8_backend_adapter_contract.py \
  tests/core/test_format_backend_adapters.py \
  tests/core/test_text_encoder_framework_adapter.py \
  tests/core/test_gpu_validation_handoff.py \
  tests/core/test_roadmap2_static_contracts.py
```

Expected: Black and diff checks pass; pytest reports 157 passed and 11 skipped
in the dependency-light environment.

- [ ] **Step 2: Commit only the cleanup**

```bash
git add docs/runner tests tools xfuser/core/utils/checkpoint_io.py \
  xfuser/model_executor/models/runner_models/loading \
  xfuser/model_executor/quant/aiter_fp8_quantizer.py
git commit -m "style: format loader additions"
```

### Task 2: Fix hybrid text-encoder FP8 routing

**Files:**
- Modify: `tests/core/test_replicated_load_decision.py`
- Modify: `xfuser/model_executor/models/runner_models/base_model.py`

- [ ] **Step 1: Write the failing regression**

Add a test beside the existing `_meta_te_kwargs` tests. Construct a runner with
an `FP8_FP4` load contract, `fp8_backend=None`, and a sentinel
`blockwise_fp8_backend`. Assert `prepare_text_encoder_fp8_load` receives the
sentinel:

```python
def test_hybrid_meta_te_uses_blockwise_fp8_backend(monkeypatch):
    sentinel = object()
    runner = SimpleNamespace(
        load_contract=SimpleNamespace(
            requested_format=SimpleNamespace(value="fp8_fp4")
        ),
        fp8_backend=None,
        blockwise_fp8_backend=sentinel,
        _replicated_broadcast_load=lambda: False,
        _memory_efficient_fsdp_load=lambda: True,
        fp8=SimpleNamespace(targets_for=lambda name: ["encoder.block"]),
        settings=SimpleNamespace(
            fp8_text_encoder_module_list=["text_encoder.encoder.block"]
        ),
        _loader=SimpleNamespace(
            meta_te_kwargs=lambda: ({"text_encoder": "meta"}, None),
            build_meta_component=lambda name, fp8=False: object(),
        ),
        _fp8_descriptor_components=set(),
        _fp8_streaming_targets=set(),
    )
    observed = []

    def prepare(adapter, **kwargs):
        observed.append(adapter)
        return SimpleNamespace(
            descriptor=SimpleNamespace(
                materialization_mode="streaming",
                log_message=lambda: "streaming",
            ),
            quantization_config=object(),
        )

    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.fp8_backends."
        "prepare_text_encoder_fp8_load",
        prepare,
    )
    monkeypatch.setattr(base_model, "log", lambda message: None)

    base_model.xFuserModel._meta_te_kwargs(runner)

    assert observed == [sentinel]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_replicated_load_decision.py::test_hybrid_meta_te_uses_blockwise_fp8_backend
```

Expected: FAIL because `_meta_te_kwargs` passes `None` or never calls the
planner.

- [ ] **Step 3: Add one central FP8 adapter selector**

In `xFuserModel`, add:

```python
def _fp8_adapter_for_contract(self):
    if self.load_contract is None:
        return None
    format_value = self.load_contract.requested_format.value
    if format_value == "fp8":
        return self.fp8_backend
    if format_value in {"fp4", "fp8_fp4"}:
        return self.blockwise_fp8_backend
    return None
```

Use this helper in `_meta_te_kwargs`. Update
`_transformer_quantization_adapter` to call it for FP8-owned components instead
of repeating the format switch.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_replicated_load_decision.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py
```

Expected: the new test and existing pure-FP8/hybrid routing tests pass.

- [ ] **Step 5: Commit**

```bash
git add xfuser/model_executor/models/runner_models/base_model.py \
  tests/core/test_replicated_load_decision.py
git commit -m "fix(fp8): route hybrid text encoders"
```

### Task 3: Reject unsupported CUDA FP8 before allocation

**Files:**
- Modify: `tests/core/test_fp8_backend_adapter_contract.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/fp8_backends.py`
- Modify: `docs/runner/runner.md`

- [ ] **Step 1: Write capability tests**

Add tests for a dedicated default accelerator probe:

```python
def test_cuda_fp8_requires_capability_89(modules):
    available, reason = modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (8, 6),
    )
    assert available is False
    assert "8.9" in reason


def test_cuda_fp8_accepts_capability_89(modules):
    assert modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: True,
        hip_probe=lambda: False,
        cuda_capability_probe=lambda: (8, 9),
    ) == (True, None)


def test_rocm_fp8_eligibility_does_not_query_cuda_capability(modules):
    assert modules.backends._probe_torchao_fp8_accelerator(
        cuda_probe=lambda: False,
        hip_probe=lambda: True,
        cuda_capability_probe=lambda: pytest.fail("CUDA probe on ROCm"),
    ) == (True, None)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/core/test_fp8_backend_adapter_contract.py \
  -k "capability_89 or rocm_fp8_eligibility"
```

Expected: FAIL because `_probe_torchao_fp8_accelerator` does not exist.

- [ ] **Step 3: Implement the injectable hardware probe**

Add `_probe_torchao_fp8_accelerator` with injectable `cuda_probe`,
`hip_probe`, and `cuda_capability_probe`. Return `(False, reason)` for
non-accelerator hosts and CUDA capability below `(8, 9)`. Preserve the existing
`torchao_accelerator_probe` injection seam in
`probe_fp8_backend_capabilities`, but allow it to return either `bool` or
`(bool, reason)` through the existing `_result` normalization pattern.

The default path calls `_probe_torchao_fp8_accelerator`; package/API probes run
only after hardware eligibility succeeds.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest -q tests/core/test_fp8_backend_adapter_contract.py
```

Expected: all dependency-light FP8 backend tests pass.

- [ ] **Step 5: Update the hardware matrix and commit**

Change the CUDA-below-8.9 row to state that FP8 is rejected before allocation.

```bash
git add xfuser/model_executor/models/runner_models/loading/fp8_backends.py \
  tests/core/test_fp8_backend_adapter_contract.py docs/runner/runner.md
git commit -m "fix(fp8): preflight CUDA capability"
```

### Task 4: Stream NVFP4 while retaining FP8 overrides

**Files:**
- Modify: `tests/core/test_format_backend_adapters.py`
- Modify: `tests/core/test_fp8_blockwise_hybrid_routing.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/format_backends.py`
- Modify: `xfuser/model_executor/models/runner_models/base_model.py`
- Modify: `docs/runner/runner.md`

- [ ] **Step 1: Replace the old rejection test with an ownership test**

Create a fake model containing targeted blocks, an FP8 prefix override, an FP8
suffix override, and an untargeted projection. Patch the adapter's config
factory and assert:

```python
prepared = b.prepare_native_transformer_format_load(
    adapter,
    component_name="transformer",
    targets=("blocks",),
    stream_quant=True,
    precision_prefixes=("0.attn",),
    precision_suffixes=(".out_proj",),
    hybrid=False,
    model_factory=FakeModel,
)

assert prepared.quantization_config is sentinel
assert captured_exclusions == [
    "blocks.0.attn.q_proj",
    "blocks.1.attn.out_proj",
    "input_proj",
]
assert prepared.streamed_targets == ("blocks.0.mlp", "blocks.1.mlp")
assert prepared.residual_targets == (
    "blocks.0.attn.q_proj",
    "blocks.1.attn.out_proj",
)
```

Keep the explicit hybrid rejection test unchanged.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/core/test_format_backend_adapters.py \
  -k "nvfp4 and override"
```

Expected: FAIL because overrides currently force a complete post-load fallback.

- [ ] **Step 3: Add a general linear ownership result**

Add:

```python
@dataclass(frozen=True)
class LinearOwnership:
    exclusions: tuple[str, ...]
    streamed: tuple[str, ...]
    residual: tuple[str, ...] = ()
```

Implement `derive_linear_ownership(model, targets, min_layer_size=0,
residual_match=None, is_linear=None)`. It validates targets once, enumerates
linear leaves once, and classifies every leaf into exactly one of excluded,
streamed, or residual. Keep `derive_linear_exclusions` as a compatibility
wrapper returning `list(ownership.exclusions)`.

For NVFP4, construct `residual_match` from prefix/suffix patterns relative to
each declared target. Pass `ownership.exclusions + ownership.residual` to
Diffusers, and return the streamed/residual leaves in `PreparedFormatLoad`.

- [ ] **Step 4: Consume exact streamed leaves**

Extend `PreparedFormatLoad`:

```python
@dataclass(frozen=True)
class PreparedFormatLoad:
    descriptor: FormatLoadDescriptor
    quantization_config: object | None = None
    streamed_targets: tuple[str, ...] = ()
    residual_targets: tuple[str, ...] = ()
```

In `_build_transformer`, record `prepared.streamed_targets` when present.
Record the original target roots only for complete native streaming plans.
The existing `_conversion_filter` then skips streamed NVFP4 leaves while
`_setup_nvfp4_gemms` converts residual override leaves to FP8.

- [ ] **Step 5: Verify ownership and routing**

Run:

```bash
python3 -m pytest -q \
  tests/core/test_format_backend_adapters.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py
```

Expected: exact ownership tests pass; hybrid remains rejected; ordinary NVFP4
and INT8 behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add xfuser/model_executor/models/runner_models/loading/format_backends.py \
  xfuser/model_executor/models/runner_models/base_model.py \
  tests/core/test_format_backend_adapters.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py docs/runner/runner.md
git commit -m "feat(fp4): stream around FP8 overrides"
```

### Task 5: Add local blockwise fallback using the existing filler

**Files:**
- Modify: `tests/core/test_meta_load_safety.py`
- Modify: `tests/core/test_fp8_blockwise_hybrid_routing.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/meta_load.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/format_backends.py`
- Modify: `xfuser/model_executor/models/runner_models/base_model.py`

- [ ] **Step 1: Write a pure planning test**

Add `plan_eager_blockwise_fallback` tests proving it selects blockwise loading
only when all requested targets are covered by declared wrap attributes, the
world size is one, the loader is standard, and native loading returned
`post_load`.

```python
plan = b.plan_eager_blockwise_fallback(
    prepared=PreparedFormatLoad(post_load_descriptor),
    targets=("blocks",),
    wrap_attrs=("blocks",),
    world_size=1,
    standard_loader=True,
    offload_requested=False,
)
assert plan.materialization_mode == "blockwise"
```

Add parameterized rejection cases for an uncovered tail target, multiple
ranks, a custom loader, native streaming, and offload.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/core/test_format_backend_adapters.py \
  -k eager_blockwise
```

Expected: FAIL because the planner does not exist.

- [ ] **Step 3: Write a local filler test**

Use a tiny meta module and safetensors checkpoint. Inject a fake quantizer that
records each block and replaces its weight with a smaller sentinel parameter.
Assert blocks are filled and quantized in order, the tail is filled once, no
group method is called, and no source block remains referenced after the next
block:

```python
loader.fill_transformer_local(
    component,
    "transformer",
    {"wrap_attrs": ["blocks"]},
    device="cpu",
    quantize_fn=quantize,
)
assert events == ["fill:0", "quantize:0", "fill:1", "quantize:1", "tail"]
assert component.norm.weight.tolist() == checkpoint_norm
```

- [ ] **Step 4: Verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_meta_load_safety.py -k local
```

Expected: FAIL because `fill_transformer_local` does not exist.

- [ ] **Step 5: Generalize `_TransformerDiskFiller` transport**

Keep one filler. Add local methods that centralize the only transport
differences:

```python
def _source_call(self, fn, *, context):
    if self.group is None:
        return fn()
    return _collective_source_call(
        self.group, self.is_src, fn, context=context
    )

def _broadcast_module(self, module):
    if self.group is not None:
        self._broadcast(module)
```

Apply the same branch to source-map resolution, missing-key checks, layout
agreement, tensor-spec reconciliation, and final broadcasts. Do not duplicate
`fill_block` or `finalize`.

Extract the block loop used by `_fill_transformer_replicated` into
`_fill_transformer_blocks(..., group, quantize_fn)`. Add
`fill_transformer_local` as a caller with `group=None`.

- [ ] **Step 6: Wire the component-level fallback**

In `_build_transformer`, after native planning returns `post_load`, call the
pure planner. If selected, build the transformer on meta with the existing
`build_meta_transformer`, record it in a local-fill set, log a `blockwise`
descriptor, and pass it to the pipeline.

Before `pipe.to(...)` in `_post_load_and_state_initialization`, call
`fill_eager_transformers`. It uses `build_block_quantize_fn`, so MXFP4, FP8,
and INT8 share target ownership with replicated/FSDP loading.

- [ ] **Step 7: Verify GREEN and regressions**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_meta_load_safety.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py \
  tests/core/test_format_backend_adapters.py
```

Expected: local filler and planner tests pass; distributed filler tests remain
green.

- [ ] **Step 8: Commit**

```bash
git add xfuser/model_executor/models/runner_models/loading/meta_load.py \
  xfuser/model_executor/models/runner_models/loading/format_backends.py \
  xfuser/model_executor/models/runner_models/base_model.py \
  tests/core/test_meta_load_safety.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py
git commit -m "feat(loader): add local blockwise fallback"
```

### Task 6: Generalize checkpoint keys and adapt distilled Wan

**Files:**
- Modify: `tests/core/test_checkpoint_request_contract.py`
- Modify: `tests/core/test_meta_load_safety.py`
- Modify: `tests/core/test_roadmap6_model_coverage.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/checkpoint.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/meta_load.py`
- Modify: `xfuser/model_executor/models/runner_models/loading/contracts.py`
- Modify: `xfuser/model_executor/models/runner_models/wan.py`
- Modify: `docs/runner/runner.md`

- [ ] **Step 1: Write mapped-source tests**

Define the desired source API in tests:

```python
source = resolve_mapped_checkpoint(
    path,
    live_key=lambda checkpoint_key: checkpoint_key.replace(
        "diffusion_model.", ""
    ),
)
ref = source.tensors["blocks.0.weight"]
assert ref.path == path.resolve()
assert ref.checkpoint_key == "diffusion_model.blocks.0.weight"
```

Add collision and duplicate-live-key tests that raise before any tensor read.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/core/test_checkpoint_request_contract.py \
  -k mapped
```

Expected: FAIL because mapped tensor references do not exist.

- [ ] **Step 3: Add a general resolved tensor source**

In `checkpoint.py`, add immutable values:

```python
@dataclass(frozen=True)
class CheckpointTensorRef:
    path: Path
    checkpoint_key: str


@dataclass(frozen=True)
class ResolvedCheckpoint:
    tensors: Mapping[str, CheckpointTensorRef]
```

Make standard resolution return live keys mapped to references with identical
checkpoint keys. Add `resolve_mapped_checkpoint(path, live_key)` for explicit
single-file sources. Canonicalize the path and reject live-key collisions.

Change `_TransformerDiskFiller` to consume `ResolvedCheckpoint`; `_fill` opens
`ref.path` and reads `ref.checkpoint_key`. Standard checkpoints retain identity
semantics.

- [ ] **Step 4: Write the distilled Wan strict-remap test**

Create a tiny two-block checkpoint with LightX2V-style keys. Assert the mapped
source fills a meta module strictly and that one unmapped or colliding key
fails. Add a model coverage test requiring distilled Wan to declare
component-level eager blockwise support while retaining FSDP and replicated
exclusions.

- [ ] **Step 5: Verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_meta_load_safety.py \
  tests/core/test_roadmap6_model_coverage.py \
  -k distilled
```

Expected: FAIL because distilled Wan still loads complete base transformers
and then complete remapped state dictionaries.

- [ ] **Step 6: Route distilled Wan through the generic source**

Extend `_build_transformer` and `build_meta_transformer` with an optional
`weight_source`. Configuration still comes from the base model request; disk
fill uses the supplied resolved checkpoint.

Change distilled Wan `_load_model` to call `_build_transformer` for each
transformer with:

```python
weight_source=resolve_mapped_checkpoint(
    self.config.distilled_transformer_path,
    live_key=_remap_lightx2v_to_diffusers,
)
```

Use the second path for `transformer_2`. Keep `LoaderAdapter.DISTILLED_WAN`,
but add a separate `supports_local_blockwise` property so custom loaders do not
gain distributed collectives implicitly.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_checkpoint_request_contract.py \
  tests/core/test_meta_load_safety.py \
  tests/core/test_roadmap6_model_coverage.py
```

Expected: mapped source, strict remapping, standard source, and custom-loader
exclusion tests pass.

- [ ] **Step 8: Commit**

```bash
git add xfuser/model_executor/models/runner_models/loading/checkpoint.py \
  xfuser/model_executor/models/runner_models/loading/meta_load.py \
  xfuser/model_executor/models/runner_models/loading/contracts.py \
  xfuser/model_executor/models/runner_models/wan.py \
  tests/core/test_checkpoint_request_contract.py \
  tests/core/test_meta_load_safety.py \
  tests/core/test_roadmap6_model_coverage.py docs/runner/runner.md
git commit -m "feat(loader): stream remapped checkpoints"
```

### Task 7: Final verification and handoff

**Files:**
- Modify: `docs/runner/runner.md`
- Modify: `docs/runner/gpu_validation_handoff.md`

- [ ] **Step 1: Format changed Python**

```bash
black \
  xfuser/model_executor/models/runner_models/base_model.py \
  xfuser/model_executor/models/runner_models/loading \
  xfuser/model_executor/models/runner_models/wan.py \
  tests/core/test_checkpoint_request_contract.py \
  tests/core/test_format_backend_adapters.py \
  tests/core/test_fp8_backend_adapter_contract.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py \
  tests/core/test_meta_load_safety.py \
  tests/core/test_replicated_load_decision.py \
  tests/core/test_roadmap6_model_coverage.py
```

- [ ] **Step 2: Run dependency-light tests**

```bash
python3 -m pytest -q \
  tests/core/test_checkpoint_request_contract.py \
  tests/core/test_loading_contracts.py \
  tests/core/test_roadmap3_runner_declarations.py \
  tests/core/test_roadmap6_model_coverage.py \
  tests/core/test_fp8_backend_adapter_contract.py \
  tests/core/test_format_backend_adapters.py \
  tests/core/test_text_encoder_framework_adapter.py \
  tests/core/test_gpu_validation_handoff.py \
  tests/core/test_roadmap2_static_contracts.py
```

Expected: all dependency-light tests pass.

- [ ] **Step 3: Run Torch-backed CPU tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/core/test_checkpoint_io.py \
  tests/core/test_fp8_blockwise_hybrid_routing.py \
  tests/core/test_meta_load_safety.py \
  tests/core/test_replicated_load_decision.py \
  tests/layers/test_mxfp4_linear_registration.py
```

Expected: all CPU-capable tests pass; accelerator-only tests skip with an
explicit reason.

- [ ] **Step 4: Compile and inspect**

```bash
python3 -m compileall -q \
  xfuser/model_executor/models/runner_models/loading \
  xfuser/model_executor/models/runner_models/base_model.py \
  xfuser/model_executor/models/runner_models/wan.py \
  tests/core
git diff --check
git status --short
```

Expected: compilation and whitespace checks pass. Only intended changes remain.

- [ ] **Step 5: Record remaining validation**

Keep GPU status as `NOT RUN`. List RDNA4 MXFP4 eager, hybrid FSDP text encoder,
Blackwell partial NVFP4, and distilled Wan as priority smoke cases. Do not
claim kernel correctness, memory savings, or output quality before those cases
produce JSONL results.

- [ ] **Step 6: Commit final documentation**

```bash
git add docs/runner/runner.md docs/runner/gpu_validation_handoff.md
git commit -m "docs(runner): update quantized loading coverage"
```
