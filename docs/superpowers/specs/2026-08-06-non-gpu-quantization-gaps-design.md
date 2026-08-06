# Non-GPU Quantization Gap Closure

## Goal

Close the reviewed correctness bug and implement every loading improvement that
can be validated without accelerator kernels. Hardware-dependent behavior must
remain explicitly unvalidated until the external GPU matrix runs.

## Scope

The work is split into four independently testable stages:

1. correct FP8 text-encoder routing in hybrid FP8/FP4 loads;
2. reject unsupported CUDA FP8 devices before model allocation;
3. stream the NVFP4-owned portion of mixed-precision CUDA loads;
4. add single-rank blockwise meta loading for formats that cannot use native
   framework streaming.

Custom loaders are addressed only when their checkpoint mapping and
configuration-only construction seam are deterministic. Distilled Wan is the
first candidate. SD3.5 composition and CausalWan remain excluded until their
loaders expose the same guarantees.

## Backend Selection

`fp8_backend` continues to represent pure FP8 contracts.
`blockwise_fp8_backend` represents FP8 storage needed by FP4 and hybrid
contracts. Text-encoder planning must select between them by requested format,
using the same ownership rule already applied to transformers.

Backend capability probes must report both API availability and accelerator
eligibility. TorchAO FP8 on CUDA requires compute capability 8.9 or newer.
ROCm eligibility remains controlled by the existing TorchAO/AITER probes.
Unsupported CUDA hardware raises `UnsupportedLoadContract` during preload
selection, before checkpoint discovery or model allocation.

## Partial NVFP4 Streaming

CUDA NVFP4 can stream FP4-owned linear weights through Diffusers while leaving
FP8 precision overrides unconverted. The native configuration derives exact
exclusions from the meta-built transformer:

- declared FP4 targets remain eligible for NVFP4 streaming;
- FP8 prefix and suffix overrides are added to
  `modules_to_not_convert`;
- untargeted linear modules remain excluded;
- post-load FP8 conversion visits only the excluded override paths.

The explicit CUDA hybrid schedule remains rejected. xDiT has no CUDA runtime
wrapper equivalent to the ROCm high/low precision schedule, so loader changes
cannot supply its execution semantics.

## Single-Rank Blockwise Meta Loading

Add a component-level blockwise plan under eager materialization. A global
materialization mode would be incorrect for dual-transformer pipelines where
one component can stream natively and another requires blockwise fallback. The
plan builds one supported transformer on `meta`, reads one declared block at a
time, applies the selected format adapter, and places the converted block on
its execution device. It reuses checkpoint requests, exact layout
reconciliation, persistent-buffer handling, and block target ownership from
the distributed loaders.

The single-rank path performs no collectives. Failure propagation is local, and
the loader must release each source block before reading the next. The expected
peak is one source block plus accumulated quantized state rather than a complete
BF16 transformer.

Automatic selection is limited to standard loaders with:

- a declared config-only construction seam;
- declared block wrapping attributes;
- a checkpoint reader supported by `CheckpointRequest`;
- an adapter whose native streaming plan returned a post-load fallback.

Direct eager loading remains the fallback for unsupported checkpoint layouts or
custom adapters. The descriptor records why blockwise meta loading was not
selected.

## Custom Loader Policy

A custom loader may enter blockwise loading after it supplies:

1. a config-only constructor;
2. a deterministic mapping from checkpoint keys to component keys;
3. block boundaries compatible with the runner strategy;
4. a test fixture proving strict key and shape reconciliation.

Distilled Wan can be adapted if its two external state dictionaries satisfy
these requirements independently. SD3.5 composition and CausalWan retain their
pre-allocation exclusions because their current wrappers do not.

## Error Handling

All new rejection paths use `UnsupportedLoadContract` and run before model
allocation. Native streaming failures caused by missing optional APIs produce a
descriptor with a specific fallback reason. Contract violations, key
mismatches, shape mismatches, and unsupported mixed ownership fail instead of
silently reverting to full-model loading.

## Testing

Development follows red-green cycles. Tests use injected capability probes,
fake Diffusers configuration objects, tiny meta modules, and temporary
safetensors checkpoints.

Required coverage:

- hybrid FSDP selects the blockwise FP8 adapter for text encoders;
- pure FP8 retains the existing adapter;
- unsupported CUDA capability rejects FP8 before allocation;
- capability 8.9 and ROCm routing remain accepted when their APIs exist;
- NVFP4 native config excludes only FP8 overrides and untargeted modules;
- post-load FP8 conversion owns the excluded overrides exactly once;
- single-rank blockwise loading preserves names, shapes, dtypes, ties, and
  persistent buffers;
- source-block lifetime stays bounded in the test loader;
- unsupported custom loaders keep their explicit exclusions.

Dependency-light tests and compilation must pass locally. Accelerator kernels,
peak device memory, numerical quality, and complete model inference remain part
of the external GPU validation handoff.

## Delivery Order

Each stage lands as a separate conventional commit after its focused tests pass.
The existing Black and documentation cleanup stays separate from behavioral
changes. Documentation must distinguish implemented routing from GPU-validated
support.
