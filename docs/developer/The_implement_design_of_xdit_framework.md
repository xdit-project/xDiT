# Framework design

[中文](The_implement_design_of_xdit_framework_zh.md)

xDiT has two integration layers. Runner adapters define the model's user interface.
Model and pipeline implementations provide distributed execution.
Use [Add a model](adding_models/readme.md) to choose the required integration work.

## Runner adapters

[xFuserModel](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/base_model.py) manages shared configuration validation, initialization, warmup, timing, and output handling. Each adapter declares checkpoint aliases, capabilities, defaults, settings, and loading routes.
It implements `_load_model()` and `_run_pipe()`.

The [FLUX adapter](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py) illustrates two paths:

- Without PipeFusion, it loads a Diffusers `FluxPipeline` with `xFuserFlux1Transformer2DWrapper`, which subclasses the Diffusers transformer and implements sequence partitioning.
- With PipeFusion, it loads `xFuserFluxPipeline`, which uses a registered composition wrapper, `xFuserFluxTransformer2DWrapper`, to divide the transformer into stages.

The [Z-Image adapter](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/z_image.py) preserves the original Diffusers pipeline.
Its transformer handles CFG and sequence partitioning. A runner adapter can thus
use the original pipeline without a new pipeline wrapper. Choose an implementation
that preserves pipeline behavior and supports the required methods.

## Component wrappers

[![Component-wrapper inheritance](../assets/wrapper-components.svg)](../assets/wrapper-components.svg)

This hierarchy describes composition wrappers. It excludes transformer subclasses such as `xFuserFlux1Transformer2DWrapper` and the separate `xFuserModel` runner contract.

| Component | Responsibility | Source |
| --- | --- | --- |
| `xFuserBaseWrapper` | Forward attributes to the wrapped module; check execution conditions | [Base wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/base_wrapper.py) |
| `xFuserPipelineBaseWrapper` | Runtime initialization, component conversion, data-parallel decorators, shared patch/output helpers | [Pipeline base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipelines/base_pipeline.py) |
| `xFuserTransformerBaseWrapper` | Divide block lists between stages and wrap selected submodules | [Transformer base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/base_transformer.py) |
| `xFuserLayerBaseWrapper` | Shared interface for layer wrappers | [Layer base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/base_layer.py) |
| `xFuserSchedulerBaseWrapper` | Shared interface for scheduler wrappers | [Scheduler base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/schedulers/base_scheduler.py) |

Registries map supported Diffusers classes to their wrappers. Register the pipeline, transformer, layer, scheduler, and attention processor that the model needs.
Import their modules to populate the registries. Inheritance alone does not register
a component or provide every parallel method.

Missing scheduler registration logs a warning and retains the original scheduler. That fallback does not establish patch-aware PipeFusion behavior. Unsupported attention processors raise an error. Inspect the [processor registry](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/attention_processor.py) when adding one.

## Distributed state

| Source | Responsibility |
| --- | --- |
| [parallel_state.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/parallel_state.py) | Initialize ranks and process groups; provide group accessors |
| [group_coordinator.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/group_coordinator.py) | Tensor communication operations |
| [runtime_state.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/runtime_state.py) | Input geometry, patch indices, attention settings, and execution metadata |
| [pipefusion/](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipefusion/__init__.py) | Patch layouts, payloads, transport, scheduling, and compilation helpers |

Runtime state is process-local. All ranks must initialize a consistent layout and enter matching collectives.
A Python global does not synchronize ranks. Before partitioning tensors, set input
metadata. Update patch state in schedule order.

## Integration checks

Preserve the model's input/output contract, scheduler behavior, positional embeddings, masks, and callback semantics. Compare the single-device path with the original pipeline. Then check distributed
layouts and supported combinations on suitable hardware. Follow the repository [testing policy](https://github.com/xdit-project/xDiT/blob/main/.github/skills/testing/SKILL.md) for CPU, accelerator, and end-to-end coverage.

<a id="1-framework-design"></a>
<a id="10-design-philosophy"></a>
<a id="11-model-implementation-organizational-structure"></a>
<a id="111-class-architecture"></a>
<a id="222-automatic-wrapping-and-registration-mechanism"></a>
<a id="22-runtime"></a>
