# Implement PipeFusion for a model

[中文](adding_model_pipefusion_zh.md)

Use FLUX.1-dev's stage-based path as the reference. PipeFusion divides transformer
blocks between ranks. After synchronous denoising warmup, image patches pass
through these stages. This path uses a dedicated pipeline and a composition
transformer wrapper. See the [PipeFusion guide](../../methods/pipefusion.md) for user flags and combination restrictions.

## Components to adapt

| Component | Current reference | Work for a new model |
| --- | --- | --- |
| Runner adapter | [flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py) | Select stage-aware loading and declare only supported options |
| Pipeline | [pipeline_flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipelines/pipeline_flux.py) | Preserve input preparation; define sync/patch execution, condition transport, scheduler calls, and output conversion |
| Transformer | [transformer_flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_flux.py) | Assign block lists, first/last-stage work, attention processors, and stage payloads |
| Patch runtime | [pipefusion/](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipefusion/__init__.py) | Reuse layouts, payloads, transport, and schedules where their contracts fit |
| Scheduler | [FlowMatch Euler wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/schedulers/scheduling_flow_match_euler_discrete.py) | Update patches without advancing timestep state once per patch |

## Pipeline execution

Before partitioning inputs, FLUX's `__call__()` sets runtime input metadata.
It prepares prompts, packed image latents, position IDs, and guidance embeddings.
These inputs follow the original pipeline's behavior. Guidance embeddings do not
create two CFG prediction branches.

`_pipefusion_async_enabled()` checks whether asynchronous execution is available.
The pipeline calls `_sync_pipeline()` for `warmup_steps`. It then calls
`_async_pipeline()` for any remaining timesteps. If no steps remain, execution
stays synchronous. Pipeline warmup steps belong to one generation. Runner
`warmup_calls` repeats complete generations.

`_async_pipeline()` constructs a `PipeFusionPatchLayout`, `PipeFusionTransport`,
and `PipeFusionImagePatchSchedule`. `PipeFusionPatchForward` binds shared conditions
and per-patch image IDs. Callbacks update patches and assemble the final output.
The schedule advances runtime patch state in order. Base pipeline helpers support
this work; each model still implements its execution loop.

Model-specific requirements include:

- Keep image tokens, position IDs, and patch sizes aligned. FLUX uses token-dimension patches; other pipelines may use spatial layouts.
- Define the condition tensors that travel between stages. Specify whether later patches reuse the first patch's condition. FLUX sets `condition_reuse=True`.
- Preserve supported callback and interruption behavior. The asynchronous runtime rejects `callback_on_step_end`. Document this restriction.
- Reassemble latents and apply model-specific scaling and shifting. Run VAE decoding on the ranks required by the selected path. A fixed output rank alone cannot support parallel VAE.

## Transformer stages and attention

`xFuserFluxTransformer2DWrapper` registers for `FluxTransformer2DModel`. It divides
`transformer_blocks` and `single_transformer_blocks` between stages. The
[transformer base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/base_transformer.py) slices these lists using `attn_layer_num_for_pp`, or divides blocks evenly by default.
It retains checkpoint metadata for stage-aware loading.

The first stage applies image and context input projections. The last stage
applies output normalization and projection. Intermediate stages return image
and condition state. Match these payloads to the pipeline's forward and transport
interfaces. Register new attention processors or layer wrappers where needed.

Patch attention uses current K/V for the active patch and cached K/V for other
patches. The FLUX processor, [USP](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/usp.py), and cache manager implement this behavior.
Preserve cache registration, patch indices, and attention-module handles.
Update each cache in one place, rather than in both the processor and USP.

## Scheduler and loading

The FlowMatch Euler scheduler advances its step index after the last patch or a
full synchronous update. A new scheduler can also require per-patch history.
Preserve its initialization and reset behavior. Without registration, xDiT logs a
warning and retains the original scheduler. This fallback does not establish
PipeFusion compatibility.

Stage-aware loading is part of runner integration. FLUX uses
`plan_pipefusion_components()` and records block ownership when available.
Declare `LoadSupport` for routes the model implements. Composition wrappers and
transformer subclasses have different construction and loading interfaces.

## Validation

Start with synchronous stage execution. Then enable multiple patches after
warmup. Check stage splits, patch order, condition reuse, KV-cache reset, scheduler
state, and final latent assembly. Test each supported combination with USP, VAE,
caching, loading, or compilation.

The [PipeFusion unit tests](https://github.com/xdit-project/xDiT/blob/main/tests/unit/model_executor/pipefusion/test_schedule.py) cover runtime interfaces. Complete inference and
output-quality comparisons require suitable accelerators. During patch execution,
PipeFusion can reuse KV from the previous step. Results can differ from full-image
attention. Performance depends on the workload.

<a id="0-naming-conventions"></a>
<a id="1-pipeline-class"></a>
<a id="11-call-modifications"></a>
<a id="2-transformer-backbone-class"></a>
<a id="21-__init__-modifications"></a>
<a id="22-forward-modifications"></a>
<a id="3-scheduler"></a>
<a id="4-layers"></a>
