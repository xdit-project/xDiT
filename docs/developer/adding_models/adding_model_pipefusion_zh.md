# 为模型实现 PipeFusion

[English](adding_model_pipefusion.md)

以当前 FLUX.1-dev 的 stage 执行路径为参考。PipeFusion 将 transformer blocks 分配到多个 rank，在同步去噪预热后让图像 patch 依次经过这些 stage。此路径使用专门的 pipeline 和组合式 transformer wrapper，与非 PipeFusion 的 FLUX 路径不同。用户参数及组合限制见 [PipeFusion 使用指南](../../methods/pipefusion.md)。

## 需要适配的组件

| 组件 | 当前参考 | 新模型需要实现的内容 |
| --- | --- | --- |
| Runner | [flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py) | 选择 stage-aware 加载方式，声明实际支持的能力 |
| Pipeline | [pipeline_flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipelines/pipeline_flux.py) | 输入处理、同步和 patch 执行、condition 传输、scheduler 调用及输出转换 |
| Transformer | [transformer_flux.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_flux.py) | Block 列表、首尾 stage 操作、attention processor 和 stage payload |
| Patch runtime | [pipefusion/](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipefusion/__init__.py) | 在接口匹配时复用 layout、payload、transport 和 schedule |
| Scheduler | [FlowMatch Euler wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/schedulers/scheduling_flow_match_euler_discrete.py) | 更新 patch，同时避免每个 patch 都推进 timestep |

## Pipeline 执行

FLUX 的 `__call__()` 在切分输入前设置 runtime 元数据，并按原始模型语义准备 prompt、packed image latents、position IDs 和 guidance embeddings。Guidance embeddings 不是 CFG 的条件/无条件分支。

`_pipefusion_async_enabled()` 决定是否进入异步路径。Pipeline 先执行 `warmup_steps` 个 `_sync_pipeline()` 去噪步骤，再用 `_async_pipeline()` 执行剩余 timestep；若没有异步步骤则保持同步。这里的预热属于单次生成内部，runner 的 `warmup_calls` 则重复完整生成。

当前 `_async_pipeline()` 创建 `PipeFusionPatchLayout`、`PipeFusionTransport` 和 `PipeFusionImagePatchSchedule`。通过 `PipeFusionPatchForward` 绑定共享条件及每个 patch 的 image IDs，提供 patch 更新和最终拼接函数，并按调度顺序推进 runtime patch 状态。基类提供共享辅助函数，不包含适用于所有模型的完整同步/异步循环。

适配时需明确以下语义：

- Image tokens、position IDs 和 patch 尺寸必须对应。FLUX 沿 token 维划分，其他模型可能使用空间布局。
- 定义 stage 间传输哪些条件，以及后续 patch 是否可复用第一个 patch 的条件。FLUX 设置 `condition_reuse=True`。
- 明确 callback 和中断行为。当前异步 runtime 拒绝 `callback_on_step_end`，不能宣称支持全部 Diffusers callback。
- 按模型的 scaling/shift 规则重组 latents，并在所选 VAE 路径要求的 rank 上解码；不能一律只在固定 rank 输出，否则会破坏并行 VAE。

## Transformer Stage 与 Attention

`xFuserFluxTransformer2DWrapper` 注册到 `FluxTransformer2DModel`，将 `transformer_blocks` 和 `single_transformer_blocks` 都纳入 stage 划分。[Transformer 基类](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/base_transformer.py) 根据 `attn_layer_num_for_pp` 划分，未指定时使用等 block 数回退方案，并保留用于 stage-aware 加载的 checkpoint 元数据。

仅首 stage 执行 image/context 输入投影，仅末 stage 执行输出归一化和投影。中间 stage 返回 image 和 condition state，payload 必须与 pipeline 的通信约定一致。新模型需要的 attention processor 或 layer wrapper 应显式注册。

Patch attention 使用当前 patch 的 K/V，并缓存其他 patch 的状态。FLUX processor、[USP](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/usp.py) 和 cache manager 共同维护此逻辑。保留 cache 注册、patch 索引和 attention module 引用，避免 processor 与 USP 对同一 cache 重复更新。

## Scheduler 与加载

FlowMatch Euler scheduler 在最后一个 patch 或完整同步更新后推进 step index。其他 scheduler 可能还需维护 patch history 和初始化/重置逻辑。未注册 scheduler 的回退行为不能证明 PipeFusion 兼容性。

FLUX runner 使用 `plan_pipefusion_components()`，并在支持时记录 blockwise ownership。只声明实际实现的 `LoadSupport` 路径；组合式 wrapper 与 transformer 子类的构建/加载接口不同。

## 验证

先验证同步 stage 执行，再验证预热后的多 patch 路径。覆盖显式 stage 划分、patch 顺序、condition 复用、多次调用时的 KV-cache 重置、scheduler 状态、最终 latent 拼接，以及声明支持的 USP、VAE、cache、加载和编译组合。

共享 [PipeFusion 单元测试](https://github.com/xdit-project/xDiT/blob/main/tests/unit/model_executor/pipefusion/test_schedule.py) 验证 runtime 接口；完整推理及输出质量仍需要加速器验证。Patch 执行会复用旧上下文，不能承诺与完整图像 attention 数值完全相同或必然加速。

<a id="0"></a>
<a id="1pipeline-class"></a>
<a id="11__call__"></a>
<a id="2transformer-backbone-class"></a>
<a id="21__init__"></a>
<a id="22forward"></a>
<a id="3scheduler"></a>
<a id="4layers"></a>
