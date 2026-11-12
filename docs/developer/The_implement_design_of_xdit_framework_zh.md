# 框架设计

[English](The_implement_design_of_xdit_framework.md)

xDiT 的模型集成分为两层：runner 适配器定义用户接口和模型能力，模型及 pipeline 实现负责分布式执行。添加模型时，先阅读[模型集成指南](adding_models/readme.md)。

## Runner 适配器

[xFuserModel](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/base_model.py) 提供配置校验、初始化、预热、计时和输出处理。模型适配器声明 checkpoint 别名、`ModelCapabilities`、`DefaultInputValues`、`ModelSettings` 和 `LoadSupport`，并实现 `_load_model()` 与 `_run_pipe()`。能力声明只是校验依据，实际加载和执行代码仍须实现对应功能。

[FLUX.1-dev 适配器](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py) 有两条执行路径：

- 未启用 PipeFusion 时，使用 Diffusers `FluxPipeline` 和继承原始 transformer 的 `xFuserFlux1Transformer2DWrapper`，在 transformer 中实现序列并行。
- 启用 PipeFusion 时，使用 `xFuserFluxPipeline` 和组合式 `xFuserFluxTransformer2DWrapper`，将 transformer 分配到多个 stage。

[Z-Image 适配器](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/z_image.py) 保留原始 Diffusers pipeline，在 transformer 中实现 CFG 和序列划分。因此，新模型不一定需要编写 pipeline wrapper。应根据模型接口和并行需求选择实现方式。

## 组件 Wrapper

[![组件 Wrapper 继承结构](../assets/wrapper-components.svg)](../assets/wrapper-components.svg)

此图只描述组合式 wrapper，不包含继承 Diffusers transformer 的实现或独立的 runner `xFuserModel`。

| 组件 | 职责 | 源码 |
| --- | --- | --- |
| `xFuserBaseWrapper` | 转发原始模块属性并检查执行条件 | [Base wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/base_wrapper.py) |
| `xFuserPipelineBaseWrapper` | 初始化 runtime、转换组件、处理 DP 装饰器及共享 patch/输出逻辑 | [Pipeline base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipelines/base_pipeline.py) |
| `xFuserTransformerBaseWrapper` | 划分 stage 的 block 列表，包装指定子模块 | [Transformer base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/base_transformer.py) |
| `xFuserLayerBaseWrapper` | Layer wrapper 的共享接口 | [Layer base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/base_layer.py) |
| `xFuserSchedulerBaseWrapper` | Scheduler wrapper 的共享接口 | [Scheduler base](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/schedulers/base_scheduler.py) |

各 registry 将 Diffusers 类型映射到对应 wrapper。新模型需要注册其实际使用的 pipeline、transformer、layer、scheduler 和 attention processor，并确保注册模块被导入。仅继承基类不能自动获得所有并行能力。

未注册的 scheduler 会保留原始实现并输出警告；这不代表它支持 PipeFusion 的 patch 更新。未支持的 attention processor 会报错，添加模型时应检查当前 [processor registry](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/attention_processor.py)。

## 分布式状态

| 源码 | 职责 |
| --- | --- |
| [parallel_state.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/parallel_state.py) | 初始化 rank 和进程组，提供访问接口 |
| [group_coordinator.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/group_coordinator.py) | Tensor 通信操作 |
| [runtime_state.py](https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/runtime_state.py) | 输入尺寸、patch 索引、attention 配置和执行元数据 |
| [pipefusion/](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/pipefusion/__init__.py) | Patch layout、payload、transport、schedule 和编译辅助逻辑 |

Runtime state 是进程内对象。每个 rank 必须初始化一致的布局，并以匹配的顺序调用 collective；读取 Python 全局变量不会同步进程。划分 tensor 前设置输入元数据，并按 schedule 顺序推进 patch 状态。

## 验证

保留模型输入输出、scheduler、位置编码、mask 和 callback 的语义。先比较原始 pipeline 与单设备路径，再在合适的硬件上验证分布式布局及声明支持的组合。测试范围遵循仓库的 [testing policy](https://github.com/xdit-project/xDiT/blob/main/.github/skills/testing/SKILL.md)。

<a id="1"></a>
<a id="10"></a>
<a id="11"></a>
<a id="111"></a>
<a id="222wrap"></a>
<a id="22"></a>
