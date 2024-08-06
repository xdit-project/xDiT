# 框架设计与实现

# 1.框架设计

## 1.0.设计哲学

整个框架的设计目标是，最小化模型改动与代码添加，尽量复用现有diffusers & transformers库中代码，尽量避免多余的代码拷贝，同时尽可能提供灵活的控制从而满足多种并行化逻辑的添加

## 1.1.模型实现组织结构

对模型的修改位于`model_executor/`目录下:

```
xfuser/model_executor
├── base_wrapper.py
├── __init__.py
├── layers
│   ├── attention_processor.py
│   ├── base_layer.py
│   ├── conv.py
│   ├── embeddings.py
│   ├── __init__.py
│   └── register.py
├── models
│   ├── base_model.py
│   ├── __init__.py
│   └── transformers
│       ├── base_transformer.py
│       ├── __init__.py
│       ├── pixart_transformer_2d.py
│       └── register.py
├── pipelines
│   ├── base_pipeline.py
│   ├── __init__.py
│   ├── pipeline_pixart_alpha.py
│   ├── pipeline_pixart_sigma.py
│   └── register.py
└── schedulers
    ├── base_scheduler.py
    ├── __init__.py
    ├── register.py
    └── scheduling_dpmsolver_multistep.py
```

### 1.1.1.类架构

在xDiT框架中，使用wrapper方式修改模型需要的class，从而为原始diffusers中的class添加的并行能力，包括PipeFusion、序列并行，和二者混合方式。并wrapper中设置getattr使得wrapper中的函数可以直接使用原始模型中的属性与方法。通过此种方式可以直接复制原模型的任意代码在wrapper中使用，而不需修改与self有关的访问控制

xDiT中模型相关文件的组织结构如下：

![class_structure.png](../../assets/developer/class_structure.png)

- 所有模型类均会继承于基类`xFuserBaseWrapper`，用以提供基础的getattr和运行时条件检查等特性
- 四个代表不同模型组分的类分别继承于`xFuserBaseWrapper`，其中：
    - `xFuserPipelineBaseWrapper:` diffusion pipeline的基类，提供所有diffusion pipeline并行化均需要用到的特性，如data parallel装饰器，warmup函数prepare_run等。同时在`__init__`中提供pipeline转换流程的逻辑，将对pipeline内部的组成部分如backbone(transformer / unet)，scheduler，vae等进行并行化处理
    - `xFuserModelBaseWrapper:` transformer / unet等模型的基类。__init__中提供对model中部分为了启用pipefusion/sequence parallel并行模式而需要改动的layer进行并行化的逻辑
    - `xFuserLayerBaseWrapper:` 模型最底层的layer继承于该类，如Conv2d，embedding，linear等
    - `xFuserSchedulerBaseWrapper:` scheduler的基类
- 新添加的model只需要继承与对应的基类即可快捷的接入xDiT，启用混合并行

### 2.2.2.自动wrap与注册机制

xDiT使用wrap方式对huggingface diffusers的class做并行化改造，增加一些属性和并行通信的逻辑，并尽可能地复用diffusers原始库中的代码逻辑。从而在保证控制足够灵活的前提下，减少添加模型时的代码量。

此部分以pixart-alpha为例，介绍逐层wrap从而并行化的机制

对于一个新加入的pipeline，其wrapper初始化逻辑直接从父类`xFuserPipelineBaseWrapper`中继承即可，不需要单独写`__init__`函数

```python
# file: xfuser/model_executor/pipelines/pipeline_pixart_alpha.py

@xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline)
class xFuserPixArtAlphaPipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(...

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(...
```

- 以下是继承于`xFuserPipelineBaseWrapper`的__init__的初始化逻辑

```python
class xFuserPipelineBaseWrapper(xFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
    ):
        self.module: DiffusionPipeline
        self._init_runtime_state(pipeline=pipeline, engine_config=engine_config)

        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(transformer)
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)

        super().__init__(module=pipeline)
       
       
    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
				...
	
            logger.info("Transformer backbone found, paralleling transformer...")
            wrapper = **xFuserTransformerWrappersRegister.get_wrapper(transformer)**
            transformer = wrapper(transformer=transformer)
        return transformer
```

pipeline初始化过程中调用了`_convert_transformer_backbone`函数与`_convert_scheduler`对于dit模型的backbone和scheduler分别进行处理。并使用了`xFuserTransformerWrappersRegister`的`get_wrapper`方法获取当前transformer backbone对应的wrapper，从而保证wrap后的并行逻辑与模型本身对应

之所以可以直接通过get_wrapper从register中获取backbone / layer / scheduler与wrapper的对应关系，是因为在wrapper中使用了对应register的装饰器将wrapper注册进了register内部的字典中。例如pixart-alpha模型需要用到的backbone `PixArtTransformer2D`，在框架中引入pixart-alpha模型时需要在`xfuser/model_executor/models/transformers`中实现对应的wrapper类`xFuserPixArtTransformer2DWrapper`，这时仅需要在类定义前加上对应的register并指明原始类即可完成注册，之后该对应关系可以直接通过对应register的get_wrapper方法查到。从而使新加入的pipeline可以直接通过继承父类的初始化函数自动化的找到对应的backbone wrapper，scheduler wrapper等

```python
**@xFuserTransformerWrappersRegister.register(PixArtTransformer2DModel)**
class xFuserPixArtTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
    ):
```

同理，transformer backbone基类`xFuserTransformerBaseWrapper`中，也会在初始化函数`__init__`中使用同样的方式对其内部的layer进行处理，即通过`xFuserLayerWrappersRegister`获取layer对应的wrapper来对layer进行并行化。具体实现代码可参见`xfuser/model_executor/models/transformers/base_transformer.py #35` 与`xfuser/model_executor/models/base_model.py #53`，此处不再赘述

## 2.2.运行时

管理运行时的主要逻辑位于`distributed/`目录下

```
xfuser/distributed
├── group_coordinator.py
├── __init__.py
├── parallel_state.py
└── runtime_state.py
```

其中`parallel_state.py`管理分布式环境(rank, world_size, communication group, etc.)的的初始化和访问

`group_coordinator.py`为communication group的wrapper，对特定的通信模式进行了封装，方便调用

`runtime_state.py`管理了除通信以外的其他运行时元信息，包含对图片patch分块元数据，`runtime_config`，`input_config`等

所有创建的运行时状态与communicator对象均为单例模式的global变量，所有模型均在运行时通过对这些全局变量的查询来获得所需元数据，从而保证全局信息的同步。