# The implement & design of xdit framework
[Chinese Version](./The_implement_design_of_xdit_framework_zh.md)

# 1. Framework Design

## 1.0. Design Philosophy

The design goal of the entire framework is to minimize model modifications and code additions, maximize reuse of existing code from diffusers & transformers libraries, avoid unnecessary code duplication, and provide flexible control to accommodate various parallelization logics.

## 1.1. Model Implementation Organizational Structure

Modifications to the model are located in the `model_executor/` directory:

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

### 1.1.1. Class Architecture

In the xDiT framework, a wrapper approach is used to modify the required classes for the model, thereby adding parallel capabilities to the original classes in diffusers, including PipeFusion, sequence parallelism, and hybrid methods. The wrapper sets up getattr to allow direct use of attributes and methods from the original model within the wrapper. This approach enables direct replication of any code from the original model for use in the wrapper without modifying self-related access controls.

The organizational structure of model-related files in xDiT is as follows:

![class_structure.png](../../assets/developer/class_structure.png)

- All model classes inherit from the base class `xFuserBaseWrapper`, which provides basic features such as getattr and runtime condition checking.
- Four classes representing different model components inherit from `xFuserBaseWrapper`, including:
    - `xFuserPipelineBaseWrapper`: The base class for diffusion pipelines, providing features necessary for parallelizing all diffusion pipelines, such as data parallel decorators and warmup functions like prepare_run. It also provides pipeline conversion logic in `__init__`, handling parallelization of internal components like backbone (transformer / unet), scheduler, vae, etc.
    - `xFuserModelBaseWrapper`: The base class for transformers / unets and other models. The `__init__` method provides logic for parallelizing layers that need modification to enable pipefusion/sequence parallel modes.
    - `xFuserLayerBaseWrapper`: The base class for the model's lowest-level layers, such as Conv2d, embedding, linear, etc.
    - `xFuserSchedulerBaseWrapper`: The base class for schedulers.
- Newly added models only need to inherit from the corresponding base class to quickly integrate with xDiT and enable hybrid parallelism.

### 2.2.2. Automatic Wrapping and Registration Mechanism

xDiT uses a wrapping approach to parallelize Hugging Face diffusers classes, adding attributes and parallel communication logic while reusing as much code as possible from the original diffusers library. This ensures flexible control while reducing the amount of code needed when adding new models.

This section uses pixart-alpha as an example to introduce the layer-by-layer wrapping mechanism for parallelization.

For a newly added pipeline, its wrapper initialization logic can be directly inherited from the parent class `xFuserPipelineBaseWrapper`, eliminating the need to write a separate `__init__` function.

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

- The following is the initialization logic inherited from `xFuserPipelineBaseWrapper`'s `__init__`:

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
            wrapper = xFuserTransformerWrappersRegister.get_wrapper(transformer)
            transformer = wrapper(transformer=transformer)
        return transformer

```

During pipeline initialization, the `_convert_transformer_backbone` function and `_convert_scheduler` are called to process the backbone and scheduler of the dit model, respectively. The `get_wrapper` method of `xFuserTransformerWrappersRegister` is used to obtain the corresponding wrapper for the current transformer backbone, ensuring that the parallelized logic matches the model itself.

The ability to directly obtain the correspondence between backbone / layer / scheduler and wrapper through get_wrapper from the register is due to the use of corresponding register decorators in the wrapper to register the wrapper into the internal dictionary of the register. For example, the backbone `PixArtTransformer2D` used by the pixart-alpha model requires implementing the corresponding wrapper class `xFuserPixArtTransformer2DWrapper` in `xfuser/model_executor/models/transformers` when introducing the pixart-alpha model to the framework. At this point, simply adding the corresponding register and specifying the original class before the class definition completes the registration. Subsequently, this correspondence can be directly queried using the get_wrapper method of the corresponding register. This allows newly added pipelines to automatically find the corresponding backbone wrapper, scheduler wrapper, etc., through the initialization function inherited from the parent class.

```python
@xFuserTransformerWrappersRegister.register(PixArtTransformer2DModel)
class xFuserPixArtTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
    ):

```

Similarly, in the transformer backbone base class `xFuserTransformerBaseWrapper`, the `__init__` function uses the same method to process its internal layers, i.e., obtaining the corresponding wrapper for the layer through `xFuserLayerWrappersRegister` to parallelize the layer. The specific implementation code can be found in `xfuser/model_executor/models/transformers/base_transformer.py #35` and `xfuser/model_executor/models/base_model.py #53`, and will not be repeated here.

## 2.2. Runtime

The main logic for managing runtime is located in the `distributed/` directory:

```
xfuser/distributed
├── group_coordinator.py
├── __init__.py
├── parallel_state.py
└── runtime_state.py

```

`parallel_state.py` manages the initialization and access of the distributed environment (rank, world_size, communication group, etc.)

`group_coordinator.py` is a wrapper for communication groups, encapsulating specific communication patterns for easy invocation.

`runtime_state.py` manages other runtime metadata besides communication, including metadata for image patch partitioning, `runtime_config`, `input_config`, etc.

All created runtime states and communicator objects are global variables with singleton patterns. All models obtain the required metadata at runtime by querying these global variables, ensuring synchronization of global information.