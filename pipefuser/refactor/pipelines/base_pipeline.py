import torch.nn as nn
import torch.distributed as dist

from diffusers import DiffusionPipeline
from pipefuser.refactor.config.config import (
    EngineConfig, 
    InputConfig, 
    ParallelConfig, 
    RuntimeConfig
)
from pipefuser.logger import init_logger
from pipefuser.refactor.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    set_random_seed,
    get_pp_group,
)
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.refactor.schedulers import *
from pipefuser.refactor.models.transformers import *


logger = init_logger(__name__)

class PipeFuserPipelineBaseWrapper(PipeFuserBaseWrapper):
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        self._check_distributed_env(parallel_config, runtime_config)
        # backbone
        transformer = getattr(pipeline, 'transformer', None)
        unet = getattr(pipeline, 'unet', None)
        # vae
        vae = getattr(pipeline, 'vae', None)
        # scheduler
        scheduler = getattr(pipeline, 'scheduler', None)

        if transformer is not None:
            pipeline.transformer = \
                self._convert_transformer_backbone(
                    transformer,
                    parallel_config=parallel_config,
                    runtime_config=runtime_config,
                )
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(
                scheduler,
                parallel_config=parallel_config,
                runtime_config=runtime_config,
            )

        super().__init__(
            module=pipeline,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )

    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config
        if hasattr(self.module, 'transformer') and \
            hasattr(self.module.transformer, 'set_input_config'):
            self.module.transformer.set_input_config(input_config)
        if hasattr(self.module, 'unet') and \
            hasattr(self.module.unet, 'set_input_config'):
            self.module.unet.set_input_config(input_config)
        if hasattr(self.module, 'vae') and \
            hasattr(self.module.vae, 'set_input_config'):
            self.module.vae.set_input_config(input_config)
        if hasattr(self.module, 'scheduler') and \
            hasattr(self.module.scheduler, 'set_input_config'):
            self.module.scheduler.set_input_config(input_config)

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def _check_distributed_env(
        self, 
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        set_random_seed(runtime_config.seed)
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not dist.is_initialized():
                init_distributed_environment()
            initialize_model_parallel(
                data_parallel_degree=parallel_config.dp_degree,
                classifier_free_guidance_degree=
                    parallel_config.cfg_degree,
                sequence_parallel_degree=parallel_config.sp_degree,
                tensor_parallel_degree=parallel_config.tp_degree,
                pipeline_parallel_degree=parallel_config.pp_degree,
            )
            get_pp_group().set_hyper_parameters(
                dtype=runtime_config.dtype,
                num_pipefusion_patches= \
                    parallel_config.pp_config.num_pipeline_patch,
            )


    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info('Transformer backbone found, paralleling transformer...')
        wrapper = PipeFuserTransformerWrappersRegister.get_wrapper(transformer)
        transformer = wrapper(
            transformer=transformer,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info('UNet Backbone found')
        raise NotImplementedError('UNet parallelisation is not supported yet')

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        logger.info('Scheduler found, paralleling scheduler...')
        wrapper = PipeFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(
            scheduler=scheduler,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        return scheduler

    def _check_size_changes(self, height: int, width: int):
        if ((height is not None or width is not None) and 
            (height != self.input_config.height or 
             width != self.input_config.width)):
            self.input_config.height = height or self.input_config.height
            self.input_config.width = width or self.input_config.width
            self.set_input_config(self.input_config)
            get_pp_group().reset_buffer()

    def forward(self):
        pass