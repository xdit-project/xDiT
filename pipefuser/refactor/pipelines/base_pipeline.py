import torch.nn as nn

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
)
from pipefuser.refactor.base_wrapper import PipeFuserBaseWrapper
from pipefuser.refactor.models.schedulers import *
from pipefuser.refactor.models.transformers import *


logger = init_logger(__name__)

class PipeFuserPipelineBaseWrapper(PipeFuserBaseWrapper):
    engine_config: EngineConfig
    input_config: InputConfig
    
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
    ):
        self.engine_config = engine_config
        self.input_config = engine_config.input_config
        self._check_distributed_env(engine_config.parallel_config)
        # backbone
        transformer = getattr(pipeline, 'transformer', None)
        unet = getattr(pipeline, 'unet', None)
        # vae
        vae = getattr(pipeline, 'vae', None)
        # scheduler
        scheduler = getattr(pipeline, 'scheduler', None)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)
        if transformer is not None:
            pipeline.transformer = \
                self._convert_transformer_backbone(transformer)
            print(47, pipeline.transformer)
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)


        super().__init__(
            module=pipeline,
            parallel_config=engine_config.parallel_config,
            runtime_config=engine_config.runtime_config,
        )
    
    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config

    def _check_distributed_env(self, parallel_config: ParallelConfig):
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            init_distributed_environment()
            initialize_model_parallel(
                data_parallel_degree=parallel_config.dp_degree,
                classifier_free_guidance_degree=
                    parallel_config.cfg_degree,
                sequence_parallel_degree=parallel_config.sp_degree,
                tensor_parallel_degree=parallel_config.tp_degree,
                pipeline_parallel_degree=parallel_config.pp_degree,
            )


    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
        logger.info('Transformer backbone found, paralleling transformer...')
        wrapper = PipeFuserTransformerWrappersRegister.get_wrapper(transformer)
        print(83, wrapper)
        transformer = wrapper(
            transformer=transformer,
            parallel_config=self.engine_config.parallel_config,
            runtime_config=self.engine_config.runtime_config,
            input_config=self.engine_config.input_config,
        )
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
    ):
        logger.info('UNet Backbone found')
        raise NotImplementedError('UNet parallelisation is not supported yet')

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
    ):
        logger.info('Scheduler found, paralleling scheduler...')
        wrapper = PipeFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(
            scheduler=scheduler,
            parallel_config=self.engine_config.parallel_config,
            runtime_config=self.engine_config.runtime_config,
        )
        return scheduler

    def forward(self):
        pass