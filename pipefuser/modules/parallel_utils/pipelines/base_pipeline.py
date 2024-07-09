from typing import Optional
import torch.nn as nn
from diffusers.models.transformers.transformer_2d import Transformer2DModel

from pipefuser.config.config import EngineConfig, InputConfig, ParallelConfig, RuntimeConfig
from pipefuser.logger import init_logger
from pipefuser.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)
from pipefuser.modules.parallel_utils.models.transformers import \
    PipeFuserTransformerWrappers


logger = init_logger(__name__)

class PipeFuserBasePipeline:
    engine_config: EngineConfig
    parallel_config: ParallelConfig
    runtime_config: RuntimeConfig
    input_config: InputConfig
    
    def __init__(
        self,
        engine_config: EngineConfig,
        **kwargs
    ):
        self.engine_config = engine_config
        self.parallel_config = engine_config.parallel_config
        self.runtime_config = engine_config.runtime_config
        self.input_config = engine_config.input_config
        # backbone
        transformer = kwargs.get('transformer', None)
        unet = kwargs.get('unet', None)
        # vae
        vae = kwargs.get('vae', None)
        # scheduler
        scheduler = kwargs.get('scheduler', None)

        if transformer is not None:
            transformer = self._convert_transformer_backbone(transformer)
            kwargs['transformer'] = transformer
        elif unet is not None:
            unet = self._convert_unet_backbone(unet)
            kwargs['unet'] = unet

        if scheduler is not None:
            scheduler = self._convert_scheduler(scheduler)
            kwargs['scheduler'] = scheduler

        super().__init__(**kwargs)
    
    def set_input_config(self, input_config: InputConfig):
        self.input_config = input_config

    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
        logger.info('Transformer backbone found, paralleling transformer...')
        wrapper = PipeFuserTransformerWrappers.get_wrapper(transformer)
        transformer = wrapper(
            module=transformer,
            parallel_config=self.parallel_config,
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