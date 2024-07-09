from diffusers.models.transformers.transformer_2d import Transformer2DModel

from pipefuser.config.config import ParallelConfig
from pipefuser.logger import init_logger
from pipefuser.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)


logger = init_logger(__name__)

class PipeFuserBasePipeline:
    parallel_config: ParallelConfig
    
    def __init__(
        self,
        parallel_config: ParallelConfig,
        **kwargs
    ):
        self.parallel_config = parallel_config
        # backbone
        transformer = kwargs.get('transformer', None)
        unet = kwargs.get('unet', None)
        # vae
        vae = kwargs.get('vae', None)

        if transformer is not None:
            logger.info('Transformer Backbone found, paralleling transformer')

            if not isinstance(transformer, Transformer2DModel):
                raise ValueError("transformer backbones except "
                                 "Transformer2DModel are not supported yet")
            self._convert_transformer_backbone(transformer)
        elif unet is not None:
            logger.info('UNet Backbone found')
            raise NotImplementedError('UNet parallelisation is not supported yet')

        super().__init__(**kwargs)

    def _convert_transformer_backbone(
        self,
        transformer,
    ):
        if isinstance(transformer, Transformer2DModel):
            logger.info('Converting transformer backbone for '
                        'pipeline parallelism')
        transformer = \
            self._convert_transformer_backbone_for_pipeline(transformer)
        

    def _convert_transformer_backbone_for_pipeline(
        self,
        transformer: Transformer2DModel,
    ):
        if get_pipeline_parallel_world_size() == 1:
            return transformer

        # transformer layer split
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        if self.parallel_config.pp_config.attn_layer_num_for_pp is not None:
            attn_layer_num_for_pp = \
                self.parallel_config.pp_config.attn_layer_num_for_pp
            assert (sum(attn_layer_num_for_pp) ==
                    len(transformer.transformer_blocks)), (
                        "Sum of attn_layer_num_for_pp should be equal to the "
                        "number of transformer blocks") 
            if pp_rank == 0:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    :attn_layer_num_for_pp[0]
                ]
            else:
                transformer.transformer_blocks = transformer.transformer_blocks[
                    sum(attn_layer_num_for_pp[: pp_rank-1]):
                    sum(attn_layer_num_for_pp[: pp_rank])
                ]
        else:
            num_blocks_per_stage = (
                len(transformer.transformer_blocks) + pp_world_size - 1
            ) // pp_world_size
            start_idx = pp_rank * num_blocks_per_stage
            end_idx = min((pp_rank + 1) * num_blocks_per_stage,
                          len(transformer.transformer_blocks))
            transformer.transformer_blocks = transformer.transformer_blocks[
                start_idx:end_idx
            ]
        # position embedding
        if pp_rank != 1:
            transformer.pos_embed = None
        return transformer