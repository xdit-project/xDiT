from typing import Dict, Type, Union, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from pipefuser.modules.parallel_utils.pipelines.pixart_alpha import (
    PixArtAlphaPipelinePP,
)
from pipefuser.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from pipefuser.config.config import EngineConfig, ParallelConfig
from pipefuser.logger import init_logger
from pipefuser.modules.parallel_utils.pipelines.base_pipeline import PipeFuserBasePipeline
from pipefuser.modules.parallel_utils.models.transformers import (
    PipeFuserTransformerBaseWrapper
)

logger = init_logger(__name__)

#TODO: pipefuser class multiple inheritance
class PipeFuserPipelineClasses:
    _PIPEFUSER_PIPE_MAPPING: Dict[
        Type[DiffusionPipeline], 
        Type[PipeFuserBasePipeline]
    ]

    @classmethod
    def register(cls, origin_pipe_class: Type[DiffusionPipeline]):
        def decorator(pipefusion_pipe_class: Type[PipeFuserBasePipeline]):
            if not issubclass(pipefusion_pipe_class, origin_pipe_class):
                raise ValueError(f"{pipefusion_pipe_class} is not a subclass of"
                                 f" {origin_pipe_class}")
            if not issubclass(pipefusion_pipe_class, PipeFuserBasePipeline):
                raise ValueError(f"{pipefusion_pipe_class} is not a subclass of"
                                 f" PipeFuserBasePipeline")
            cls._PIPEFUSER_PIPE_MAPPING[origin_pipe_class] = \
                pipefusion_pipe_class
            return pipefusion_pipe_class
        return decorator

    @classmethod
    def get_class(
        cls,
        pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]]
    ) -> Type[PipeFuserBasePipeline]:
        if isinstance(pipe, type):
            for (origin_model_class, 
                 pipefuser_model_class) in cls._PIPEFUSER_PIPE_MAPPING.items():
                if issubclass(pipe, origin_model_class):
                    return pipefuser_model_class
            raise ValueError(f"Diffusion Pipeline class {pipe} "
                             f"is not supported by PipeFuser")
        elif isinstance(pipe, DiffusionPipeline):
            for (origin_model_class, 
                 pipefuser_model_class) in cls._PIPEFUSER_PIPE_MAPPING.items():
                if isinstance(pipe, origin_model_class):
                    return pipefuser_model_class
            raise ValueError(f"Diffusion Pipeline class {pipe.__class__} "
                             f"is not supported by PipeFuser")
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")

class PipeFuserPipelineConverters:
    _PIPEFUSER_CONVERTER_MAPPING: Dict[
        Type[DiffusionPipeline],
        Callable[[DiffusionPipeline], PipeFuserBasePipeline]
    ]

    @classmethod
    def register(cls, origin_pipe_class: Type[DiffusionPipeline]):
        def decorator(converter: Callable[[DiffusionPipeline], 
                                          PipeFuserBasePipeline]):
            cls._PIPEFUSER_CONVERTER_MAPPING[origin_pipe_class] = converter
            return converter
        return decorator
    
    @classmethod
    def get_converter(
        cls,
        pipe: DiffusionPipeline
    ) -> Callable[[DiffusionPipeline], PipeFuserBasePipeline]:
        assert isinstance(pipe, DiffusionPipeline), "pipe must be a DiffusionPipeline instance"
        for (origin_model_class, 
             converter) in cls._PIPEFUSER_CONVERTER_MAPPING.items():
            if isinstance(pipe, origin_model_class):
                return converter
        raise ValueError(f"Diffusion Pipeline class {pipe.__class__} "
                         f"is not supported by PipeFuser")




class Parallel:
    def __init__(self, parallel_config: ParallelConfig):
        init_distributed_environment()
        initialize_model_parallel(
            data_parallel_degree=parallel_config.dp_degree,
            classifier_free_guidance_degree=parallel_config.cfg_degree,
            sequence_parallel_degree=parallel_config.sp_degree,
            tensor_parallel_degree=parallel_config.tp_degree,
            pipefusion_parallel_degree=parallel_config.pp_degree,
        )
        self.dp_degree = parallel_config.dp_degree
        self.cfg_degree = parallel_config.cfg_degree
        self.sp_degree = parallel_config.sp_degree
        self.tp_degree = parallel_config.tp_degree
        self.pp_degree = parallel_config.pp_degree

        self.parallel_config = parallel_config


    def __call__(
        self, 
        pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]]
    ):
        pipefuser_pipe_class = PipeFuserPipelineClasses.get_class(pipe)
        if isinstance(pipe, type):
            return pipefuser_pipe_class
        elif isinstance(pipe, DiffusionPipeline):
            return pipefuser_pipe_class()
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")



