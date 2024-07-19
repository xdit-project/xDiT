
from typing import Dict, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from pipefuser.logger import init_logger
from pipefuser.pipelines.base_pipeline import PipeFuserPipelineBaseWrapper

logger = init_logger(__name__)

class PipeFuserPipelineWrapperRegister:
    _PIPEFUSER_PIPE_MAPPING: Dict[
        Type[DiffusionPipeline], 
        Type[PipeFuserPipelineBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_pipe_class: Type[DiffusionPipeline]):
        def decorator(pipefusion_pipe_class: Type[PipeFuserPipelineBaseWrapper]):
            if not issubclass(pipefusion_pipe_class, PipeFuserPipelineBaseWrapper):
                raise ValueError(f"{pipefusion_pipe_class} is not a subclass of"
                                 f" PipeFuserPipelineBaseWrapper")
            cls._PIPEFUSER_PIPE_MAPPING[origin_pipe_class] = \
                pipefusion_pipe_class
            return pipefusion_pipe_class
        return decorator

    @classmethod
    def get_class(
        cls,
        pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]]
    ) -> Type[PipeFuserPipelineBaseWrapper]:
        if isinstance(pipe, type):
            candidate = None
            candidate_origin = None
            for (origin_model_class, 
                 pipefuser_model_class) in cls._PIPEFUSER_PIPE_MAPPING.items():
                if issubclass(pipe, origin_model_class):
                    if ((candidate is None and candidate_origin is None) or 
                        issubclass(origin_model_class, candidate_origin)):
                        candidate_origin = origin_model_class
                        candidate = pipefuser_model_class
            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe} "
                                 f"is not supported by PipeFuser")
            else:
                return candidate
        elif isinstance(pipe, DiffusionPipeline):
            candidate = None
            candidate_origin = None
            for (origin_model_class, 
                 pipefuser_model_class) in cls._PIPEFUSER_PIPE_MAPPING.items():
                if isinstance(pipe, origin_model_class):
                    if ((candidate is None and candidate_origin is None) or 
                        issubclass(origin_model_class, candidate_origin)):
                        candidate_origin = origin_model_class
                        candidate = pipefuser_model_class

            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe.__class__} "
                                 f"is not supported by PipeFuser")
            else:
                return candidate
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")