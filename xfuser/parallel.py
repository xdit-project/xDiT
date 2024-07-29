from typing import Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.config import EngineConfig
from xfuser.logger import init_logger
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)



class Parallel:
    def __init__(self, engine_config: EngineConfig):
        init_distributed_environment(random_seed=engine_config.runtime_config.seed)
        initialize_model_parallel(
            data_parallel_degree=engine_config.parallel_config.dp_degree,
            classifier_free_guidance_degree=
                engine_config.parallel_config.cfg_degree,
            sequence_parallel_degree=engine_config.parallel_config.sp_degree,
            ulysses_degree=engine_config.parallel_config.ulysses_degree,
            ring_degree=engine_config.parallel_config.ring_degree,
            tensor_parallel_degree=engine_config.parallel_config.tp_degree,
            pipefusion_parallel_degree=engine_config.parallel_config.pp_degree,
        )

        self.engine_config = engine_config

    def __call__(
        self, 
        pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]],
    ) -> Union[
        xFuserPipelineBaseWrapper,
        Type[xFuserPipelineBaseWrapper]
    ]:
        if isinstance(pipe, type):
            xfuser_pipe_class = \
                xFuserPipelineWrapperRegister.get_class(pipe)
            return xfuser_pipe_class
        elif isinstance(pipe, DiffusionPipeline):
            xfuser_pipe_wrapper = \
                xFuserPipelineWrapperRegister.get_class(pipe)
            return xfuser_pipe_wrapper(
                pipeline=pipe, 
                engine_config=self.engine_config
            )
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")
