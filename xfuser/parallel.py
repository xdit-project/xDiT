import os
from typing import Any, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.config.config import InputConfig
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.config import EngineConfig
from xfuser.core.distributed.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    is_dp_last_group,
)
from xfuser.logger import init_logger
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)


class xDiTParallel:
    def __init__(self, pipe, engine_config: EngineConfig, input_config: InputConfig):
        xfuser_pipe_wrapper = xFuserPipelineWrapperRegister.get_class(pipe)
        self.pipe = xfuser_pipe_wrapper(pipeline=pipe, engine_config=engine_config)
        self.config = engine_config
        self.pipe.prepare_run(input_config)

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        self.result = self.pipe(*args, **kwargs)
        return self.result

    def save(self, directory: str, prefix: str):
        dp_rank = get_data_parallel_rank()
        parallel_info = (
            f"dp{self.config.parallel_config.dp_degree}_cfg{self.config.parallel_config.cfg_degree}_"
            f"ulysses{self.config.parallel_config.ulysses_degree}_ring{self.config.parallel_config.ring_degree}_"
            f"pp{self.config.parallel_config.pp_degree}_patch{self.config.parallel_config.pp_config.num_pipeline_patch}"
        )
        prefix = f"{directory}/{prefix}_result_{parallel_info}_dprank{dp_rank}"
        if is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(self.result.images):
                image.save(f"{prefix}_image{i}.png")
                print(f"{prefix}_image{i}.png")
