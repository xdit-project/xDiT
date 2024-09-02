from xfuser.config import EngineConfig
from xfuser.logger import init_logger
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)


def xdit_parallel(pipe, engine_config: EngineConfig):
    if isinstance(pipe, type):
        xfuser_pipe_class = xFuserPipelineWrapperRegister.get_class(pipe)
        return xfuser_pipe_class
    else:
        xfuser_pipe_wrapper = xFuserPipelineWrapperRegister.get_class(pipe)
        return xfuser_pipe_wrapper(pipeline=pipe, engine_config=engine_config)
