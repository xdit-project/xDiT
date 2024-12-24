from abc import ABC, abstractmethod

from xfuser.core.distributed import (
    get_world_group,
)
from xfuser.config.config import EngineConfig, InputConfig,ParallelConfig
from xfuser.core.distributed import init_distributed_environment

class WorkerBase(ABC):
    def __init__(
        self,
    ) -> None:
        pass
    
    @abstractmethod
    def from_pretrained(
        self, PipelineClass, pretrained_model_name_or_path: str, engine_config: EngineConfig,**kwargs,
    ):
        raise NotImplementedError
    
    @abstractmethod
    def prepare_run(self,input_config: InputConfig,steps: int = 3,sync_steps: int = 1):
        raise NotImplementedError
    @abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError


class Worker(WorkerBase):
    """
    A worker class that executes (a partition of) the model on a GPU.
    """
    parallel_config: ParallelConfig
    def __init__(
        self,
        parallel_config: ParallelConfig,
        rank: int,
    ) -> None:
        WorkerBase.__init__(self)
        self.parallel_config = parallel_config
        self.rank = rank
        self.pipe = None
    
    def init_worker_distributed_environment(self):
        init_distributed_environment(
            rank=self.rank,
            world_size=self.parallel_config.world_size,
        )

    def from_pretrained(self,PipelineClass, pretrained_model_name_or_path: str, engine_config: EngineConfig,**kwargs,):
        local_rank = get_world_group().local_rank
        for key, value in dict(kwargs).items():
            if isinstance(value, dict) and 'model_class' in value:
                encoder_config = kwargs.pop(key)
                encoder_class = encoder_config.pop('model_class') 
                encoder_instance = encoder_class.from_pretrained(**encoder_config)
                kwargs[key] = encoder_instance
        pipe = PipelineClass.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            engine_config=engine_config,
            **kwargs
        ).to(f"cuda:{local_rank}")
        self.pipe = pipe
        return
    
    def prepare_run(self,input_config: InputConfig,steps: int = 3,sync_steps: int = 1):
        self.pipe.prepare_run(input_config,steps,sync_steps)

    def execute(self, **kwargs):
        output = self.pipe(**kwargs)
        if output is not None:
            return output.images # FIXME: can't serialize output, so return images only
        else:
            return None
