from abc import ABC, abstractmethod
import torch
from xfuser.config.config import EngineConfig, InputConfig,ParallelConfig
from xfuser.core.distributed import (
    init_distributed_environment,
    get_world_group,
    get_runtime_state,
    init_vae_group,
)
from xfuser.model_executor.pipelines.base_pipeline import xFuserVAEWrapper
from xfuser.core.distributed.parallel_state import initialize_model_parallel
import datetime
from diffusers import FluxPipeline

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


class DiTWorker(WorkerBase):
    """
    A worker class that executes the DiT pipeline on a GPU.
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

    def from_pretrained(
        self,
        PipelineClass, 
        pretrained_model_name_or_path: str, 
        engine_config: EngineConfig,
        **kwargs
    ):
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
    
    def prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        if self.pipe is not None:
            self.pipe.prepare_run(input_config, steps, sync_steps)

    def execute(self, **kwargs):
        if self.pipe is not None:
            output = self.pipe(**kwargs)
            if output is not None:
                return output.images
        return None


class VAEWorker(WorkerBase):
    """
    A worker class that executes the VAE on a GPU.
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
        self.vae = None
    
    def init_worker_distributed_environment(self):
        init_distributed_environment(
            rank=self.rank,
            world_size=self.parallel_config.world_size,
        )
        init_vae_group(self.parallel_config.dit_parallel_size, self.parallel_config.vae_parallel_size, torch.distributed.Backend.NCCL)
        
    def from_pretrained(
        self,
        PipelineClass, 
        pretrained_model_name_or_path: str, 
        engine_config: EngineConfig,
        **kwargs
    ):
        local_rank = get_world_group().local_rank
        for key, value in dict(kwargs).items():
            if isinstance(value, dict) and 'model_class' in value:
                encoder_config = kwargs.pop(key)
                encoder_class = encoder_config.pop('model_class') 
                encoder_instance = encoder_class.from_pretrained(**encoder_config)
                kwargs[key] = encoder_instance
        
        pipe = FluxPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        vae = getattr(pipe, "vae", None).to(f"cuda:{local_rank}")
        
        self.vae = xFuserVAEWrapper(
            vae,
            engine_config=engine_config,
            dit_parallel_config=self.parallel_config,
            use_parallel=engine_config.runtime_config.use_parallel_vae,
            image_processor=pipe.image_processor
        )
        return
    
    def prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        if self.vae is not None:
            return self.vae.execute(output_type=input_config.output_type)
        return None

    def execute(self, **kwargs):
        output_type = kwargs.get('output_type', 'pil')
        if self.vae is not None:
            return self.vae.execute(output_type=output_type)
        return None
