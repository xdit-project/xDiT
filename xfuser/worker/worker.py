"""A GPU worker class."""
import gc
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from abc import ABC, abstractmethod
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
)
from xfuser.envs import environment_variables
from xfuser.config.config import EngineConfig, InputConfig,ParallelConfig
from xfuser.core.distributed import init_distributed_environment
from xfuser import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserFluxPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
)

class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        input_config: InputConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def from_pretrained(
        self, PipelineClass, engine_config: EngineConfig,**kwargs,
    ):
        raise NotImplementedError


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
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
        time_start = time.time()
        output = self.pipe(**kwargs)
        time_end = time.time()
        if self.pipe.is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(output.images):
                image.save(
                    f"/data/results/stable_diffusion_3_result_{i}.png"
                )
                print(
                    f"image {i} saved to /data/results/stable_diffusion_3_result_{i}.png"
                )
            print(f"time cost: {time_end - time_start}")
        return output
