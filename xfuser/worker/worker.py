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
    def load_model(
        self, engine_config: EngineConfig,*args, **kwargs
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

    def load_model(self,engine_config: EngineConfig):
        local_rank = get_world_group().local_rank
        pipeline_map = {
            "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
            "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
            "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
            "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
            "FLUX.1-schnell": xFuserFluxPipeline,
        }
        model_name = engine_config.model_config.model.split("/")[-1]
        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")
        if model_name == "stable-diffusion-3-medium-diffusers": # FIXME: hard code 
            text_encoder = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
            pipe = PipelineClass.from_pretrained(
                pretrained_model_name_or_path=engine_config.model_config.model,
                engine_config=engine_config,
                torch_dtype=torch.float16,
                text_encoder_3=text_encoder, # FIXME: hard code 
            ).to(f"cuda:{local_rank}")
        else:
            pipe = PipelineClass.from_pretrained(
                pretrained_model_name_or_path=engine_config.model_config.model,
                engine_config=engine_config,
                torch_dtype=torch.float16,
            ).to(f"cuda:{local_rank}")
        self.pipe = pipe
        return

    def execute(self, input_config: InputConfig):
        self.pipe.prepare_run(input_config)
        output = self.pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
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
        return
