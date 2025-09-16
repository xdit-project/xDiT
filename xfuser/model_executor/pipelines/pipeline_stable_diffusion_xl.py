from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import os
from xfuser.model_executor.patch.unet_patch import apply_unet_cfg_parallel_monkey_patch

from diffusers import StableDiffusionXLPipeline
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
)
from xfuser.config import EngineConfig, InputConfig
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

@xFuserPipelineWrapperRegister.register(StableDiffusionXLPipeline)
class xFuserStableDiffusionXLPipeline(xFuserPipelineBaseWrapper):
    def __init__(self, pipeline: StableDiffusionXLPipeline, engine_config: EngineConfig):
        super().__init__(pipeline=pipeline, engine_config=engine_config)
        if get_classifier_free_guidance_world_size() == 2:
            self.module = apply_unet_cfg_parallel_monkey_patch(self.module)
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config)

    @xFuserPipelineBaseWrapper.check_model_parallel_state(
        sequence_parallel_available=False,
        pipefusion_parallel_available=False,
    )
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    @xFuserPipelineBaseWrapper.enable_data_parallel
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        return self.module(*args, **kwargs)
