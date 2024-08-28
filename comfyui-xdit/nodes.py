import time
import os
import torch
import torch.distributed

from .utils import convert_images_to_tensors
from xfuser.core.distributed import init_distributed_environment
from xfuser import xFuserHunyuanDiTPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
)
from xfuser.config.config import (
    EngineConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    ModelConfig,
    InputConfig,
    RuntimeConfig,
)
from multiprocessing import Process

class XfuserClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"multiline": True}),
            "negative": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRINGC", "STRINGC",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "concat_embeds"

    CATEGORY = "Xfuser"

    def concat_embeds(self, positive, negative):
        return (positive, negative,)

class XfuserPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRINGC",),
                "negative": ("STRINGC",),
                "model_path": (list(["/cfs/dit/HunyuanDiT-v1.2-Diffusers"]), ),
            }
        }

    RETURN_TYPES = ("XFUSER_PIPELINE",)

    FUNCTION = "load_pipeline"

    CATEGORY = "Xfuser"        

    def load_pipeline(self, model_path, positive, negative):
        engine_config, input_config = self.setup_config(model_path, positive, negative)
        local_rank = get_world_group().local_rank
        pipeline = xFuserHunyuanDiTPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
        ).to(f"cuda:{local_rank}")
        pipeline.prepare_run(input_config)

        return (pipeline, )
    
    def setup_config(self, model_path, positive, negative):
        if not torch.distributed.is_initialized():
            # logger.warning(
            #     "Distributed environment is not initialized. " "Initializing..."
            # )
            print("Distributed environment is not initialized. " "Initializing...")
            init_distributed_environment()

        model_config = ModelConfig(
            model=model_path,
            download_dir=None,
            trust_remote_code=False,
        )

        runtime_config = RuntimeConfig(
            warmup_steps=0,
            # use_cuda_graph=self.use_cuda_graph,
            use_parallel_vae=False,
            use_torch_compile=False,
            # use_profiler=self.use_profiler,
        )

        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=1,
                use_cfg_parallel=False,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=1,
                ring_degree=1,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=1,
                split_scheme="row",
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=1,
                num_pipeline_patch=None,
                attn_layer_num_for_pp=None,
            ),
        )

        engine_config = EngineConfig(
            model_config=model_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
        )

        input_config = InputConfig(
            height=1024,
            width=1024,
            use_resolution_binning=False,
            batch_size=1,
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=20,
            seed=0,
            output_type="pil",
        )

        return engine_config, input_config

class XfuserPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("XFUSER_PIPELINE",),
                "positive": ("STRINGC",),
                "negative": ("STRINGC",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "predict"

    CATEGORY = "Xfuser"

    def predict(self, pipeline, positive, negative, width, height, steps, seed):
        output = pipeline(
            height=height,
            width=width,
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=steps,
            output_type="pil",
            generator=torch.Generator(device="cuda").manual_seed(0)
        )
        images = output.images
        return (convert_images_to_tensors(images), )

NODE_CLASS_MAPPINGS = {
    "XfuserClipTextEncode": XfuserClipTextEncode,
    "XfuserPipelineLoader": XfuserPipelineLoader,
    "XfuserPipeline": XfuserPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XfuserClipTextEncode": "XfuserClipTextEncode",
    "XfuserPipelineLoader": "XfuserPipelineLoader",
    "XfuserPipeline": "XfuserPipeline"
}
