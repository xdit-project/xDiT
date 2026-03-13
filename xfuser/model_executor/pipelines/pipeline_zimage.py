import os
from typing import Dict, Optional, Union

import torch
from diffusers import ZImagePipeline

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import get_runtime_state
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(ZImagePipeline)
class xFuserZImagePipeline(xFuserPipelineBaseWrapper):
    """xFuser wrapper for ZImagePipeline.

    Supported parallelism:
      - Sequence Parallel (SP): handled inside xFuserZImageTransformer2DWrapper
        on the unified image+text stream (the 30 main layers).
      - Data Parallel (DP): handled by enable_data_parallel decorator.

    Not supported (raises RuntimeError if requested):
      - PipeFusion (PP): variable-resolution List[Tensor] inputs are incompatible
        with fixed-patch pipeline stages.
      - CFG Parallel: Z-Image Turbo runs at guidance_scale=0; CFG is optional
        and managed inline in the HF pipeline, not via a separate GPU group.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    def prepare_run(
        self,
        input_config: InputConfig,
        steps: int = 3,
        sync_steps: int = 1,
    ):
        """Warmup run to initialise CUDA graphs / compile caches."""
        prompt = ["" ] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            max_sequence_length=input_config.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    @xFuserPipelineBaseWrapper.check_model_parallel_state(
        cfg_parallel_available=False,
        pipefusion_parallel_available=False,
    )
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    @xFuserPipelineBaseWrapper.enable_data_parallel
    def __call__(
        self,
        prompt=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 9,
        **kwargs,
    ):
        # Z-Image defaults to 1024x1024 when not specified.
        height = height or 1024
        width = width or 1024

        # Determine batch size from prompt or prompt_embeds.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            prompt_embeds = kwargs.get("prompt_embeds", None)
            batch_size = len(prompt_embeds) if prompt_embeds is not None else 1

        # Initialise the runtime state so that SP-aware layers (USP attention)
        # inside xFuserZImageTransformer2DWrapper know the current input shape.
        # split_text_embed_in_sp=False because Z-Image text embeddings are
        # variable-length List[Tensor], not a dense padded tensor to chunk.
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=False,
        )

        return self.module(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )
