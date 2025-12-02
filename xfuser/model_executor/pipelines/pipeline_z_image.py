import os
import torch
from typing import Dict, Optional, Union, List, Callable, Any
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from xfuser.config import EngineConfig
from .register import xFuserPipelineWrapperRegister
from diffusers import ZImagePipeline

@xFuserPipelineWrapperRegister.register(ZImagePipeline)
class xFuserZImagePipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict={},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = ZImagePipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        return self.module.__call__(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )