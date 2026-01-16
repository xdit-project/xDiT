import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    DefaultInputValues,
)
from xfuser.core.distributed import (
    get_world_group,
)

@register_model("Tongyi-MAI/Z-Image-Turbo")
@register_model("Z-Image-Turbo")
class xFuserZImageTurboModel(xFuserModel):

    model_name: str = "Tongyi-MAI/Z-Image-Turbo"
    output_name: str = "z_image_turbo"
    model_output_type: str = "image"
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> BaseOutput:
        prompt = str(input_args["prompt"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output
