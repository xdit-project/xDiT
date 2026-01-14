import torch
from diffusers import HunyuanVideoPipeline
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import xFuserHunyuanVideoTransformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
)
from xfuser.core.distributed import (
    get_world_group,
)

@register_model("tencent/HunyuanVideo")
@register_model("Hunyuanvideo")
class xFuserHunyuanvideoModel(xFuserModel):

    fps = 24
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )

    model_name: str = "tencent/HunyuanVideo"
    output_name: str = "hunyuan_video"
    model_output_type: str = "video"

    def _load_model(self):
        transformer = xFuserHunyuanVideoTransformer3DWrapper.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            revision="refs/pr/18",
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        local_rank = get_world_group().local_rank
        pipe = pipe.to(f"cuda:{local_rank}")
        return pipe

    def _run_pipe(self, input_args: dict):
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
