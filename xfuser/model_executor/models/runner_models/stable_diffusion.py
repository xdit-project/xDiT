import torch
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities
)

@register_model("stabilityai/stable-diffusion-3.5-large")
@register_model("stable-diffusion-3.5-large")
@register_model("SD3.5")
class xFuserZImageTurboModel(xFuserModel):

    model_name: str = "stabilityai/stable-diffusion-3.5-large"
    output_name: str = "stable_diffusion_3_5_large"
    model_output_type: str = "image"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=True,
        use_cfg_parallel=True,
    )

    def _load_model(self):
        engine_args = xFuserArgs.from_cli_args(self.config)
        engine_config, _ = engine_args.create_config()
        pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict):
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output
