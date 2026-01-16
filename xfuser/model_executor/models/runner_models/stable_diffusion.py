import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
)

@register_model("stabilityai/stable-diffusion-3.5-large")
@register_model("stable-diffusion-3.5-large")
@register_model("SD3.5")
class xFuserStableDiffusionModel(xFuserModel):

    model_name: str = "stabilityai/stable-diffusion-3.5-large"
    output_name: str = "stable_diffusion_3_5_large"
    model_output_type: str = "image"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=True,
        use_cfg_parallel=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
    )

    def _load_model(self) -> DiffusionPipeline:
        dtype = torch.float16 if self.config.pipefusion_parallel_degree > 1 else torch.bfloat16
        engine_args = xFuserArgs.from_cli_args(self.config)
        engine_config, _ = engine_args.create_config()
        pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            engine_config=engine_config,
            torch_dtype=dtype,
        )
        return pipe

    def _compile_model(self, input_args: dict) -> None:
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder, mode="default")
        self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2, mode="default")
        self.pipe.text_encoder_3 = torch.compile(self.pipe.text_encoder_3, mode="default")
        self._run_timed_pipe(input_args)

    def _run_pipe(self, input_args: dict) -> BaseOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output
