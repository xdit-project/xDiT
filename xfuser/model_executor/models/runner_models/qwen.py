import torch
import copy
from diffusers import QwenImageEditPipeline, QwenImagePipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.model_executor.models.runner_models.base_model import (
    register_model,
    xFuserModel,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper
from xfuser import xFuserArgs

@register_model("Qwen/Qwen-Image-Edit-2511")
@register_model("Qwen/Qwen-Image-Edit-2509")
@register_model("Qwen/Qwen-Image-Edit")
@register_model("Qwen-Image-Edit-2511")
@register_model("Qwen-Image-Edit-2509")
@register_model("Qwen-Image-Edit")
class xFuserQwenImageEditModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
    )
    default_input_values = DefaultInputValues(
        num_inference_steps=50,
        guidance_scale=4.0,
        negative_prompt=" ",
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image-Edit",
        output_name="qwen_image_edit",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            }
        },
        fp8_gemm_module_list=["transformer.transformer_blocks"],
    )

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        if "2511" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2511"
            self.settings.output_name = "qwen_image_edit_2511"
        elif "2509" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2509"
            self.settings.output_name = "qwen_image_edit_2509"

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserQwenImageTransformerWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = QwenImageEditPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "image": input_args["input_images"][0],
            "prompt": input_args["prompt"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        }
        if "height" in input_args: kwargs["height"] = input_args["height"]
        if "width" in input_args: kwargs["width"] = input_args["width"]

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)


    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Qwen Image Edit model.")

@register_model("Qwen/Qwen-Image-2512")
@register_model("Qwen/Qwen-Image")
@register_model("Qwen-Image-2512")
@register_model("Qwen-Image")
class xFuserQwenImageModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=928,
        width=1664,
        num_inference_steps=50,
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image",
        output_name="qwen_image",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            }
        },
    )

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        if "2512" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-2512"
            self.settings.output_name = "qwen_image_2512"

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserQwenImageTransformerWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "prompt": input_args["prompt"],
            "height": input_args["height"],
            "width": input_args["width"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        }

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)