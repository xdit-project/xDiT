import torch
from diffusers import WanImageToVideoPipeline, WanPipeline
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
)
from xfuser.core.distributed import (
    get_world_group,
)
from xfuser.core.utils.runner_utils import log


@register_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
@register_model("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
@register_model("Wan2.2-I2V")
@register_model("Wan2.1-I2V")
class xFuserWanI2VModel(xFuserModel):

    mod_value = 16 # vae_scale_factor_spatial * patch_size[1] = 8
    fps = 16

    def __init__(self, config: dict):
        self.is_wan_2_2 = "2.2" in config.model
        if self.is_wan_2_2:
            self.model_name: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            self.output_name: str = "wan2.2_i2v"
        else:
            self.model_name: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
            self.output_name = "wan2.1_i2v"
        self.model_output_type: str = "video"
        super().__init__(config)

    def _load_model(self):
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )

        if self.is_wan_2_2:
            transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                subfolder="transformer_2",
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
            )
        return pipe


    def _run_pipe(self, input_args: dict):
        output = self.pipe(
            image=input_args["image"],
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output

    def _preprocess_args_images(self, input_args):
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_image", False):
            image = self._resize_and_crop_image(image, width, height, self.mod_value)
        else:
            image = self._resize_image_to_max_area(image, height, width, self.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def validate_args(self, input_args: dict):
        """ Validate input arguments """
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Wan I2V model.")


@register_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
@register_model("Wan-AI/Wan2.1-T2V-14B-Diffusers")
@register_model("Wan2.2-T2V")
@register_model("Wan2.1-T2V")
class xFuserWanT2VModel(xFuserModel):

    mod_value = 8 # vae_scale_factor_spatial * patch_size[1] = 8
    fps = 16


    def __init__(self, config: dict):
        super().__init__(config)
        self.is_wan_2_2 = "2.2" in config.model
        if self.is_wan_2_2:
            self.model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            self.output_name: str = "wan2.2_t2v"
        else:
            self.model_name: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
            self.output_name = "wan2.1_t2v"
        self.model_output_type: str = "video"

    def _load_model(self):
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )

        if self.is_wan_2_2:
            transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                subfolder="transformer_2",
            )
            pipe = WanPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
            )
        else:
            pipe = WanPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
            )

        local_rank = get_world_group().local_rank
        pipe = pipe.to(f"cuda:{local_rank}")
        return pipe

    def _run_pipe(self, input_args: dict):
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output

