import torch
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
    resize_image_to_max_area,
    quantize_linear_layers_to_fp8,
)


@register_model("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
@register_model("Wan2.1-I2V")
class xFuserWan21I2VModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
        use_fsdp=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
    )
    settings = ModelSettings(
        model_name = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        output_name = "wan2.1_i2v",
        model_output_type = "video",
        mod_value = 16, # vae_scale_factor_spatial * patch_size[1] = 8
        fps = 16,
        fsdp_strategy={
            "transformer": {
                "block_attr": "blocks",
                "dtype": torch.bfloat16,
                "children_to_device": [{
                    "submodule_key": "",
                    "exclude_keys": ["blocks"]
                }]
            },
            "text_encoder": {
                "block_attr": "block",
                "shard_submodule_key": "encoder",
                "children_to_device": [
                    {
                        "submodule_key": "encoder",
                        "exclude_keys": ["block"]
                    },
                    {
                        "exclude_keys": ["encoder"]
                    }
                ],
            }
        }
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            image=input_args["image"],
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, used_inputs=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, width, height, self.settings.mod_value)
        else:
            image = resize_image_to_max_area(image, height, width, self.settings.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Wan I2V model.")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.blocks, device=device)


@register_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
@register_model("Wan2.2-I2V")
class xFuserWan22I2VModel(xFuserWan21I2VModel):

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        self.settings.model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_i2v"
        self.settings.fsdp_strategy["transformer_2"] = {
                "block_attr": "blocks",
                "dtype": torch.bfloat16,
                "children_to_device": [{
                    "submodule_key": "",
                    "exclude_keys": ["blocks"]
                }]
        }


    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
        )
        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.blocks, device=device)
            quantize_linear_layers_to_fp8(self.pipe.transformer_2.blocks, device=device)



@register_model("Wan-AI/Wan2.1-T2V-14B-Diffusers")
@register_model("Wan2.1-T2V")
class xFuserWan21T2VModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
    )
    settings = ModelSettings(
        mod_value=8,
        fps=16,
        model_output_type="video",
        model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        output_name="wan2.1_t2v",
        fsdp_strategy={
            "transformer": {
                "block_attr": "blocks",
                "dtype": torch.bfloat16,
                "children_to_device": [{
                    "submodule_key": "",
                    "exclude_keys": ["blocks"]
                }]
            },
            "text_encoder": {
                "block_attr": "block",
                "shard_submodule_key": "encoder",
                "children_to_device": [
                    {
                        "submodule_key": "encoder",
                        "exclude_keys": ["block"]
                    },
                    {
                        "exclude_keys": ["encoder"]
                    }
                ],
            }
        }
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.blocks, device=device)

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            negative_prompt=str(input_args["negative_prompt"]),
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, used_inputs=input_args)


@register_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
@register_model("Wan2.2-T2V")
class xFuserWan22T2VModel(xFuserWan21T2VModel):

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        self.settings.model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_t2v"
        self.settings.fsdp_strategy["transformer_2"] = {
                "block_attr": "blocks",
                "dtype": torch.bfloat16,
                "children_to_device": [{
                    "submodule_key": "",
                    "exclude_keys": ["blocks"]
                }]
        }

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
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
        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.blocks, device=device)
            quantize_linear_layers_to_fp8(self.pipe.transformer_2.blocks, device=device)