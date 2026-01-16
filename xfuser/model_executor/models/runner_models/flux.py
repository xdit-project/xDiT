import torch
from diffusers import FluxPipeline, FluxKontextPipeline, Flux2Pipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFlux1Transformer2DWrapper
from xfuser.model_executor.models.transformers.transformer_flux2 import xFuserFlux2Transformer2DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
)
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
    quantize_linear_layers_to_fp8
)
from xfuser.core.distributed import (
    get_runtime_state,
    get_pipeline_parallel_world_size
)
from xfuser import xFuserFluxPipeline, xFuserArgs

@register_model("black-forest-labs/FLUX.1-dev")
@register_model("FLUX.1-dev")
class xFuserFluxModel(xFuserModel):

    model_name: str = "black-forest-labs/FLUX.1-dev"
    output_name: str = "flux_1_dev"
    model_output_type: str = "image"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        max_sequence_length=512,
    )

    def _load_model(self) -> DiffusionPipeline:
        if self.config.pipefusion_parallel_degree > 1:
            engine_args = xFuserArgs.from_cli_args(self.config) # Models using the xFuser pipeline require these
            engine_config, _ = engine_args.create_config()
            pipe = xFuserFluxPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.float16,
                engine_config=engine_config
            )
        else:
            transformer = xFuserFlux1Transformer2DWrapper.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                subfolder="transformer",
            )
            pipe = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
            )

        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.transformer_blocks, device=device)
            quantize_linear_layers_to_fp8(self.pipe.transformer.single_transformer_blocks, device=device)



    def _run_pipe(self, input_args: dict) -> BaseOutput:
        get_runtime_state().set_input_parameters(
            batch_size=1,
            num_inference_steps=input_args["num_inference_steps"],
            max_condition_sequence_length=input_args["max_sequence_length"],
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output


@register_model("black-forest-labs/FLUX.1-Kontext-dev")
@register_model("FLUX.1-Kontext-dev")
class xFuserFluxKontextModel(xFuserModel):

    mod_value: int = 8 * 2 # TODO: Check if correct
    model_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    output_name: str = "flux_1_kontext_dev"
    model_output_type: str = "image"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=2.5,
        max_sequence_length=256,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserFlux1Transformer2DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = FluxKontextPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> BaseOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            image=input_args["image"],
            max_area=input_args["max_area"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, input_args["width"], input_args["height"], self.mod_value)
            input_args["height"], input_args["width"] = image.height, image.width
        input_args["image"] = image
        input_args["max_area"] = input_args["height"] * input_args["width"]
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Flux.1-Kontext-dev model.")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.transformer_blocks, device=device)
            quantize_linear_layers_to_fp8(self.pipe.transformer.single_transformer_blocks, device=device)



@register_model("black-forest-labs/FLUX.2-dev")
@register_model("FLUX.2-dev")
class xFuserFlux2Model(xFuserModel):

    mod_value: int = 8 * 2 # TODO: Check if correct
    model_name: str = "black-forest-labs/FLUX.2-dev"
    output_name: str = "flux_2_dev"
    model_output_type: str = "image"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=4.0,
        max_sequence_length=256,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserFlux2Transformer2DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = Flux2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        return pipe

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if provided """
        input_args = super()._preprocess_args_images(input_args)
        images = input_args["input_images"]
        if not images:
            images = None
        elif input_args.get("resize_input_images", False):
            images = [self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.mod_value) for image in images]
        input_args["images"] = images
        return input_args

    def _run_pipe(self, input_args: dict) -> BaseOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            image=input_args["images"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        device = self.pipe.device
        if self.config.use_fp8_gemms:
            quantize_linear_layers_to_fp8(self.pipe.transformer.transformer_blocks, device=device)
            quantize_linear_layers_to_fp8(self.pipe.transformer.single_transformer_blocks, device=device)