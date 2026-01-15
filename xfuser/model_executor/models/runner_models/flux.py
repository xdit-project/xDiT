import torch
from diffusers import FluxPipeline, FluxKontextPipeline, Flux2Pipeline
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFlux1Transformer2DWrapper
from xfuser.model_executor.models.transformers.transformer_flux2 import xFuserFlux2Transformer2DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
)
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_pipeline_parallel_world_size
)
from xfuser import xFuserFluxPipeline, xFuserArgs

@register_model("black-forest-labs/FLUX.1-dev")
@register_model("FLUX.1-dev")
class xFuserFluxModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=True,
    )

    model_name: str = "black-forest-labs/FLUX.1-dev"
    output_name: str = "flux_1_dev"
    model_output_type: str = "image"

    def _load_model(self):
        if self.config.pipefusion_parallel_degree > 1:
            engine_args = xFuserArgs.from_cli_args(self.config)
            engine_config, _ = engine_args.create_config()
            pipe = xFuserFluxPipeline.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.bfloat16,
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

    def _post_load_and_state_initialization(self, input_args):
        super()._post_load_and_state_initialization(input_args)
        get_runtime_state().set_input_parameters(
            batch_size=1,
            num_inference_steps=self.config.num_inference_steps,
            max_condition_sequence_length=self.config.max_sequence_length,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

    def _run_pipe(self, input_args: dict):
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
class xFuserFluxKontextModel(xFuserFluxModel):

    mod_value: int = 8 * 2 # TODO: Check if correct
    model_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    output_name: str = "flux_1_kontext_dev"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )

    def _load_model(self):
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

    def _run_pipe(self, input_args: dict):
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

    def _preprocess_args_images(self, input_args):
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        if input_args.get("resize_input_images", False):
            image = self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.mod_value)
            input_args["height"], input_args["width"] = image.height, image.width
        input_args["image"] = image
        input_args["max_area"] = input_args["height"] * input_args["width"]
        return input_args

    def validate_args(self, input_args: dict):
        """ Validate input arguments """
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Flux.1-Kontext-dev model.")



@register_model("black-forest-labs/FLUX.2-dev")
@register_model("FLUX.2-dev")
class xFuserFlux2Model(xFuserFluxModel):

    mod_value: int = 8 * 2 # TODO: Check if correct
    model_name: str = "black-forest-labs/FLUX.2-dev"
    output_name: str = "flux_2_dev"
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )

    def _load_model(self):
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

    def _preprocess_args_images(self, input_args):
        """ Preprocess input images if provided """
        input_args = super()._preprocess_args_images(input_args)
        images = input_args["input_images"]
        if not images:
            images = None
        elif input_args.get("resize_input_images", False):
            images = [self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.mod_value) for image in images]
        input_args["images"] = images
        return input_args

    def _run_pipe(self, input_args: dict):
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