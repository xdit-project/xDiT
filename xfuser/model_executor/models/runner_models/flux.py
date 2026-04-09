import torch
from diffusers import FluxPipeline, FluxKontextPipeline, Flux2Pipeline, Flux2KleinPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFlux1Transformer2DWrapper
from xfuser.model_executor.models.transformers.transformer_flux2 import xFuserFlux2Transformer2DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.core.utils.runner_utils import (
    log,
    resize_and_crop_image,
    quantize_linear_layers_to_fp8
)
from xfuser.core.distributed import (
    get_runtime_state,
    get_pipeline_parallel_world_size
)
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser import xFuserFluxPipeline, xFuserArgs


def _setup_parallel_vae(vae) -> None:
    """ Parallalizes the VAE decoder using distvae """
    try:
        from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter
        patched_decoder = DecoderAdapter(
            vae.decoder, vae_group=get_vae_parallel_group().device_group
        ).to(vae.device)
        vae.decoder = patched_decoder
        log(f"Parallel VAE decoder enabled successfully.")
    except ImportError:
        raise ValueError(
            "DistVAE library is missing or does not support DecoderAdapter. "
            "Try installing latest DistVAE from https://github.com/xdit-project/DistVAE."
            )
    except Exception as e:
        raise ValueError(f"Failed to patch VAE decoder. {e}")


@register_model("black-forest-labs/FLUX.1-dev")
@register_model("FLUX.1-dev")
class xFuserFluxModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
        use_parallel_vae=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        max_sequence_length=512,
    )
    settings = ModelSettings(
        model_name="black-forest-labs/FLUX.1-dev",
        output_name="flux_1_dev",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
    )

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        if self.config.use_parallel_vae:
            _setup_parallel_vae(self.pipe.vae)

    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile."""
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead") # Better perf for FLUX.1
        # two steps to warmup the torch compiler
        input_args["num_inference_steps"] = 2
        self._run_timed_pipe(input_args)

    def _load_model(self) -> DiffusionPipeline:
        if self.config.pipefusion_parallel_degree > 1:
            pipe = xFuserFluxPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.float16,
                engine_config=self.engine_config
            )
        else:
            transformer = xFuserFlux1Transformer2DWrapper.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                subfolder="transformer",
            )
            pipe = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
            )

        return pipe


    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        batch_size = self.config.batch_size if self.config.batch_size else 1
        get_runtime_state().set_input_parameters(
            batch_size=batch_size,
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
        images = output.images if output else [] # For legacy pipelines
        return DiffusionOutput(images=images, pipe_args=input_args)


@register_model("black-forest-labs/FLUX.1-Kontext-dev")
@register_model("FLUX.1-Kontext-dev")
class xFuserFluxKontextModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=2.5,
        max_sequence_length=256,
    )
    settings = ModelSettings(
        model_name="black-forest-labs/FLUX.1-Kontext-dev",
        output_name="flux_1_kontext_dev",
        model_output_type="image",
        mod_value=16,
        fp8_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserFlux1Transformer2DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = FluxKontextPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        batch_size = self.config.batch_size if self.config.batch_size else 1
        get_runtime_state().set_input_parameters(
            batch_size=batch_size,
            num_inference_steps=input_args["num_inference_steps"],
            max_condition_sequence_length=input_args["max_sequence_length"],
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
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
        return DiffusionOutput(images=output.images, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, input_args["width"], input_args["height"], self.settings.mod_value)
            input_args["height"], input_args["width"] = image.height, image.width
        input_args["image"] = image
        input_args["max_area"] = input_args["height"] * input_args["width"]
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Flux.1-Kontext-dev model.")



@register_model("black-forest-labs/FLUX.2-dev")
@register_model("FLUX.2-dev")
class xFuserFlux2Model(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        fully_shard_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=4.0,
        max_sequence_length=256,
    )
    settings = ModelSettings(
        model_name="black-forest-labs/FLUX.2-dev",
        output_name="flux_2_dev",
        model_output_type="image",
        mod_value=16,
        fp8_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
        fp4_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks", "single_transformer_blocks"],
            }
        }
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserFlux2Transformer2DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = Flux2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
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
            images = [self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.settings.mod_value) for image in images]
        input_args["images"] = images
        return input_args

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
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
        return DiffusionOutput(images=output.images, pipe_args=input_args)


@register_model("black-forest-labs/FLUX.2-klein-9B")
@register_model("FLUX.2-klein-9B")
class xFuserFlux2Klein9BModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
    )

    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=9,
        guidance_scale=1.0,
    )
    settings = ModelSettings(
        model_name="black-forest-labs/FLUX.2-klein-9B",
        output_name="flux_2_klein_9b",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserFlux2Transformer2DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            image=input_args["images"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if provided """
        input_args = super()._preprocess_args_images(input_args)
        images = input_args["input_images"]
        if not images:
            images = None
        elif input_args.get("resize_input_images", False):
            images = [self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.settings.mod_value) for image in images]
        input_args["images"] = images
        return input_args


@register_model("black-forest-labs/FLUX.2-klein-4B")
@register_model("FLUX.2-klein-4B")
class xFuserFlux2Klein4BModel(xFuserFlux2Klein9BModel):

    settings = ModelSettings(
        model_name="black-forest-labs/FLUX.2-klein-4B",
        output_name="flux_2_klein_4b",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks", "transformer.single_transformer_blocks"],
    )