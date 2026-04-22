import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
)



def _set_effective_heads_for_ulysses(transformer, ulysses_degree: int) -> None:
    """Expose a Ulysses-divisible head count for runtime validation.

    Keep the real model head layout untouched (e.g., n_heads=30) and only set
    config.num_attention_heads used by runtime pre-checks.
    """
    ulysses_degree = int(ulysses_degree or 1)
    if ulysses_degree <= 1:
        return

    real_heads = getattr(transformer.config, "n_heads", None)
    if not isinstance(real_heads, int):
        real_heads = getattr(transformer.config, "num_attention_heads", None)
    if not isinstance(real_heads, int):
        return

    effective_heads = ((real_heads + ulysses_degree - 1) // ulysses_degree) * ulysses_degree
    if effective_heads == real_heads:
        return

    transformer.config.num_attention_heads = effective_heads

@register_model("Tongyi-MAI/Z-Image")
@register_model("Z-Image")
class xFuserZImageModel(xFuserModel):

    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
    )
    capabilities = ModelCapabilities(
        use_cfg_parallel=True
    )
    settings = ModelSettings(
        model_name="Tongyi-MAI/Z-Image",
        output_name="z_image",
        model_output_type="image",
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        _set_effective_heads_for_ulysses(transformer, getattr(self.config, "ulysses_degree", 1))
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = list(input_args["prompt"]) if isinstance(input_args["prompt"], list) else prompt
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)


@register_model("Tongyi-MAI/Z-Image-Turbo")
@register_model("Z-Image-Turbo")
class xFuserZImageTurboModel(xFuserModel):

    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        model_name="Tongyi-MAI/Z-Image-Turbo",
        output_name="z_image_turbo",
        model_output_type="image",
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        _set_effective_heads_for_ulysses(transformer, getattr(self.config, "ulysses_degree", 1))
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = list(input_args["prompt"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)
