import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)

@register_model("stabilityai/stable-diffusion-3.5-large")
@register_model("stable-diffusion-3.5-large")
@register_model("SD3.5")
class xFuserStableDiffusionModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=True,
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
    )
    settings = ModelSettings(
        model_name="stabilityai/stable-diffusion-3.5-large",
        output_name="stable_diffusion_3_5_large",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder_3": {
                "wrap_attrs": ["encoder.block"],
            },
        },
        fp8_gemm_module_list=["transformer.transformer_blocks", "text_encoder_3.encoder.block"],
    )

    def _load_model(self) -> DiffusionPipeline:
        # SD3's wrapper is composition-style (wraps a transformer instance) and lacks
        # ConfigMixin.load_config, so it cannot be built on meta like flux/z_image. Load real on
        # every rank; on the replicated multi-GPU path the broadcast_fill_replicated is_meta guard
        # keeps these real weights (skips the destructive per-block fill) and the AITER fp8 walk
        # quantizes them CPU->GPU afterwards.
        dtype = torch.float16 if self.config.pipefusion_parallel_degree > 1 else torch.bfloat16
        return xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            engine_config=self.engine_config,
            torch_dtype=dtype,
        )

    def _get_compiled_pipe_components(self):
        return ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3"]

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=self._make_generator(input_args["seed"]),
        )
        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)
