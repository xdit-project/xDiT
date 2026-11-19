import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser import xFuserArgs, xFuserStableDiffusion3Pipeline
from xfuser.model_executor.cache import (
    DBCachePreset,
    CacheDitAdapterConfig,
    DBCacheSettings,
)
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
    uses_pipeline_stage_meta,
)


@register_model("stabilityai/stable-diffusion-3.5-large")
@register_model("stable-diffusion-3.5-large")
@register_model("SD3.5")
class xFuserStableDiffusionModel(xFuserModel):
    # From the registered checkpoint's transformer config.
    attention_heads = 38

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=True,
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp8_comms=True,
        supports_step_caching=True,
        use_fp8_text_encoder=True,
        use_parallel_vae=True,
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
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        step_cache_config={
            "dbcache": DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("transformer_blocks", "Pattern_1"),),
                ),
                preset=DBCachePreset(
                    Fn_compute_blocks=2,
                    residual_diff_threshold=0.08,
                    scm_policy="fast",
                    enable_encoder_calibrator=False,
                ),
            ),
        },
        fp8_text_encoder_module_list=["text_encoder_3.encoder.block"],
    )

    def _validate_config(self, config: xFuserArgs) -> None:
        super()._validate_config(config)
        if uses_pipeline_stage_meta(config) and config.fully_shard_degree > 1:
            raise ValueError("SD3.5 stage-local replicated loading does not support FSDP.")
        if config.memory_efficient_sharding and config.fully_shard_degree > 1:
            raise ValueError("SD3.5 does not support --memory_efficient_sharding.")
        if uses_pipeline_stage_meta(config) and config.use_fp8_gemms:
            raise ValueError("SD3.5 replicated loading does not support quantized GEMMs.")

    def _load_model(self) -> DiffusionPipeline:
        dtype = torch.float16 if self.config.pipefusion_parallel_degree > 1 else torch.bfloat16
        transformer = None
        pipeline_kwargs = {}
        if uses_pipeline_stage_meta(self.config):
            from diffusers import SD3Transformer2DModel

            transformer, pipeline_kwargs = self.loader.plan_pipefusion_components(
                SD3Transformer2DModel,
                torch_dtype=dtype,
            )
        pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            engine_config=self.engine_config,
            torch_dtype=dtype,
            **pipeline_kwargs,
        )
        if transformer is not None:
            self.loader.mark_pipeline_stage_blockwise(pipe.transformer, transformer)
        return pipe

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
