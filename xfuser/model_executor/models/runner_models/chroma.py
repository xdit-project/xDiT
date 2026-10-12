import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)


@register_model("lodestones/Chroma1-HD")
@register_model("Chroma1-HD")
class xFuserChromaModel(xFuserModel):
    # ChromaPipeline first shipped in 0.34, but the wrapper reuses the FluxAttention
    # processor API that arrived in 0.35.
    min_diffusers_version = "0.35.0"

    # The transformer config states this too. Stated here so a Ulysses degree
    # that cannot work is refused before the download.
    attention_heads = 24
    attention_head_dims = frozenset({128})

    # Chroma's text mask reaches attention as a dense additive bias, which only
    # the SDPA kernels apply; the other backends would drop it.
    supported_attn_backends = frozenset({AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH})
    unsupported_attn_backend_reason = "Chroma's text attention mask is an additive bias that only SDPA applies."

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        meta_text_encoders=("text_encoder",),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        # Ring attention exchanges K/V blocks without slicing the bias for each
        # step, so the masked attention would be wrong.
        ring_degree=False,
        pipefusion_parallel_degree=False,
        use_cfg_parallel=True,
        fully_shard_degree=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=35,
        guidance_scale=5.0,
        max_sequence_length=512,
    )
    settings = ModelSettings(
        model_name="lodestones/Chroma1-HD",
        output_name="chroma1_hd",
        model_output_type="image",
        default_attention_backend="SDPA",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks", "single_transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["encoder.block"],
            },
        },
    )

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        ulysses_degree = config.ulysses_degree or 1
        if self.attention_heads % ulysses_degree != 0:
            raise ValueError(
                f"Chroma1-HD has {self.attention_heads} attention heads, so "
                f"--ulysses_degree must divide {self.attention_heads}, got "
                f"{ulysses_degree}."
            )

    def _load_model(self) -> DiffusionPipeline:
        from xfuser.model_executor.models.transformers.transformer_chroma import (
            xFuserChromaTransformer2DWrapper,
        )
        from xfuser.model_executor.pipelines.pipeline_chroma import (
            xFuserChromaPipeline,
        )

        transformer = self.loader.load_transformer(xFuserChromaTransformer2DWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        return xFuserChromaPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=te_quant,
            **te_kwargs,
        )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            prompt=input_args["prompt"],
            negative_prompt=input_args.get("negative_prompt"),
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            generator=self._make_generator(input_args["seed"]),
        )
        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)
