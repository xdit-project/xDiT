import torch
from diffusers.pipelines.krea2 import Krea2Pipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.core.distributed import get_runtime_state
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.transformers.transformer_krea2 import (
    xFuserKrea2Transformer2DWrapper,
)

_QUANT_GEMM_MODULES = ["transformer.transformer_blocks"]


class _Krea2BaseModel(xFuserModel):
    """Shared base for the Krea-2-Raw and Krea-2-Turbo runner models."""

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=False,
        data_parallel_degree=True,
        use_cfg_parallel=False,
        use_parallel_vae=False,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        fully_shard_degree=True,
    )

    def _load_model(self) -> DiffusionPipeline:
        log(f"Loading {self.settings.model_name}")
        transformer = xFuserKrea2Transformer2DWrapper.from_pretrained(
            self.settings.model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = Krea2Pipeline.from_pretrained(
            self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # DiTRuntimeState reads backbone_patch_size from transformer.config.patch_size.
        # Krea2Transformer2DModel stores patch_size on the pipeline, not the transformer
        # config, so copy it across before the runtime state is initialised.
        pipe.transformer.config.patch_size = pipe.patch_size
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        batch_size = self.config.batch_size if self.config.batch_size else 1
        max_seq = input_args.get(
            "max_sequence_length",
            self.default_input_values.max_sequence_length or 256,
        )

        get_runtime_state().set_input_parameters(
            height=input_args["height"],
            width=input_args["width"],
            batch_size=batch_size,
            num_inference_steps=input_args["num_inference_steps"],
            max_condition_sequence_length=max_seq,
            split_text_embed_in_sp=False,
        )

        output = self.pipe(
            prompt=input_args["prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            output_type=input_args.get("output_type", "pil"),
            max_sequence_length=max_seq,
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )

        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)


@register_model("krea/krea-2-raw")
@register_model("krea/Krea-2-Raw")
@register_model("Krea-2-Raw")
class xFuserKrea2RawModel(_Krea2BaseModel):
    """Krea-2-Raw: undistilled base checkpoint. 28 steps, guidance_scale=3.5."""

    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=52,
        guidance_scale=3.5,
        max_sequence_length=256,
    )

    settings = ModelSettings(
        model_name="krea/krea-2-raw",
        output_name="krea2_raw",
        model_output_type="image",
        mod_value=16,
        fp8_gemm_module_list=_QUANT_GEMM_MODULES,
        fp4_gemm_module_list=_QUANT_GEMM_MODULES,
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
            "text_encoder": {
                "wrap_attrs": ["model.layers"],
                "offload_policy": "cpu",
            },
        },
    )


@register_model("krea/krea-2-turbo")
@register_model("krea/Krea-2-Turbo")
@register_model("Krea-2-Turbo")
class xFuserKrea2TurboModel(_Krea2BaseModel):
    """Krea-2-Turbo: 8-step CFG-free distilled checkpoint. Supports up to 2048px."""

    _TURBO_STEPS = 8
    _TURBO_GUIDANCE = 0.0

    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=_TURBO_STEPS,
        guidance_scale=_TURBO_GUIDANCE,
        max_sequence_length=256,
    )

    settings = ModelSettings(
        model_name="krea/krea-2-turbo",
        output_name="krea2_turbo",
        model_output_type="image",
        mod_value=16,
        fp8_gemm_module_list=_QUANT_GEMM_MODULES,
        fp4_gemm_module_list=_QUANT_GEMM_MODULES,
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
            "text_encoder": {
                "wrap_attrs": ["model.layers"],
                "offload_policy": "cpu",
            },
        },
    )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        steps = input_args.get("num_inference_steps")
        if steps != self._TURBO_STEPS:
            raise ValueError(
                f"Krea-2-Turbo uses a fixed {self._TURBO_STEPS}-step schedule; "
                f"num_inference_steps must be {self._TURBO_STEPS}, got {steps}."
            )
        guidance = input_args.get("guidance_scale")
        if guidance != self._TURBO_GUIDANCE:
            log(
                f"Krea-2-Turbo is a CFG-free distilled model; "
                f"forcing guidance_scale={self._TURBO_GUIDANCE} (got {guidance})."
            )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        # Guidance is baked into the distilled weights; always run CFG-free.
        return super()._run_pipe({**input_args, "guidance_scale": self._TURBO_GUIDANCE})
