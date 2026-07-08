from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.config.diffusers import (
    get_minimum_diffusers_version,
    has_valid_diffusers_version,
)
from xfuser.core.distributed import get_runtime_state
from xfuser.core.utils.runner_utils import log
from xfuser.envs import _is_hip
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    _parse_attention_backend,
    register_model,
    xFuserModel,
)

_QUANT_GEMM_MODULES = ["transformer.transformer_blocks"]

# Backends that implement the _varlen_pack mask path and can correctly exclude
# padding key positions.  SDPA_FLASH is excluded because
# aten._scaled_dot_product_flash_attention has no mask parameter.
# Quantised backends (FP8, SAGE, MLA, etc.) are excluded for the same reason.
_KREA2_SUPPORTED_ATTN_BACKENDS = frozenset(
    {
        AttentionBackendType.AITER,
        AttentionBackendType.SDPA,
        AttentionBackendType.SDPA_MATH,
        AttentionBackendType.FLASH,
        AttentionBackendType.FLASH_3,
        AttentionBackendType.FLASH_4,
    }
)


def _patch_text_encoder_linear_for_rocm(text_encoder: "torch.nn.Module") -> None:
    """Compute all text encoder linear layers in float32, store in bfloat16.

    Workaround for ROCm 7.13 bfloat16 GEMMs with large M,N,K shapes that may
    produce NaNs when a split-K kernel is used.
    Computing in float32 avoids the potential NaN issue in bfloat16 split-K path.
    """

    def _make_f32_forward(m: torch.nn.Linear):
        def _f32_forward(x: torch.Tensor) -> torch.Tensor:
            return F.linear(
                x.float(),
                m.weight.float(),
                m.bias.float() if m.bias is not None else None,
            ).to(x.dtype)

        return _f32_forward

    count = 0
    for module in text_encoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = _make_f32_forward(module)
            count += 1

    log(
        f"Patched {count} Linear layers to float32 compute "
        "(ROCm 7.13 bfloat16 split-K NaN fix for Qwen3VL shapes)."
    )


class _Krea2BaseModel(xFuserModel):
    """Shared base for the Krea-2-Raw and Krea-2-Turbo runner models."""

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        pipefusion_parallel_degree=False,
        data_parallel_degree=True,
        use_cfg_parallel=False,
        use_parallel_vae=False,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        fully_shard_degree=True,
        enable_tiling=True,
        enable_slicing=True,
    )

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        backend = _parse_attention_backend(
            config.attention_backend, "attention backend"
        )
        if backend is not None and backend not in _KREA2_SUPPORTED_ATTN_BACKENDS:
            supported = ", ".join(
                sorted(b.name for b in _KREA2_SUPPORTED_ATTN_BACKENDS)
            )
            raise ValueError(
                f"Krea-2 does not support --attention_backend {backend.name}. "
                f"The attention mask requires a backend with varlen support. "
                f"Supported backends: {supported}"
            )

    def _load_model(self) -> DiffusionPipeline:
        if not has_valid_diffusers_version("krea2"):
            raise ImportError(
                f"Krea-2 models require diffusers>={get_minimum_diffusers_version('krea2')}."
            )

        from diffusers.pipelines.krea2 import Krea2Pipeline
        from xfuser.model_executor.models.transformers.transformer_krea2 import (
            xFuserKrea2Transformer2DWrapper,
        )

        log(f"Loading {self.settings.model_name}")
        transformer = xFuserKrea2Transformer2DWrapper.from_pretrained(
            self.settings.model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        # On ROCm 7.13, Qwen3VL bfloat16 GEMM shapes (with max_sequence_length > 448)
        # may produce non-deterministic NaN via a split-K uninitialized-output issue.
        pipeline_kwargs: dict = {}
        if _is_hip():
            from transformers import Qwen3VLModel

            te = Qwen3VLModel.from_pretrained(
                self.settings.model_name,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )
            _patch_text_encoder_linear_for_rocm(te)
            pipeline_kwargs["text_encoder"] = te

        pipe = Krea2Pipeline.from_pretrained(
            self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            **pipeline_kwargs,
        )
        # DiTRuntimeState reads backbone_patch_size from transformer.config.patch_size.
        pipe.transformer.config.patch_size = pipe.patch_size
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        batch_size = self.config.batch_size if self.config.batch_size else 1
        max_seq = input_args.get(
            "max_sequence_length",
            self.default_input_values.max_sequence_length or 512,
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
            negative_prompt=input_args["negative_prompt"],
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
    """Krea-2-Raw: base checkpoint. 52 steps, guidance_scale=3.5."""

    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=52,
        guidance_scale=3.5,
        max_sequence_length=512,
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
    """Krea-2-Turbo: 8-step CFG-free distilled checkpoint."""

    _TURBO_STEPS = 8
    _TURBO_GUIDANCE = 0.0

    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=_TURBO_STEPS,
        guidance_scale=_TURBO_GUIDANCE,
        max_sequence_length=512,
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
