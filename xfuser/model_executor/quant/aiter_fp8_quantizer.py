"""Streaming FP8 block-scale quantize-on-load for both the DiT and its text encoder.

The DiT is a diffusers ``ModelMixin`` and the text encoder is a transformers
``PreTrainedModel``; each ``from_pretrained`` calls a different quantizer framework, so this
module holds one thin adapter per framework over a shared swap + quant path:

  * ``AiterFp8BlockScaleQuantizer`` (diffusers ``DiffusersQuantizer``) — the DiT, injected via
    the transformer wrapper's ``from_pretrained(quantization_config=...)``.
  * ``AiterFp8BlockScaleTEQuantizer`` (transformers ``HfQuantizer``) — pipeline sub-models
    (text encoders), routed via diffusers ``PipelineQuantizationConfig(quant_mapping=...)``.

Both swap targeted ``nn.Linear`` leaves for meta ``xFuserFP8BlockScaleLinear`` before load, then
quantize each weight as it streams off disk. So the full bf16 module never materializes on host:
peak ~= one streamed weight + accumulating fp8. This is the load-time complement to the post-load
walk in ``runner_utils.quantize_linear_layers_to_fp8_blockscale``.

The text-encoder path is what added this: on multi-GPU FP8 FSDP the DiT already streams fp8, but
the text encoder (e.g. Mistral3 / Qwen3) loaded full bf16 on every node-local rank, which was the
host-RAM balloon during load. Streaming it to fp8 removes that.

Layout difference between the two framework paths:
  * diffusers (DiT): stores fp8 under ``weight_fp8`` + ``weight_scale`` and nulls the meta
    ``weight`` after quant; forward reads ``weight_fp8``.
  * transformers (TE): the loader requires the target ``weight`` attr to pre-exist and the
    checkpoint ``weight`` key to be an expected param, so fp8 is stored under ``weight`` (+
    ``weight_scale``); forward falls back to ``weight`` when ``weight_fp8`` is absent.
"""

from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

from xfuser.model_executor.layers.fp8_linear import (
    xFuserFP8BlockScaleLinear,
    quantize_weight_to_fp8_blockscale_plain,
    _FP8_BLOCK,
)


class AiterQuantMethod(str, Enum):
    """str-Enum so diffusers' `quant_method.value` access works while dict lookups
    keyed by the plain string still match (str subclass)."""
    AITER_FP8_BLOCKSCALE = "aiter_fp8_blockscale"


AITER_FP8_BLOCKSCALE_QUANT_METHOD = AiterQuantMethod.AITER_FP8_BLOCKSCALE
AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD = "aiter_fp8_blockscale_te"


def _swap_linears_to_fp8(
    module: nn.Module, *, preshuffle: bool = True, add_scale_buffer: bool = False
) -> None:
    """Recursively replace nn.Linear children with meta xFuserFP8BlockScaleLinear placeholders.

    Weights stay on meta (no quant here) — the framework quantizer fills them as they stream.
    A meta bf16 `weight` matching the checkpoint shape is registered so the loader's expected-keys
    diff matches (no missing/unexpected warning) and the streamed weight is routed to quantization;
    it must be bf16 (not fp8) because the transformers loader forces the checkpoint tensor to the
    placeholder's dtype before quantizing.

    add_scale_buffer: register a meta `weight_scale` buffer too (transformers plain layout, where
    the quantize op assigns weight_scale into a pre-existing buffer). The diffusers path leaves it
    off — create_quantized_param attaches weight_fp8 + weight_scale itself.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            fp8_layer = xFuserFP8BlockScaleLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device="meta",
                dtype=child.weight.dtype,
                preshuffle=preshuffle,
            )
            fp8_layer.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(
                        child.out_features, child.in_features,
                        device="meta", dtype=child.weight.dtype,
                    ),
                    requires_grad=False,
                ),
            )
            if add_scale_buffer:
                n_blocks = (child.out_features + _FP8_BLOCK - 1) // _FP8_BLOCK
                k_blocks = (child.in_features + _FP8_BLOCK - 1) // _FP8_BLOCK
                fp8_layer.register_buffer(
                    "weight_scale",
                    torch.empty(n_blocks, k_blocks, device="meta", dtype=torch.float32),
                    persistent=True,
                )
            setattr(module, name, fp8_layer)
        elif next(child.children(), None) is not None:
            _swap_linears_to_fp8(
                child, preshuffle=preshuffle, add_scale_buffer=add_scale_buffer
            )


# --------------------------------------------------------------------------------------
# diffusers path (DiT)
# --------------------------------------------------------------------------------------

from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import (
    QuantizationConfigMixin as DiffusersQuantizationConfigMixin,
)
from diffusers.utils import get_module_from_name as _diffusers_get_module_from_name


class AiterFp8BlockScaleConfig(DiffusersQuantizationConfigMixin):
    """Config for streaming AITER FP8 block-128 w8a8 quantize-on-load of a diffusers model.

    target_modules: transformer-relative sub-module names whose nn.Linear leaves get quantized
    (e.g. ["transformer_blocks"], Wan ["blocks"], VACE ["blocks","vace_blocks"]). Must match the
    post-load fp8_gemm_module_list targets with the pipe prefix stripped.
    """

    def __init__(self, target_modules: Optional[list[str]] = None, **kwargs):
        self.quant_method = AITER_FP8_BLOCKSCALE_QUANT_METHOD
        self.target_modules = list(target_modules or [])

    def to_diff_dict(self) -> dict:
        # Base to_json_string(use_diff=True) calls this; we have no default to diff
        # against, so return the full dict.
        return self.to_dict()


class AiterFp8BlockScaleQuantizer(DiffusersQuantizer):
    """Quantizes targeted nn.Linear weights to FP8 block-scale as they stream off disk."""

    requires_calibration = False
    required_packages = ["aiter"]
    use_keep_in_fp32_modules = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.target_modules = list(quantization_config.target_modules)

    def validate_environment(self, *args, **kwargs):
        from xfuser.core.utils.runner_utils import _use_aiter_fp8_rdna4
        if not _use_aiter_fp8_rdna4():
            raise RuntimeError(
                "AiterFp8BlockScaleQuantizer requires AITER FP8 on RDNA4 (ROCm gfx1200/gfx1201)."
            )

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap every nn.Linear under a target sub-module for a meta xFuserFP8BlockScaleLinear."""
        for target in self.target_modules:
            _swap_linears_to_fp8(model.get_submodule(target))

    def check_if_quantized_param(
        self,
        model,
        param_value: torch.Tensor,
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        if not param_name.endswith(".weight"):
            return False
        module, _ = _diffusers_get_module_from_name(model, param_name)
        return isinstance(module, xFuserFP8BlockScaleLinear)

    def create_quantized_param(
        self,
        model,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        module, _ = _diffusers_get_module_from_name(model, param_name)
        # Quantize on GPU (aiter block-scale + amax need a device); param_value streams in on
        # host. load_and_quantize_weights moves it, produces weight_fp8 + weight_scale.
        compute_device = f"cuda:{torch.cuda.current_device()}"
        module.load_and_quantize_weights(param_value, bias=None, device=compute_device)
        # Drop the meta `weight` placeholder added at swap time (meta tensors can't be
        # moved with .to()); forward uses weight_fp8.
        module.register_parameter("weight", None)
        # Offload: move only the produced fp8 tensors to host. The meta `bias` is streamed
        # to target_device separately by the normal loader (not a quantized param).
        if target_device is not None and torch.device(target_device).type != "cuda":
            module.weight_fp8 = nn.Parameter(
                module.weight_fp8.data.to(target_device), requires_grad=False
            )
            module.weight_scale = module.weight_scale.to(target_device)

    def _process_model_after_weight_loading(self, model, **kwargs):
        # Streaming quant leaves transient staging cached by the allocator (bf16 stage +
        # preshuffle's second fp8 copy in shuffle_weight). Return it to HIP once, so it
        # doesn't sit in reserved VRAM through inference and starve later phases (e.g. VAE
        # decode) of headroom.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True


# --------------------------------------------------------------------------------------
# transformers path (text encoders)
# --------------------------------------------------------------------------------------

from transformers.core_model_loading import ConversionOps
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import (
    get_module_from_name as _transformers_get_module_from_name,
)
from transformers.utils.quantization_config import (
    QuantizationConfigMixin as TransformersQuantizationConfigMixin,
)


class AiterFp8BlockScaleTEConfig(TransformersQuantizationConfigMixin):
    """Config for streaming AITER FP8 block-128 w8a8 quantize-on-load of a transformers model.

    target_modules: model-relative sub-module names whose nn.Linear leaves get quantized
    (e.g. ["model.language_model.layers"]). Must match the fp8_gemm_module_list targets with the
    pipe-component prefix stripped. Targeting decoder LAYERS structurally excludes the tied
    lm_head / embed_tokens.
    """

    def __init__(self, target_modules: Optional[list[str]] = None, **kwargs):
        self.quant_method = AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD
        self.target_modules = list(target_modules or [])


class xFuserFp8BlockScaleQuantizeOp(ConversionOps):
    """Block-quantizes a streamed bf16 weight to plain FP8 block-scale, emitting the fp8
    weight and its scale so the loader assigns both and frees the bf16 tensor."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    @torch.no_grad()
    def convert(self, input_dict: dict, **kwargs) -> dict:
        # AITER fp8 cast needs a GPU (gated on RDNA4 ROCm, so cuda is present); quantize there,
        # then move the fp8 result back to the tensor's intended device to respect placement.
        compute_device = f"cuda:{torch.cuda.current_device()}"
        result: dict = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            base = key[: -len(".weight")] if key.endswith(".weight") else key
            orig_device = tensor.device
            w_q, w_scale = quantize_weight_to_fp8_blockscale_plain(tensor, device=compute_device)
            if orig_device.type != "cuda":
                w_q = w_q.to(orig_device)
                w_scale = w_scale.to(orig_device)
            result[f"{base}.weight"] = nn.Parameter(w_q, requires_grad=False)
            result[f"{base}.weight_scale"] = w_scale
        return result


class AiterFp8BlockScaleTEQuantizer(HfQuantizer):
    """Quantizes targeted nn.Linear weights of a transformers model to FP8 block-scale as they
    stream off disk (quantize-on-load)."""

    requires_calibration = False
    required_packages = ["aiter"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.target_modules = list(quantization_config.target_modules)

    def validate_environment(self, *args, **kwargs):
        from xfuser.core.utils.runner_utils import _use_aiter_fp8_rdna4
        if not _use_aiter_fp8_rdna4():
            raise RuntimeError(
                "AiterFp8BlockScaleTEQuantizer requires AITER FP8 on RDNA4 (ROCm gfx1200/gfx1201)."
            )

    def _process_model_before_weight_loading(self, model, **kwargs):
        for target in self.target_modules:
            _swap_linears_to_fp8(
                model.get_submodule(target), preshuffle=False, add_scale_buffer=True
            )

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        if self.pre_quantized or not param_name.endswith(".weight"):
            return False
        module, _ = _transformers_get_module_from_name(model, param_name)
        return isinstance(module, xFuserFP8BlockScaleLinear)

    def get_quantize_ops(self):
        return xFuserFp8BlockScaleQuantizeOp(self)

    def _process_model_after_weight_loading(self, model, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True


def register_aiter_fp8_quantizers() -> None:
    """Register both quantizers into their respective auto-mappings so
    ``from_pretrained(quantization_config=...)`` routes to us. Idempotent."""
    from diffusers.quantizers.auto import (
        AUTO_QUANTIZER_MAPPING as DIFFUSERS_QUANTIZER_MAPPING,
        AUTO_QUANTIZATION_CONFIG_MAPPING as DIFFUSERS_CONFIG_MAPPING,
    )
    DIFFUSERS_QUANTIZER_MAPPING[AITER_FP8_BLOCKSCALE_QUANT_METHOD] = AiterFp8BlockScaleQuantizer
    DIFFUSERS_CONFIG_MAPPING[AITER_FP8_BLOCKSCALE_QUANT_METHOD] = AiterFp8BlockScaleConfig

    from transformers.quantizers.auto import (
        AUTO_QUANTIZER_MAPPING as TRANSFORMERS_QUANTIZER_MAPPING,
        AUTO_QUANTIZATION_CONFIG_MAPPING as TRANSFORMERS_CONFIG_MAPPING,
    )
    TRANSFORMERS_QUANTIZER_MAPPING[AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD] = AiterFp8BlockScaleTEQuantizer
    TRANSFORMERS_CONFIG_MAPPING[AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD] = AiterFp8BlockScaleTEConfig


register_aiter_fp8_quantizers()
