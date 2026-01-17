import functools
import torch
import torch.nn.functional as F
from enum import Enum
from xfuser.envs import PACKAGES_CHECKER, environment_variables

ATTENTION_FUNCTION_REGISTRY = {}
AITER_FP8_STATIC_SCALE_WITH_DESCALE = environment_variables["AITER_FP8_STATIC_SCALE_WITH_DESCALE"]()
AITER_FP8_STATIC_SCALE_NO_DESCALE = 1.0 # This value should be 1.0 when descale vectors are not used.

aten = torch.ops.aten
env_info = PACKAGES_CHECKER.get_packages_info()
if env_info["has_aiter"]:
    import aiter
    from aiter import flash_attn_func as flash_attn_func_aiter
    import inspect
    try:
        AITER_HAS_ROUND_MODE = inspect.signature(flash_attn_func_aiter).parameters.get("how_v3_bf16_cvt") is not None
    except (AttributeError, TypeError):
        AITER_HAS_ROUND_MODE = False
    if AITER_HAS_ROUND_MODE:
        import os
        HOW_V3_BF16_CVT = int(os.environ.get("HOW_V3_BF16_CVT", "2"))

    try:
        AITER_FP8_HAS_DESCALE = inspect.signature(aiter.flash_attn_fp8_pertensor_func).parameters.get("q_descale") is not None
    except (AttributeError, TypeError):
        AITER_FP8_HAS_DESCALE = False

if env_info["has_flash_attn"]:
    from flash_attn import flash_attn_func as flash_attn_func_2
if env_info["has_flash_attn_3"]:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
if env_info["has_flash_attn_4"]:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_4
if env_info["has_npu_flash_attn"]:
    import torch_npu

class AttentionBackendType(Enum):
    SDPA = "SDPA"
    SDPA_MATH = "SDPA with Math backend"
    SDPA_EFFICIENT = "SDPA with memory-efficient backend"
    SDPA_FLASH = "SDPA with FLASH backend"
    FLASH = "Flash Attention V2"
    CUDNN =  "cuDNN"
    FLASH_3 = "Flash Attention V3"
    FLASH_3_FP8 = "Flash Attention v3 FP8"
    FLASH_4 = "Flash Attention V4"
    AITER = "AITER"
    AITER_FP8 = "AITER FP8"
    NPU = "NPU"

def register_attention_function(backend_type):
    """
    Decorator to register attention functions with their corresponding backend type.
    """
    def decorator(func):
        ATTENTION_FUNCTION_REGISTRY[backend_type] = func
        return func
    return decorator

@register_attention_function(AttentionBackendType.SDPA)
def _sdpa_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs attention through PyTorch's scaled_dot_product_attention.
    Allows Pytorch to decide which SDPA backend to use.
    """
    output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )
    return output, None

@register_attention_function(AttentionBackendType.SDPA_FLASH)
def _sdpa_flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs flash attention using Pytorch's internal implementation.
    """
    output, softmax_lse, *rest = aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    return output, softmax_lse

@register_attention_function(AttentionBackendType.SDPA_MATH)
def _sdpa_math_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs attention using Pytorch's internal math implementation.
    """
    output, softmax_lse = aten._scaled_dot_product_attention_math(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    return output, softmax_lse

@register_attention_function(AttentionBackendType.SDPA_EFFICIENT)
def _sdpa_efficient_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs attention using Pytorch's internal memory-efficient implementation.
    """
    output, softmax_lse, *rest = aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    return output, softmax_lse

@register_attention_function(AttentionBackendType.CUDNN)
def _cudnn_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through cuDNN backend
    """
    output, softmax_lse, *rest = aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    softmax_lse = softmax_lse.squeeze(-1)
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH_3)
def _flash_attn_3_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn V3
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = flash_attn_func_3(
        query,
        key,
        value,
        causal=is_causal,
        return_attn_probs=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@functools.lru_cache()
def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except TypeError:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax

def per_tensor_quant(
    x, scale=None, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fn, dtypeMax=None
):
    x = x.to(torch.float32)
    if scale is None:
        if dtypeMax is None:
            dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x).max() / dtypeMax
    y = x / scale
    return y.to(quant_dtype), scale.expand(*x.shape[:2]).to(scale_dtype)

@register_attention_function(AttentionBackendType.FLASH_3_FP8)
def _flash_attn_3_fp8_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn V3
    """
    # quantize
    query, scale_query = per_tensor_quant(query)
    key, scale_key = per_tensor_quant(key)
    value, scale_value = per_tensor_quant(value)
    # run
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = flash_attn_func_3(
        query,
        key,
        value,
        causal=is_causal,
        return_attn_probs=True,
        q_descale=scale_query,
        k_descale=scale_key,
        v_descale=scale_value,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH_4)
@torch.compiler.disable # Disabling compile, as it is not currently supported with FAv4
def _flash_attn_4_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn V4
    """

    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = flash_attn_func_4(
        query,
        key,
        value,
        causal=is_causal,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER_FP8)
def _aiter_fp8_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    softmax_lse = None
    quant_dtype = aiter.dtypes.fp8
    if AITER_FP8_HAS_DESCALE:
        # Skip calling .max() every attention call, but take height for relatively large values,
        # to avoid overflows and possible NaNs in attention computation.
        scale=torch.tensor(AITER_FP8_STATIC_SCALE_WITH_DESCALE, dtype=torch.float32, device=query.device)
        # TODO: Is it possible to improve dynamic scaling perf?
    else:
        # Use static scale of 1.0, since descale is not available.
        scale = torch.tensor(AITER_FP8_STATIC_SCALE_NO_DESCALE, dtype=torch.float32, device=query.device)
    quant_q, q_descale = aiter.per_tensor_quant(query,
                                                scale=scale,
                                                quant_dtype=quant_dtype)
    quant_k, k_descale = aiter.per_tensor_quant(key,
                                                scale=scale,
                                                quant_dtype=quant_dtype)
    quant_v, v_descale = aiter.per_tensor_quant(value,
                                                scale=scale,
                                                quant_dtype=quant_dtype)

    attn_kwargs = {}
    if AITER_FP8_HAS_DESCALE:
        attn_kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }
    torch._dynamo.graph_break()
    output = aiter.flash_attn_fp8_pertensor_func(
        quant_q, quant_k, quant_v,
        causal=is_causal,
        **attn_kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER)
def _aiter_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    attn_kwargs = {
        "dropout_p": dropout_p,
        "causal": is_causal,
        "return_attn_probs": False,
        "return_lse": True,
    }
    if AITER_HAS_ROUND_MODE:
        attn_kwargs["how_v3_bf16_cvt"] = HOW_V3_BF16_CVT
    output, softmax_lse = flash_attn_func_aiter(
        query,
        key,
        value,
        **attn_kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH)
def _flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn
    """
    query = torch.permute(query, [0, 2, 1, 3])
    key = torch.permute(key, [0, 2, 1, 3])
    value = torch.permute(value, [0, 2, 1, 3])
    output, softmax_lse, S_mask = flash_attn_func_2(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_attn_probs=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.NPU)
def npu_flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor transpose and
    then calls attention through npu_fused_infer_attention_score
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    head_num = query.shape[3]
    softmax_scale = query.shape[-1] ** -0.5
    block_out, block_lse = torch_npu.npu_fused_infer_attention_score(query, key, value,
                                                                     num_heads=head_num,
                                                                     input_layout="BSND",
                                                                     scale=softmax_scale,
                                                                     softmax_lse_flag=True,
                                                                     pre_tokens=65535,
                                                                     next_tokens=65535
                                                                     )
    block_out = block_out.transpose(1, 2)
    block_lse = block_lse.squeeze(-1)
    return block_out, block_lse

