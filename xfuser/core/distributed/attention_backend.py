import functools
import torch
import inspect
import math
import torch.nn.functional as F
from enum import Enum
from xfuser.envs import PACKAGES_CHECKER, environment_variables
from xfuser.core.distributed.ssta import setup_ssta, get_sparse_mask, untile_ssta_output
from xfuser.core.distributed import get_ulysses_parallel_world_size

ATTENTION_FUNCTION_REGISTRY = {}

def _setup_aiter_environment_variables():
    AITER_FP8_STATIC_SCALE_WITH_DESCALE = environment_variables["AITER_FP8_STATIC_SCALE_WITH_DESCALE"]()
    try:
        scale = float(AITER_FP8_STATIC_SCALE_WITH_DESCALE)
        AITER_FP8_STATIC_SCALE_WITH_DESCALE = scale if scale > 1 else None
    except (TypeError, ValueError):
        AITER_FP8_STATIC_SCALE_WITH_DESCALE = None
    AITER_FP8_STATIC_SCALE_NO_DESCALE = 1.0 # This value should be 1.0 when descale vectors are not used.
    _aiter_sage_v2_block_r = environment_variables["AITER_SAGE_V2_BLOCK_R"]()
    try:
        _block_r = int(_aiter_sage_v2_block_r)
        AITER_SAGE_V2_BLOCK_R = _block_r if _block_r in [16, 32, 64, 128] else 128
    except (TypeError, ValueError):
        AITER_SAGE_V2_BLOCK_R = 128
    return AITER_FP8_STATIC_SCALE_WITH_DESCALE, AITER_FP8_STATIC_SCALE_NO_DESCALE, AITER_SAGE_V2_BLOCK_R

def _check_aiter_round_mode():
    HOW_V3_BF16_CVT = None
    try:
        AITER_HAS_ROUND_MODE = inspect.signature(flash_attn_func_aiter).parameters.get("how_v3_bf16_cvt") is not None
    except (AttributeError, TypeError):
        AITER_HAS_ROUND_MODE = False
    if AITER_HAS_ROUND_MODE:
        HOW_V3_BF16_CVT = 2
    return AITER_HAS_ROUND_MODE, HOW_V3_BF16_CVT

def _check_aiter_fp8_has_descale():
    try:
        AITER_FP8_HAS_DESCALE = inspect.signature(aiter.flash_attn_fp8_pertensor_func).parameters.get("q_descale") is not None
    except (AttributeError, TypeError):
        AITER_FP8_HAS_DESCALE = False
    return AITER_FP8_HAS_DESCALE

def _aiter_sage_v2_hadamard_matrix(block_r):
    hadamard_matrix = {}
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import create_hadamard_matrix
        # Create the hadamard_matrix and replicate it on each available GPU
        _hadamard = create_hadamard_matrix(block_r, dtype=torch.bfloat16) / (block_r ** 0.5)
    except ImportError:
        # If create_hadamard_matrix is not available, set the hadamard_matrix to None.
        _hadamard = None
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            hadamard_matrix[device] = _hadamard.to(device) if _hadamard is not None else None
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        hadamard_matrix[device] = _hadamard.to(device) if _hadamard is not None else None
    return hadamard_matrix

aten = torch.ops.aten
env_info = PACKAGES_CHECKER.get_packages_info()
if env_info["has_aiter"]:
    import aiter
    from aiter import flash_attn_func as flash_attn_func_aiter
    try:
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func, get_sage_fwd_configs
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE is not available.
    try:
        from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
            fav3_sage_mxfp4_wrapper, get_sage_fwd_configs_mxfp4
        )
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE_V2 is not available.
    try:
        from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SPARSE_SAGE is not available.

    AITER_FP8_STATIC_SCALE_WITH_DESCALE, AITER_FP8_STATIC_SCALE_NO_DESCALE, AITER_SAGE_V2_BLOCK_R = _setup_aiter_environment_variables()
    AITER_HAS_ROUND_MODE, HOW_V3_BF16_CVT = _check_aiter_round_mode()
    AITER_FP8_HAS_DESCALE = _check_aiter_fp8_has_descale()
    HADAMARD_MATRIX = _aiter_sage_v2_hadamard_matrix(AITER_SAGE_V2_BLOCK_R)
    

if env_info["has_flash_attn"]:
    from flash_attn import flash_attn_func as flash_attn_func_2
if env_info["has_flash_attn_3"]:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
if env_info["has_flash_attn_4"]:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_4
if env_info["has_transformer_engine"]:
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast
    from transformer_engine.common import recipe

    TE_FP8_SCALING = recipe.DelayedScaling(
        fp8_dpa=True,
    )
if env_info["has_sage"]:
    from sageattention import sageattn
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
    NVTE_FP8 = "NVTE FP8"
    FLASH_4 = "Flash Attention V4"
    SAGE = "Sage Attention"
    AITER = "AITER"
    AITER_FP8 = "AITER FP8"
    AITER_SAGE = "AITER Sage"
    AITER_SPARSE_SAGE = "AITER Sparse Sage"
    AITER_SAGE_V2 = "AITER Sage V2"
    AITER_SPARSE_SAGE_V2 = "AITER Sparse Sage V2"
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
def _sdpa_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs attention through PyTorch's scaled_dot_product_attention.
    Allows Pytorch to decide which SDPA backend to use.
    """
    output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )
    return output, None

@register_attention_function(AttentionBackendType.SDPA_FLASH)
def _sdpa_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _sdpa_math_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _sdpa_efficient_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _cudnn_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _flash_attn_3_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _flash_attn_3_fp8_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _flash_attn_4_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _aiter_fp8_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    softmax_lse = None
    quant_dtype = aiter.dtypes.fp8
    dtypeMax = torch.finfo(quant_dtype).max
    if AITER_FP8_HAS_DESCALE:
        # If AITER_FP8_STATIC_SCALE_WITH_DESCALE is not set, use dynamic scaling.
        # Set the environment variable XFUSER_AITER_FP8_STATIC_SCALE_WITH_DESCALE
        # to a float value (i.e 2.5) to use static scaling.
        if AITER_FP8_STATIC_SCALE_WITH_DESCALE is None:
            scale = None
        else:
            scale=torch.tensor(AITER_FP8_STATIC_SCALE_WITH_DESCALE, dtype=torch.float32, device=query.device)
    else:
        # Use static scale of 1.0, since descale is not available.
        scale = torch.tensor(AITER_FP8_STATIC_SCALE_NO_DESCALE, dtype=torch.float32, device=query.device)
    quant_q, q_descale = aiter.per_tensor_quant(query,
                                                scale=scale,
                                                quant_dtype=quant_dtype,
                                                dtypeMax=dtypeMax)
    quant_k, k_descale = aiter.per_tensor_quant(key,
                                                scale=scale,
                                                quant_dtype=quant_dtype,
                                                dtypeMax=dtypeMax)
    quant_v, v_descale = aiter.per_tensor_quant(value,
                                                scale=scale,
                                                quant_dtype=quant_dtype,
                                                dtypeMax=dtypeMax)

    kwargs = {}
    if AITER_FP8_HAS_DESCALE:
        kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }
    output = aiter.flash_attn_fp8_pertensor_func(
        quant_q, quant_k, quant_v,
        causal=is_causal,
        **kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER)
def _aiter_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    kwargs = {
        "dropout_p": dropout_p,
        "causal": is_causal,
        "return_attn_probs": False,
        "return_lse": True,
    }
    if AITER_HAS_ROUND_MODE:
        kwargs["how_v3_bf16_cvt"] = HOW_V3_BF16_CVT
    output, softmax_lse = flash_attn_func_aiter(
        query,
        key,
        value,
        **kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH)
def _flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def npu_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor transpose and
    then calls attention through npu_fused_infer_attention_score
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    head_num = query.shape[2]
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

@register_attention_function(AttentionBackendType.AITER_SAGE)
def _aiter_sage_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    # Pass layout="bhsd" to avoid permutation
    softmax_lse = None
    attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd")
    output = attn_fn(query, key, value)
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER_SAGE_V2)
def _aiter_sage_v2_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    # Contiguous is needed for Sage v2 in older AITER versions. 
    # This has been fixed in newer version of AITER, meaning the
    # contiguous calls can be removed in the future.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    softmax_lse = None
    attn_fn = functools.partial(fav3_sage_mxfp4_wrapper, layout="bhsd", hadamard_rotation=True, R=HADAMARD_MATRIX[query.device])
    output = attn_fn(query, key, value, causal=is_causal)
    return output, softmax_lse


@register_attention_function(AttentionBackendType.SAGE)
def _sage_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    output, softmax_lse = sageattn(
        query,
        key,
        value,
        is_causal=is_causal,
        return_lse=True
    )
    return output, softmax_lse


@register_attention_function(AttentionBackendType.AITER_SPARSE_SAGE)
def _aiter_sparse_sage_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    attention_kwargs["sp_size"] = get_ulysses_parallel_world_size()
    block_size = math.prod(attention_kwargs["tile_size"])
    config = get_sage_fwd_configs()
    config["BLOCK_M"] = block_size
    config["BLOCK_N"] = block_size
    attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd", config=config)
    q, k, v, mask_config, ssta_state = setup_ssta(query, key, value, attention_kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=attention_kwargs["attn_sparse_type"])
    block_lut = block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1])
    output = attn_fn(q, k, v, block_lut=block_lut)
    output = untile_ssta_output(output, ssta_state, attention_kwargs["encoder_sequence_length"], attention_kwargs["sp_size"])
    return output, None


@register_attention_function(AttentionBackendType.AITER_SPARSE_SAGE_V2)
def _aiter_sparse_sage_v2_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    attention_kwargs["sp_size"] = get_ulysses_parallel_world_size()
    block_size = math.prod(attention_kwargs["tile_size"])
    config = get_sage_fwd_configs_mxfp4()
    config["BLOCK_M"] = block_size
    config["BLOCK_N"] = block_size
    attn_fn = functools.partial(fav3_sage_mxfp4_wrapper, layout="bhsd", hadamard_rotation=True, R=HADAMARD_MATRIX[query.device], config=config)
    q, k, v, mask_config, ssta_state = setup_ssta(query, key, value, attention_kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=attention_kwargs["attn_sparse_type"])
    block_lut = block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1])
    output = attn_fn(q, k, v, causal=is_causal, block_lut=block_lut)
    output = untile_ssta_output(output, ssta_state, attention_kwargs["encoder_sequence_length"], attention_kwargs["sp_size"])
    return output, None

@functools.lru_cache(maxsize=32)
def _get_cached_te_fp8_dot_product_attention(
    num_attention_heads: int,
    kv_channels: int,
    attn_mask_type: str,
    device_index: int,
):
    return DotProductAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        qkv_format="bshd",
        attn_mask_type=attn_mask_type,
        attention_dropout=0.0,
    ).to(torch.device("cuda", device_index)).eval()

@register_attention_function(AttentionBackendType.NVTE_FP8)
def _nvte_fp8_flash_attn_call(query, key, value, dropout_p, is_causal):
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    batch, seqlen, num_heads, head_dim = query.shape
    attn_mask_type = "causal" if is_causal else "no_mask"
    device_index = query.device.index if query.device.index is not None else 0
    dpa = _get_cached_te_fp8_dot_product_attention(
        num_heads,
        head_dim,
        attn_mask_type,
        device_index,
    )
    with fp8_autocast(enabled=True, fp8_recipe=TE_FP8_SCALING):
        out = dpa(query, key, value, attn_mask_type=attn_mask_type)
    out = out.view(batch, seqlen, num_heads, head_dim).permute(0, 2, 1, 3)
    return out, None
