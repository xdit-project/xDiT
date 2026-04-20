import functools
import torch
import inspect
import torch.nn.functional as F
from enum import Enum
from xfuser.envs import PACKAGES_CHECKER, environment_variables

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

def _get_mla_cache_device_key(device):
    if device.type == "cuda":
        return (device.type, torch.cuda.current_device() if device.index is None else device.index)
    return (device.type, device.index)

_MLA_PREFILL_QK_HEAD_DIM = 192
_MLA_BLOCK_SIZE = 1
_MLA_TILE_Q = 256
_MLA_TILE_KV = 128
_MLA_METADATA_CACHE = {}

def _launch_mla_prefill_reduce(
    query_ragged, key_ragged, value_ragged,
    metadata, qk_head_dim, num_heads, v_head_dim,
    query_scale, key_scale, value_scale,
    batch_size, q_seq_len, kv_seq_len,
    device
):
    """Launch MLA prefill and reduce kernels, returning output in ragged layout."""
    total_q_tokens = batch_size * q_seq_len
    total_kv_tokens = batch_size * kv_seq_len
    softmax_scale = qk_head_dim ** -0.5

    output = torch.empty(
        (total_q_tokens, num_heads, v_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    partial_tiles = metadata["reduce_partial_map"].size(0) * _MLA_TILE_Q
    logits = torch.empty(
        (partial_tiles, num_heads, v_head_dim),
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        (partial_tiles, num_heads),
        dtype=torch.float32,
        device=device,
    )
    final_lse = torch.empty(
        (total_q_tokens, num_heads),
        dtype=torch.float32,
        device=device,
    )

    aiter.mla_prefill_ps_asm_fwd(
        query_ragged,
        key_ragged,
        value_ragged,
        metadata["qo_indptr"],
        metadata["kv_indptr"],
        metadata["kv_indices"],
        metadata["work_indptr"],
        metadata["work_info"],
        metadata["max_seqlen_q"],
        softmax_scale,
        False,  # is_causal (prefill is non-causal)
        logits,
        attn_lse,
        output,
        query_scale,
        key_scale,
        value_scale,
    )
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        metadata["reduce_indptr"],
        metadata["reduce_final_map"],
        metadata["reduce_partial_map"],
        _MLA_TILE_Q,
        output,
        final_lse,
    )

    return output


def _run_mla_bshd(q_bshd, k_bshd, v_bshd):
    """Execute MLA prefill+reduce for tensors in BSHD layout."""
    _batch, _q_seq, _num_heads, _qk_head_dim = q_bshd.shape
    _, _kv_seq, _num_kv_heads, _v_head_dim = v_bshd.shape

    q_for_kernel = q_bshd
    k_for_kernel = k_bshd
    if _qk_head_dim < _MLA_PREFILL_QK_HEAD_DIM:
        pad_qk = _MLA_PREFILL_QK_HEAD_DIM - _qk_head_dim
        q_for_kernel = F.pad(q_for_kernel, (0, pad_qk))
        k_for_kernel = F.pad(k_for_kernel, (0, pad_qk))

    fp8_dtype = aiter.dtypes.fp8
    query_fp8, query_scale = aiter.per_tensor_quant(q_for_kernel, quant_dtype=fp8_dtype)
    key_fp8, key_scale = aiter.per_tensor_quant(k_for_kernel, quant_dtype=fp8_dtype)
    value_fp8, value_scale = aiter.per_tensor_quant(v_bshd, quant_dtype=fp8_dtype)

    total_q_tokens = _batch * _q_seq
    total_kv_tokens = _batch * _kv_seq
    query_ragged = query_fp8.reshape(total_q_tokens, _num_heads, query_fp8.shape[-1])
    key_ragged = key_fp8.reshape(total_kv_tokens, _num_kv_heads, key_fp8.shape[-1])
    value_ragged = value_fp8.reshape(total_kv_tokens, _num_kv_heads, _v_head_dim)

    metadata = _build_aiter_mla_metadata(
        batch_size=_batch,
        q_seq_len=_q_seq,
        kv_seq_len=_kv_seq,
        num_heads=_num_heads,
        num_kv_heads=_num_kv_heads,
        device=q_bshd.device,
    )

    output_ragged = _launch_mla_prefill_reduce(
        query_ragged,
        key_ragged,
        value_ragged,
        metadata,
        _qk_head_dim,
        _num_heads,
        _v_head_dim,
        query_scale,
        key_scale,
        value_scale,
        _batch,
        _q_seq,
        _kv_seq,
        q_bshd.device,
    )

    return output_ragged.view(_batch, _q_seq, _num_heads, _v_head_dim)

def _build_aiter_mla_metadata(batch_size, q_seq_len, kv_seq_len, num_heads, num_kv_heads, device):
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"AITER MLA requires num_heads ({num_heads}) to be divisible by num_kv_heads ({num_kv_heads})."
        )

    cache_key = (
        _get_mla_cache_device_key(device),
        batch_size,
        q_seq_len,
        kv_seq_len,
        num_heads,
        num_kv_heads,
    )
    cached = _MLA_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gqa_ratio = num_heads // num_kv_heads
    blocks_per_seq = (kv_seq_len + _MLA_BLOCK_SIZE - 1) // _MLA_BLOCK_SIZE
    num_blocks = batch_size * blocks_per_seq
    max_qlen = q_seq_len

    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * q_seq_len
    kv_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * blocks_per_seq
    kv_seq_lens_cpu = torch.full((batch_size,), kv_seq_len, dtype=torch.int32)
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)

    qhead_granularity = gqa_ratio
    qlen_granularity = _MLA_TILE_Q // qhead_granularity
    kvlen_granularity = max(_MLA_TILE_KV, _MLA_BLOCK_SIZE)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_kv_heads,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
    )

    work_metadata_ptrs = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device=device
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device=device
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device=device
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
    )

    aiter.get_ps_metadata_v1(
        qo_indptr_cpu,
        kv_indptr_cpu,
        kv_seq_lens_cpu,
        gqa_ratio,
        num_kv_heads,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=_MLA_BLOCK_SIZE,
        is_causal=False,
    )

    metadata = {
        "qo_indptr": qo_indptr_cpu.to(device),
        "kv_indptr": kv_indptr_cpu.to(device),
        "kv_indices": kv_indices,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "max_seqlen_q": max_qlen,
    }
    _MLA_METADATA_CACHE[cache_key] = metadata
    return metadata

aten = torch.ops.aten
env_info = PACKAGES_CHECKER.get_packages_info()
if env_info["has_aiter"]:
    import aiter
    from aiter import flash_attn_func as flash_attn_func_aiter
    try:
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE is not available.
    try:
        from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
            fav3_sage_mxfp4_wrapper,
        )
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE_V2 is not available.

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
if env_info["has_flash_attn_4_fp4"]:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_4_fp4
    from xfuser.core.distributed.fp4_quantize import quantize_qk_to_fp4
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
    FLASH_4_FP4 = "Flash Attention V4 FP4"
    SAGE = "Sage Attention"
    AITER = "AITER"
    AITER_FP8 = "AITER FP8"
    AITER_MLA = "AITER MLA"
    AITER_SAGE = "AITER Sage"
    AITER_SAGE_V2 = "AITER Sage V2"
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

@register_attention_function(AttentionBackendType.FLASH_4_FP4)
@torch.compiler.disable
def _flash_attn_4_fp4_call(query, key, value, dropout_p, is_causal):
    """
    Flash Attention V4 with runtime FP4 quantization of Q and K.

    Input tensors arrive in (batch, nheads, seqlen, headdim) from the USP layer.
    The FAv4 kernel expects (batch, seqlen, nheads, headdim).
    Q and K are quantized to NVFP4 via flashinfer's nvfp4_quantize; V stays in BF16.
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    q_fp4, mSFQ = quantize_qk_to_fp4(query)
    k_fp4, mSFK = quantize_qk_to_fp4(key)

    output, softmax_lse = flash_attn_func_4_fp4(
        q_fp4,
        k_fp4,
        value,
        causal=is_causal,
        mSFQ=mSFQ,
        mSFK=mSFK,
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

    attn_kwargs = {}
    if AITER_FP8_HAS_DESCALE:
        attn_kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }
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

@register_attention_function(AttentionBackendType.AITER_MLA)
def _aiter_mla_attn_call(query, key, value, dropout_p, is_causal):
    """Entry point for AITER MLA prefill backend. Thin wrapper around _run_mla_bshd."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("AITER MLA expects query, key, and value tensors in BHSD layout.")

    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    batch_size, q_seq_len, num_heads, qk_head_dim = query.shape
    key_batch_size, kv_seq_len, num_kv_heads, key_head_dim = key.shape
    value_batch_size, value_seq_len, value_num_kv_heads, v_head_dim = value.shape

    if key_batch_size != batch_size or value_batch_size != batch_size:
        raise ValueError("AITER MLA requires matching batch sizes for query, key, and value.")
    if value_seq_len != kv_seq_len:
        raise ValueError("AITER MLA requires key and value to have the same sequence length.")
    if value_num_kv_heads != num_kv_heads:
        raise ValueError("AITER MLA requires key and value to have the same number of KV heads.")
    if key_head_dim != qk_head_dim:
            raise ValueError("AITER MLA prefill backend currently assumes QK head dimensions to be equal.")
    if num_heads != num_kv_heads:
        raise ValueError(
            "AITER MLA prefill backend currently assumes Hq == Hkv for diffusion inference."
        )
    if qk_head_dim > _MLA_PREFILL_QK_HEAD_DIM:
        raise ValueError(
            f"AITER MLA supports QK head dimensions up to {_MLA_PREFILL_QK_HEAD_DIM}, got {qk_head_dim}."
        )

    original_dtype = query.dtype

    # Some MLA kernels reject multi-head settings for D=128 (e.g. H=5), while H=1 is supported.
    # Avoid the failing kernel path up front by scheduling per-query-head MLA calls.
    use_per_head_schedule = (
        qk_head_dim == 128
        and num_heads != 1
        and num_heads != 2
        and num_heads != 4
        and num_heads != 8
    )

    if use_per_head_schedule:
        head_outputs = []
        for h in range(num_heads):
            head_outputs.append(
                _run_mla_bshd(
                    query[:, :, h : h + 1, :],
                    key[:, :, h : h + 1, :],
                    value[:, :, h : h + 1, :],
                )
            )
        output = torch.cat(head_outputs, dim=2)
    else:
        output = _run_mla_bshd(query, key, value)

    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    output = torch.permute(output, [0, 2, 1, 3])
    return output, None

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
def _aiter_sage_attn_call(query, key, value, dropout_p, is_causal):
    # Pass layout="bhsd" to avoid permutation
    softmax_lse = None
    attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd")
    output = attn_fn(query, key, value)
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER_SAGE_V2)
def _aiter_sage_v2_attn_call(query, key, value, dropout_p, is_causal):
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
def _sage_attn_call(query, key, value, dropout_p, is_causal):
    output, softmax_lse = sageattn(
        query,
        key,
        value,
        is_causal=is_causal,
        return_lse=True
    )
    return output, softmax_lse

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