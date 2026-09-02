import functools
from dataclasses import dataclass, replace
import torch
import inspect
import math
import torch.nn.functional as F
from enum import Enum
from xfuser.envs import PACKAGES_CHECKER, environment_variables
from xfuser.core.distributed.ssta import (
    setup_ssta,
    get_sparse_mask,
    untile_ssta_output,
    expand_block_mask,
)
from xfuser.core.distributed import get_ulysses_parallel_world_size, get_ring_parallel_world_size
from xfuser.core.sparge_attention.sparge import (
    setup_sparge,
    compute_sparge_block_mask,
    restore_sparge_output,
    mask_padded_kv_blocks,
)
from xfuser.core.sparge_attention.head_balance import COST_SINK_KEY
from xfuser.logger import init_logger

logger = init_logger(__name__)

ATTENTION_FUNCTION_REGISTRY = {}


@dataclass(frozen=True)
class _AiterMhaV4Capabilities:
    enabled: bool = False
    is_gfx942: bool = False
    block_mask: bool = False
    mxfp8_block_mask: bool = False
    kv_tile: int = 128


_AITER_MHA_V4 = _AiterMhaV4Capabilities()


def _probe_aiter_mha_v4_capabilities(mha_v4_fn) -> _AiterMhaV4Capabilities:
    arch_name = (
        torch.cuda.get_device_properties(0).gcnArchName
        if torch.cuda.is_available()
        else ""
    )
    is_gfx942 = "gfx942" in arch_name
    enabled = is_gfx942 or "gfx950" in arch_name
    block_mask = inspect.signature(mha_v4_fn).parameters.get("block_mask") is not None
    try:
        from aiter.ops.mha_v4 import mha_v4_kv_tile as _aiter_mha_v4_kv_tile
        kv_tile = int(_aiter_mha_v4_kv_tile())
    except ImportError:
        kv_tile = 64 if is_gfx942 else 128
    return _AiterMhaV4Capabilities(
        enabled=enabled,
        is_gfx942=is_gfx942,
        block_mask=block_mask,
        kv_tile=kv_tile,
    )


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

def _check_aiter_sage_supports_ring():
    try:
        parameters = inspect.signature(fav3_sage_wrapper_func).parameters
        return "return_lse" in parameters and "smooth_k" in parameters
    except (NameError, ImportError, AttributeError, TypeError):
        return False

def _check_aiter_sage_v2_supports_ring():
    try:
        return inspect.signature(fav3_sage_mxfp4_wrapper).parameters.get("return_lse") is not None
    except (NameError, ImportError, AttributeError, TypeError):
        return False

def _check_aiter_flydsl_generalized():
    # Older builds raise on cross-attn and on >0.5%-padded non-causal self-attn; the gfx1201
    # kernel gained both in the same commit that added softmax_scale, so probe for that.
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_func
        return "softmax_scale" in inspect.signature(flydsl_flash_attn_func).parameters
    except (NameError, ImportError, AttributeError, TypeError):
        return False

def _build_hadamard_matrix(block_r, dtype=torch.bfloat16, allow_sylvester_fallback=True):
    """Normalized Hadamard matrix (block_r x block_r, R @ R.T == I; block_r a
    power of two). Uses aiter's create_hadamard_matrix. If that's unavailable,
    falls back to a local Sylvester construction when allow_sylvester_fallback
    is set, otherwise returns None."""
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                create_hadamard_matrix,
            )
        return create_hadamard_matrix(block_r, dtype=dtype) / (block_r ** 0.5)
    except ImportError:
        if not allow_sylvester_fallback:
            return None
        # Local Sylvester construction: H1=[[1]], H2n=[[Hn,Hn],[Hn,-Hn]].
        assert block_r > 0 and (block_r & (block_r - 1)) == 0, 'Hadamard block_r must be a positive power of 2'
        H = torch.ones((1, 1), dtype=torch.float32)
        while H.shape[0] < block_r:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return (H / (block_r ** 0.5)).to(dtype)


def _replicate_hadamard_per_device(hadamard):
    """Replicate a single Hadamard matrix on each available device, keyed by
    torch.device (CPU plus all GPUs when CUDA is available). A None matrix maps
    to None on every device."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices += [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return {
        device: (hadamard.to(device) if hadamard is not None else None)
        for device in devices
    }


def _aiter_hadamard_matrix(block_r, allow_sylvester_fallback=True):
    """Build a normalized Hadamard matrix and replicate it across devices."""
    return _replicate_hadamard_per_device(
        _build_hadamard_matrix(
            block_r, dtype=torch.bfloat16, allow_sylvester_fallback=allow_sylvester_fallback
        )
    )

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
AITER_FP8_DTYPE = torch.float8_e4m3fn  # fallback; fp8 comms requires aiter
FP8_HADAMARD_MATRIX = _aiter_hadamard_matrix(128)
if env_info["has_aiter"]:
    import aiter
    AITER_FP8_DTYPE = aiter.dtypes.fp8
    from aiter import flash_attn_func as flash_attn_func_aiter
    from aiter import flash_attn_varlen_func as flash_attn_varlen_func_aiter
    try:
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func, get_sage_fwd_configs
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE is not available.
    try:
        from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
            fav3_sage_mxfp4_wrapper,
            get_sage_fwd_configs_mxfp4,
        )
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SAGE_V2 is not available.
    try:
        from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
    except ImportError:
        pass # Error is rasied in runtime_state.py if AITER_SPARSE_SAGE is not available.

    try:
        from aiter.ops.mha_v4 import (
            AttentionFormat as _AiterAttentionFormat,
            AttentionScaleMode as _AiterAttentionScaleMode,
            mha_v4 as _aiter_mha_v4,
            mha_v4_packed as _aiter_mha_v4_packed,
            mha_v4_q_multiplier as _aiter_mha_v4_q_multiplier,
            native_fp8_format as _aiter_native_fp8_format,
            quantize_fp8 as _aiter_quantize_fp8,
            quantize_mxfp8_k as _aiter_quantize_mxfp8_k,
            quantize_mxfp8_q as _aiter_quantize_mxfp8_q,
        )
        _AITER_MHA_V4 = _probe_aiter_mha_v4_capabilities(_aiter_mha_v4)
    except ImportError:
        pass # Error is raised in runtime_state.py when an MHA v4 backend is selected.

    # MXFP8 shipped after the base MHA v4 API; keep it optional for older AITER builds.
    try:
        from aiter.ops.mha_v4 import (
            mha_v4_mxfp8 as _aiter_mha_v4_mxfp8,
        )
        _AITER_MHA_V4 = replace(
            _AITER_MHA_V4,
            mxfp8_block_mask=(
                inspect.signature(_aiter_mha_v4_mxfp8).parameters.get("block_mask")
                is not None
            ),
        )
    except ImportError:
        pass # Error is raised in runtime_state.py when AITER_MXFP8 is selected.

    AITER_FP8_STATIC_SCALE_WITH_DESCALE, AITER_FP8_STATIC_SCALE_NO_DESCALE, AITER_SAGE_V2_BLOCK_R = _setup_aiter_environment_variables()
    AITER_HAS_ROUND_MODE, HOW_V3_BF16_CVT = _check_aiter_round_mode()
    AITER_FP8_HAS_DESCALE = _check_aiter_fp8_has_descale()
    AITER_SAGE_SUPPORTS_RING = _check_aiter_sage_supports_ring()
    AITER_SAGE_V2_SUPPORTS_RING = _check_aiter_sage_v2_supports_ring()
    AITER_FLYDSL_GENERALIZED = _check_aiter_flydsl_generalized()
    # sage_v2 relies on aiter's own matrix and has no Sylvester fallback (None
    # disables hadamard_rotation when create_hadamard_matrix is unavailable).
    HADAMARD_MATRIX = _aiter_hadamard_matrix(AITER_SAGE_V2_BLOCK_R, allow_sylvester_fallback=False)
    # 128-blocked FP8 Hadamard matrix (per-device), used by every model with
    # head_dim a multiple of 128.
    FP8_HADAMARD_MATRIX = _aiter_hadamard_matrix(128)
    # Extra FP8 Hadamard matrices for smaller power-of-two head dims (e.g. LTX-2.5
    # audio head_dim=64), built lazily and keyed by head_dim.
    _FP8_HADAMARD_MATRICES: dict = {}

    def _get_fp8_hadamard_matrix(head_dim: int, device: torch.device) -> torch.Tensor:
        # 128-blocked rotation for head_dim that is a multiple of 128 (all existing
        # models); full-head rotation for smaller power-of-two dims (audio=64).
        if head_dim % 128 == 0:
            return FP8_HADAMARD_MATRIX[device]
        if head_dim not in _FP8_HADAMARD_MATRICES:
            _FP8_HADAMARD_MATRICES[head_dim] = _aiter_hadamard_matrix(head_dim)
        return _FP8_HADAMARD_MATRICES[head_dim][device]

    _TRITON_SSTA_BLOCK_SIZE = 128
    

if env_info["has_aiter"]:
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_func as flydsl_flash_attn_func_aiter
        from torch.library import custom_op, register_fake

        # fp8 quant ships in a newer aiter than the bf16 kernel, so keep it optional: a
        # bf16-only build still registers both ops, and runtime_state refuses the fp8 backend.
        try:
            from aiter.ops.flydsl import flydsl_fp8_quant as flydsl_fp8_quant_aiter
        except ImportError:
            flydsl_fp8_quant_aiter = None

        # Two ops mirror the AITER / AITER_FP8 split: AITER_FLYDSL -> xfuser::flydsl_attn (bf16),
        # AITER_FLYDSL_FP8 -> xfuser::flydsl_attn_fp8. fp8 is unfused (faster e2e) so it holds fp8
        # Q/K/V alongside the live bf16 Q/K/V -> higher peak VRAM; pick AITER_FLYDSL when tight.

        # fp8 wins only above a seq crossover (quant pre-pass cost vs K/V HBM bytes saved), which
        # depends on (head_dim, num_heads). Measured on gfx1201: D64/H38 and D128/H<=32 flip at
        # S~2560; D128 high head-count (wan H40) at S~3584. Below: fp8 loses 7-18%; above: wins <4%.
        def _flydsl_fp8_min_seq(head_dim: int, num_heads: int) -> int:
            if head_dim >= 128 and num_heads > 32:
                return 3584
            return 2560

        def _flydsl_fp8_attn(query, key, value, is_causal):
            # flydsl_fp8_quant returns fp8 q/k/v + descales (real = fp8 * descale).
            qq, kk, vv, sq, sk, sv = flydsl_fp8_quant_aiter(query, key, value, rotation=True)
            return flydsl_flash_attn_func_aiter(
                qq, kk, vv, causal=is_causal,
                q_descale=sq, k_descale=sk, v_descale=sv,
                waves_per_eu=2, daz=True,
            )

        # Attn shape is constant across denoise steps, so log the chosen path once per shape.
        _flydsl_logged = set()

        def _flydsl_log_once(key_t, msg):
            if key_t in _flydsl_logged:
                return
            _flydsl_logged.add(key_t)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info(msg)

        @custom_op("xfuser::flydsl_attn", mutates_args=())
        def _flydsl_attn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool,
        ) -> torch.Tensor:
            B, S_real, H, D = query.shape
            is_cross = key.shape[1] != S_real
            _flydsl_log_once(
                (B, S_real, H, D, is_cross, query.dtype),
                f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16",
            )
            return flydsl_flash_attn_func_aiter(
                query, key, value, causal=is_causal, waves_per_eu=2, daz=True
            )

        @register_fake("xfuser::flydsl_attn")
        def _flydsl_attn_fake(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool,
        ) -> torch.Tensor:
            return torch.empty_like(query)

        @custom_op("xfuser::flydsl_attn_fp8", mutates_args=())
        def _flydsl_attn_fp8_kernel(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool,
        ) -> torch.Tensor:
            # fp8 only for bf16 self-attn above the crossover; else fall back to the bf16 kernel.
            B, S_real, H, D = query.shape
            is_cross = key.shape[1] != S_real
            min_seq = _flydsl_fp8_min_seq(D, H)
            use_fp8 = query.dtype == torch.bfloat16 and not is_cross and S_real >= min_seq
            if use_fp8:
                msg = (
                    f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> fp8 (S>={min_seq}) "
                    "(fp8 pre-pass adds a transient QKV copy -> higher peak VRAM)"
                )
            elif query.dtype == torch.bfloat16 and not is_cross:
                msg = f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16 (S<{min_seq})"
            else:
                msg = (
                    f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16 "
                    f"(not fp8-eligible: dtype={query.dtype}, cross={is_cross})"
                )
            _flydsl_log_once((B, S_real, H, D, is_cross, query.dtype), msg)
            if use_fp8:
                return _flydsl_fp8_attn(query, key, value, is_causal)
            return flydsl_flash_attn_func_aiter(
                query, key, value, causal=is_causal, waves_per_eu=2, daz=True
            )

        @register_fake("xfuser::flydsl_attn_fp8")
        def _flydsl_attn_fp8_fake(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool,
        ) -> torch.Tensor:
            return torch.empty_like(query)

    except ImportError:
        pass
if env_info["has_flash_attn"]:
    from flash_attn import flash_attn_func as flash_attn_func_2
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_2
if env_info["has_flash_attn_3"]:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_3
if env_info["has_flash_attn_4"]:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_4
    from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_4
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
if env_info["has_flex_block_attn"]:
    from flex_block_attn import flex_block_attn_func
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
    FLEX_BLOCK_ATTN = "Flex Block Attention"
    AITER = "AITER"
    AITER_MLA = "AITER MLA" # deprecated, use AITER_FP8
    AITER_I8FP8 = "AITER I8FP8"
    AITER_FP8 = "AITER FP8"
    AITER_MXFP8 = "AITER MXFP8"
    AITER_F8F6 = "AITER F8F6"
    AITER_MXFP6 = "AITER MXFP6"
    AITER_F6F4 = "AITER F6F4"
    AITER_MXFP4 = "AITER MXFP4"
    AITER_F4F4 = "AITER F4F4"
    AITER_I8FP8_SPARGE = "AITER I8FP8 Sparge"
    AITER_FP8_SPARGE = "AITER FP8 Sparge"
    AITER_MXFP8_SPARGE = "AITER MXFP8 Sparge"
    AITER_F8F6_SPARGE = "AITER F8F6 Sparge"
    AITER_MXFP6_SPARGE = "AITER MXFP6 Sparge"
    AITER_F6F4_SPARGE = "AITER F6F4 Sparge"
    AITER_MXFP4_SPARGE = "AITER MXFP4 Sparge"
    AITER_F4F4_SPARGE = "AITER F4F4 Sparge"
    AITER_SAGE = "AITER Sage"
    AITER_SPARSE_SAGE = "AITER Sparse Sage"
    AITER_SAGE_V2 = "AITER Sage V2"
    AITER_SPARSE_SAGE_V2 = "AITER Sparse Sage V2"
    AITER_SPARGE = "AITER Sparge"
    AITER_SPARGE_V2 = "AITER Sparge V2"
    AITER_VSA = "AITER VSA CK"
    FLEX_BLOCK_SPARGE = "Flex Block Sparge"
    AITER_FLYDSL = "AITER FlyDSL"
    AITER_FLYDSL_FP8 = "AITER FlyDSL FP8"
    NPU = "NPU"


AITER_LOW_PRECISION_BACKENDS = (
    AttentionBackendType.AITER_I8FP8,
    AttentionBackendType.AITER_FP8,
    AttentionBackendType.AITER_MXFP8,
    AttentionBackendType.AITER_F8F6,
    AttentionBackendType.AITER_MXFP6,
    AttentionBackendType.AITER_F6F4,
    AttentionBackendType.AITER_MXFP4,
    AttentionBackendType.AITER_F4F4,
)
AITER_MHA_V4_SPARGE_BACKENDS = (
    AttentionBackendType.AITER_I8FP8_SPARGE,
    AttentionBackendType.AITER_FP8_SPARGE,
    AttentionBackendType.AITER_MXFP8_SPARGE,
    AttentionBackendType.AITER_F8F6_SPARGE,
    AttentionBackendType.AITER_MXFP6_SPARGE,
    AttentionBackendType.AITER_F6F4_SPARGE,
    AttentionBackendType.AITER_MXFP4_SPARGE,
    AttentionBackendType.AITER_F4F4_SPARGE,
)
AITER_MHA_V4_ONLY_BACKENDS = tuple(
    backend
    for backend in AITER_LOW_PRECISION_BACKENDS
    if backend != AttentionBackendType.AITER_FP8
)
AITER_MHA_V4_ONLY_BACKEND_SET = frozenset(AITER_MHA_V4_ONLY_BACKENDS)
AITER_MHA_V4_SPARGE_BACKEND_SET = frozenset(AITER_MHA_V4_SPARGE_BACKENDS)
AITER_MHA_V4_GFX942_SPARGE_BACKENDS = (
    AttentionBackendType.AITER_I8FP8_SPARGE,
    AttentionBackendType.AITER_FP8_SPARGE,
)
AITER_MHA_V4_GFX942_SPARGE_BACKEND_SET = frozenset(AITER_MHA_V4_GFX942_SPARGE_BACKENDS)


def _mha_v4_sparge_tile():
    """Return Sparge tile sizes matching the active MHA v4 sparse KV geometry."""
    return {"BLOCK_M": 256, "BLOCK_N": _AITER_MHA_V4.kv_tile}


_FP8_INPUT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

SUPPORTS_PRE_QUANTIZATION_BACKENDS = {
    AttentionBackendType.AITER_FP8,
}


def register_attention_function(backend_type):
    """
    Decorator to register attention functions with their corresponding backend type.
    """
    def decorator(func):
        ATTENTION_FUNCTION_REGISTRY[backend_type] = func
        return func
    return decorator

def _varlen_pack_keys(query_bshd, key_bshd, value_bshd, attention_kwargs):
    """Pack K/V using pre-computed mask indices for varlen attention kernels.

    Called after BHSD->BSHD permute.  Returns a tuple with everything needed
    to call a varlen function, or None when no mask indices are present.
    Only K/V are packed and Q is never filtered (all B*S query positions are kept).
    """
    indices_k = (attention_kwargs or {}).get("indices_k")
    if indices_k is None:
        return None
    B, S, H, D = query_bshd.shape
    Sk = key_bshd.shape[1]  # key seqlen may differ from query in cross-attention
    k_flat = key_bshd.reshape(B * Sk, H, D)
    v_flat = value_bshd.reshape(B * Sk, H, D)
    k_packed = torch.index_select(k_flat, 0, indices_k)
    v_packed = torch.index_select(v_flat, 0, indices_k)
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=query_bshd.device) * S
    return (
        query_bshd.reshape(B * S, H, D),
        k_packed,
        v_packed,
        cu_seqlens_q,
        attention_kwargs["cu_seqlens_k"],
        attention_kwargs["max_seqlen_k"],
        B, S, H, D,
    )


@register_attention_function(AttentionBackendType.SDPA)
def _sdpa_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs attention through PyTorch's scaled_dot_product_attention.
    Allows Pytorch to decide which SDPA backend to use.
    """
    attn_mask = attention_kwargs.get("attn_mask") if attention_kwargs else None
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
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
    attn_mask = attention_kwargs.get("attn_mask") if attention_kwargs else None
    output, softmax_lse = aten._scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask=attn_mask,
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
    packed = _varlen_pack_keys(query, key, value, attention_kwargs)
    if packed is not None:
        q_flat, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, B, S, H, D = packed
        output, softmax_lse = flash_attn_varlen_func_3(
            q_flat, k_packed, v_packed,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=S, max_seqlen_k=max_seqlen_k,
            causal=is_causal, return_attn_probs=True,
        )
        output = output.reshape(B, S, H, D)
    else:
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
    packed = _varlen_pack_keys(query, key, value, attention_kwargs)
    if packed is not None:
        q_flat, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, B, S, H, D = packed
        output, softmax_lse = flash_attn_varlen_func_4(
            q_flat, k_packed, v_packed,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=S, max_seqlen_k=max_seqlen_k,
            causal=is_causal,
        )
        output = output.reshape(B, S, H, D)
    else:
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
def _flash_attn_4_fp4_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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

def _fp8_hadamard_rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard rotation along head_dim (in blocks of R.shape[-1]).
    Rotating both Q and K by R leaves Q @ K.T unchanged (kernel sees identical
    scores) while spreading outliers to cut fp8 quant error."""
    if R is None:
        return x
    d = x.shape[-1]
    block_r = R.shape[-1]
    R = R.to(x.dtype)
    if block_r == d:
        return torch.matmul(x, R)
    return torch.matmul(x.unflatten(-1, (d // block_r, block_r)), R).flatten(-2)


def rotate_qk_for_fp8_comms(query, key, backend):
    """Rotate Q,K exactly as the fp8-comms path does before quantizing them.

    Returns the inputs untouched for backends whose fp8-comms path does not rotate,
    so callers can apply this unconditionally. Single definition shared by the
    quantization site in USP and the calibration site in the attention processor:
    both must measure and quantize the same distribution, otherwise the frozen
    per-layer scale describes a tensor that is never quantized.
    """
    if backend != AttentionBackendType.AITER_FP8:
        return query, key
    R = _get_fp8_hadamard_matrix(query.shape[-1], query.device)
    return (
        _fp8_hadamard_rotate(query, R).contiguous(),
        _fp8_hadamard_rotate(key, R).contiguous(),
    )


def _quantize_aiter_fp8_inputs(query, key, value):
    quant_dtype = aiter.dtypes.fp8
    dtype_max = torch.finfo(quant_dtype).max
    if AITER_FP8_HAS_DESCALE:
        if AITER_FP8_STATIC_SCALE_WITH_DESCALE is None:
            scale = None
        else:
            scale = torch.tensor(
                AITER_FP8_STATIC_SCALE_WITH_DESCALE,
                dtype=torch.float32,
                device=query.device,
            )
    else:
        scale = torch.tensor(
            AITER_FP8_STATIC_SCALE_NO_DESCALE,
            dtype=torch.float32,
            device=query.device,
        )

    quant_q, q_descale = aiter.per_tensor_quant(
        query,
        scale=scale,
        quant_dtype=quant_dtype,
        dtypeMax=dtype_max,
    )
    quant_k, k_descale = aiter.per_tensor_quant(
        key,
        scale=scale,
        quant_dtype=quant_dtype,
        dtypeMax=dtype_max,
    )
    quant_v, v_descale = aiter.per_tensor_quant(
        value,
        scale=scale,
        quant_dtype=quant_dtype,
        dtypeMax=dtype_max,
    )
    return (
        quant_q.to(quant_dtype),
        quant_k.to(quant_dtype),
        quant_v.to(quant_dtype),
        q_descale,
        k_descale,
        v_descale,
    )


def _aiter_fp8_dense_attention(query, key, value, softmax_scale, is_causal):
    quant_q, quant_k, quant_v, q_descale, k_descale, v_descale = (
        _quantize_aiter_fp8_inputs(query, key, value)
    )
    kwargs = {}
    if AITER_FP8_HAS_DESCALE:
        kwargs = {
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }
    return aiter.flash_attn_fp8_pertensor_func(
        quant_q,
        quant_k,
        quant_v,
        causal=is_causal,
        softmax_scale=softmax_scale,
        **kwargs,
    )


@torch.library.custom_op("xfuser::aiter_fp8_varlen_attention", mutates_args=())
def _aiter_fp8_varlen_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    quant_q, quant_k, quant_v, q_descale, k_descale, v_descale = (
        _quantize_aiter_fp8_inputs(query, key, value)
    )
    kwargs = {}
    if AITER_FP8_HAS_DESCALE:
        kwargs = {
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }
    varlen_func = getattr(aiter, "flash_attn_varlen_fp8_pertensor_func", None)
    if varlen_func is None:
        raise RuntimeError(
            "AITER varlen FP8 flash attention is not available, please update AITER."
        )
    return varlen_func(
        quant_q,
        quant_k,
        quant_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=is_causal,
        **kwargs,
    )


@_aiter_fp8_varlen_attention_kernel.register_fake
def _aiter_fp8_varlen_attention_kernel_fake(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal,
):
    return torch.empty_like(query)


def _validate_aiter_low_precision_dropout(dropout_p):
    if dropout_p != 0.0:
        raise NotImplementedError("AITER low-precision attention does not support dropout")


def _validate_aiter_mha_v4_request(dropout_p, is_causal):
    _validate_aiter_low_precision_dropout(dropout_p)
    if is_causal:
        raise NotImplementedError("MHA v4 does not support causal masking")


def _use_aiter_mha_v4_fp8(query, is_causal):
    return (
        _AITER_MHA_V4.enabled
        and query.is_cuda
        and query.shape[-1] == 128
        and not is_causal
    )


@register_attention_function(AttentionBackendType.AITER_FP8)
def _aiter_fp8_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    _validate_aiter_low_precision_dropout(dropout_p)
    attention_kwargs = attention_kwargs or {}
    pre_quantized = attention_kwargs.get("pre_quantized", False)

    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    if pre_quantized:
        # Q/K/V arrive already FP8 from fp8 comms (quantized before the Ulysses
        # all-to-all).
        output = aiter.flash_attn_fp8_pertensor_func(
            query,
            key,
            value,
            causal=is_causal,
            softmax_scale=query.shape[-1] ** -0.5,
            q_descale=attention_kwargs["q_descale"],
            k_descale=attention_kwargs["k_descale"],
            v_descale=attention_kwargs["v_descale"],
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, None

    packed = _varlen_pack_keys(query, key, value, attention_kwargs)
    use_mha_v4 = packed is None and _use_aiter_mha_v4_fp8(query, is_causal)
    if use_mha_v4:
        # The raw MHA v4 API owns canonical Q/K rotation and FP8 quantization.
        fp8_format = _aiter_native_fp8_format()
        output = _aiter_mha_v4(
            query,
            key,
            value,
            fp8_format,
            fp8_format,
            fp8_format,
        )
    else:
        if packed is not None:
            (
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_k,
                batch_size,
                sequence_length,
                num_heads,
                head_dim,
            ) = packed

        # Varlen and legacy FP8 attention expect pre-rotated Q/K.
        R = _get_fp8_hadamard_matrix(query.shape[-1], query.device)
        query = _fp8_hadamard_rotate(query, R).contiguous()
        key = _fp8_hadamard_rotate(key, R).contiguous()

        if packed is not None:
            output = _aiter_fp8_varlen_attention_kernel(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                sequence_length,
                max_seqlen_k,
                head_dim**-0.5,
                is_causal,
            ).reshape(batch_size, sequence_length, num_heads, head_dim)
        else:
            output = _aiter_fp8_dense_attention(
                query,
                key,
                value,
                query.shape[-1] ** -0.5,
                is_causal,
            )

    output = torch.permute(output, [0, 2, 1, 3])
    return output, None


def _aiter_mixed_attn_call(
    query, key, value, qk_format, v_format, dropout_p, is_causal
):
    _validate_aiter_mha_v4_request(dropout_p, is_causal)
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    output = _aiter_mha_v4(
        query,
        key,
        value,
        qk_format,
        qk_format,
        v_format,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, None


@register_attention_function(AttentionBackendType.AITER_I8FP8)
def _aiter_i8fp8_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER INT8 Q/K and FP8 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.INT8,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP8)
def _aiter_mxfp8_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER MXFP8 Q/K and per-tensor FP8 V recipe."""
    _validate_aiter_mha_v4_request(dropout_p, is_causal)
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output = _aiter_mha_v4_mxfp8(query, key, value)
    return torch.permute(output, [0, 2, 1, 3]), None


@register_attention_function(AttentionBackendType.AITER_F8F6)
def _aiter_f8f6_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER per-tensor FP8 Q/K and MXFP6 V recipe."""
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        fp8_format,
        _AiterAttentionFormat.MXFP6,
        dropout_p,
        is_causal,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP4)
def _aiter_mxfp4_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER MXFP4 Q/K and FP8 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP4,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
    )


@register_attention_function(AttentionBackendType.AITER_F4F4)
def _aiter_f4f4_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER MXFP4 Q/K/V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        dropout_p,
        is_causal,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP6)
def _aiter_mxfp6_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER MXFP6 Q/K and FP8 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
    )


@register_attention_function(AttentionBackendType.AITER_F6F4)
def _aiter_f6f4_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run the AITER MXFP6 Q/K and MXFP4 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        dropout_p,
        is_causal,
    )


def _validate_aiter_mha_v4_sparge_request(
    query,
    key,
    value,
    dropout_p,
    is_causal,
    attention_kwargs,
    *,
    qk_format=None,
    v_format=None,
    mxfp8=False,
):
    _validate_aiter_mha_v4_request(dropout_p, is_causal)
    if not _AITER_MHA_V4.block_mask:
        raise RuntimeError(
            "MHA v4 Sparge requires an AITER build whose mha_v4 accepts block_mask"
        )
    if not _AITER_MHA_V4.enabled:
        raise RuntimeError("MHA v4 Sparge attention requires gfx950 or gfx942")
    if _AITER_MHA_V4.is_gfx942 and (
        mxfp8
        or qk_format not in (
            _aiter_native_fp8_format(),
            _AiterAttentionFormat.INT8,
        )
        or v_format != _aiter_native_fp8_format()
    ):
        raise NotImplementedError(
            "MHA v4 Sparge on gfx942 currently supports native FP8/FP8 and INT8/FP8 only"
        )
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        raise NotImplementedError("MHA v4 Sparge currently supports head dimension 128 only")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise NotImplementedError("MHA v4 Sparge currently supports MHA only")
    if (attention_kwargs or {}).get("indices_k") is not None:
        raise NotImplementedError("MHA v4 Sparge does not support varlen packed keys")


def _aiter_launch_mxfp8_sparse(query, key, value, block_mask):
    try:
        lut_fn = block_attn_mask_to_ragged_lut
    except NameError as exc:
        raise RuntimeError(
            "MHA v4 MXFP8 Sparge requires AITER block_attn_mask_to_ragged_lut"
        ) from exc
    lut = lut_fn(block_mask, return_none_if_dense=False)
    if lut is None:
        raise RuntimeError("block_attn_mask_to_ragged_lut returned None")
    kv_block_indices, lut_start, lut_count = lut
    softmax_scale = query.shape[-1] ** -0.5
    query_q, query_scale = _aiter_quantize_mxfp8_q(
        query, _aiter_mha_v4_q_multiplier(softmax_scale)
    )
    key_q, key_scale = _aiter_quantize_mxfp8_k(key)
    value_q, value_scale = _aiter_quantize_fp8(value)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        query_q,
        key_q,
        value_q,
        query_scale,
        key_scale,
        value_scale,
        fp8_format,
        fp8_format,
        fp8_format,
        _AiterAttentionScaleMode.E8M0_PER_1X32,
        _AiterAttentionScaleMode.E8M0_PER_1X32,
        _AiterAttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=softmax_scale,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
    )


def _aiter_mha_v4_sparge_call(
    query,
    key,
    value,
    qk_format,
    v_format,
    dropout_p,
    is_causal,
    attention_kwargs=None,
    *,
    mxfp8=False,
):
    """Build a Sparge mask at the MHA v4 sparse tile and run the matching sparse row."""
    _validate_aiter_mha_v4_sparge_request(
        query,
        key,
        value,
        dropout_p,
        is_causal,
        attention_kwargs,
        qk_format=qk_format,
        v_format=v_format,
        mxfp8=mxfp8,
    )
    q, k, v, state, block_mask, _ = _build_sparge_block_mask(
        query,
        key,
        value,
        is_causal,
        attention_kwargs,
        _mha_v4_sparge_tile(),
        pad_block_divisible=True,
    )
    q = torch.permute(q, [0, 2, 1, 3]).contiguous()
    k = torch.permute(k, [0, 2, 1, 3]).contiguous()
    v = torch.permute(v, [0, 2, 1, 3]).contiguous()
    if mxfp8:
        if _AITER_MHA_V4.mxfp8_block_mask:
            output = _aiter_mha_v4_mxfp8(q, k, v, block_mask=block_mask)
        else:
            output = _aiter_launch_mxfp8_sparse(q, k, v, block_mask)
    else:
        output = _aiter_mha_v4(
            q,
            k,
            v,
            qk_format,
            qk_format,
            v_format,
            block_mask=block_mask,
        )
    output = torch.permute(output, [0, 2, 1, 3])
    return restore_sparge_output(output, state), None


@register_attention_function(AttentionBackendType.AITER_I8FP8_SPARGE)
def _aiter_i8fp8_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER INT8 Q/K and FP8 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _AiterAttentionFormat.INT8,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_FP8_SPARGE)
def _aiter_fp8_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER per-tensor FP8 MHA v4 row."""
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        fp8_format,
        fp8_format,
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP8_SPARGE)
def _aiter_mxfp8_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER MXFP8 Q/K and per-tensor FP8 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        None,
        None,
        dropout_p,
        is_causal,
        attention_kwargs,
        mxfp8=True,
    )


@register_attention_function(AttentionBackendType.AITER_F8F6_SPARGE)
def _aiter_f8f6_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER per-tensor FP8 Q/K and MXFP6 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _aiter_native_fp8_format(),
        _AiterAttentionFormat.MXFP6,
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP4_SPARGE)
def _aiter_mxfp4_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER MXFP4 Q/K and FP8 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP4,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_F4F4_SPARGE)
def _aiter_f4f4_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER MXFP4 Q/K/V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_MXFP6_SPARGE)
def _aiter_mxfp6_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER MXFP6 Q/K and FP8 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _aiter_native_fp8_format(),
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER_F6F4_SPARGE)
def _aiter_f6f4_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """Run Sparge + the AITER MXFP6 Q/K and MXFP4 V MHA v4 row."""
    return _aiter_mha_v4_sparge_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        dropout_p,
        is_causal,
        attention_kwargs,
    )


@register_attention_function(AttentionBackendType.AITER)
def _aiter_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key   = torch.permute(key,   [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    packed = _varlen_pack_keys(query, key, value, attention_kwargs)

    if packed is not None:
        q_flat, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, B, S, H, D = packed
        varlen_kwargs = {
            "softmax_scale": D ** -0.5,
            "dropout_p": dropout_p,
            "causal": is_causal,
            "return_lse": True,
            "return_attn_probs": False,
        }
        if AITER_HAS_ROUND_MODE:
            varlen_kwargs["how_v3_bf16_cvt"] = HOW_V3_BF16_CVT
        output, softmax_lse = flash_attn_varlen_func_aiter(
            q_flat, k_packed, v_packed,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=S, max_seqlen_k=max_seqlen_k,
            **varlen_kwargs,
        )
        output = output.reshape(B, S, H, D)
        output = torch.permute(output, [0, 2, 1, 3])

    else:
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


@register_attention_function(AttentionBackendType.AITER_VSA)
@torch.compiler.disable
def _aiter_vsa_attn_call(
    query,
    key,
    value,
    dropout_p,
    is_causal,
    attention_kwargs=None,
):
    """Jenga mask selection backed by AITER's CK-Tile VSA kernel.

    VSA is a non-causal self-attention backend. Calls without Wan's ``thw``
    metadata, including text/image cross-attention, use dense AITER attention.
    """
    attention_kwargs = attention_kwargs or {}
    thw = attention_kwargs.get("thw")
    is_self_attention = query.shape == key.shape == value.shape
    if thw is None or not is_self_attention:
        return _aiter_attn_call(
            query,
            key,
            value,
            dropout_p,
            is_causal,
            attention_kwargs,
        )
    if is_causal:
        raise ValueError("AITER VSA CK does not support causal attention")
    if dropout_p not in (None, 0.0):
        raise ValueError("AITER VSA CK does not support attention dropout")

    from xfuser.core.vsa_attention import (
        aiter_vsa_attention,
        jenga_scheduled_drop_rate,
    )

    drop_rate = None
    drop_rates = attention_kwargs.get("vsa_drop_rates")
    if drop_rates:
        drop_rate = attention_kwargs.get("vsa_effective_drop_rate")
        if drop_rate is None:
            drop_rate = jenga_scheduled_drop_rate(
                int(attention_kwargs.get("vsa_step_index", 0)),
                int(attention_kwargs.get("vsa_num_steps", 1)),
                drop_rates,
            )
        use_dense = bool(
            attention_kwargs.get("vsa_use_dense", drop_rate <= 0.25)
        )
        attention_kwargs["vsa_effective_drop_rate"] = drop_rate
        attention_kwargs["vsa_use_dense"] = use_dense
        if use_dense:
            return _aiter_attn_call(
                query,
                key,
                value,
                dropout_p,
                is_causal,
                attention_kwargs,
            )

    collect_density = bool(attention_kwargs.get("vsa_collect_density", False))
    output, density = aiter_vsa_attention(
        query,
        key,
        value,
        thw=tuple(thw),
        sp_size=get_ulysses_parallel_world_size(),
        block_size=int(attention_kwargs.get("vsa_block_size", 128)),
        top_k=int(attention_kwargs.get("vsa_top_k", 1)),
        top_k_ratio=float(attention_kwargs.get("vsa_top_k_ratio", 0.0)),
        drop_rate=drop_rate,
        prob_threshold=float(
            attention_kwargs.get("vsa_prob_threshold", 0.9)
        ),
        reorder_sequence=bool(
            attention_kwargs.get("vsa_reorder_sequence", True)
        ),
        use_static_block_mask=bool(
            attention_kwargs.get("use_vsa_static_block_mask", True)
        ),
        use_first_frame_mask=bool(
            attention_kwargs.get("use_vsa_first_frame_mask", True)
        ),
        collect_density=collect_density,
    )
    if density is not None:
        attention_kwargs["vsa_last_density"] = density.detach()
    return output, None

@register_attention_function(AttentionBackendType.AITER_MLA)
def _aiter_mla_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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
def _flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn
    """
    query = torch.permute(query, [0, 2, 1, 3])
    key = torch.permute(key, [0, 2, 1, 3])
    value = torch.permute(value, [0, 2, 1, 3])
    packed = _varlen_pack_keys(query, key, value, attention_kwargs)
    if packed is not None:
        q_flat, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, B, S, H, D = packed
        output = flash_attn_varlen_func_2(
            q_flat, k_packed, v_packed,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=S, max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p, softmax_scale=D ** -0.5,
            causal=is_causal,
        )
        softmax_lse = None
        output = output.reshape(B, S, H, D)
    else:
        output, softmax_lse, _ = flash_attn_func_2(
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
    if AITER_SAGE_SUPPORTS_RING and get_ring_parallel_world_size() > 1:
        attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd", return_lse=True, smooth_k=True)
        output, softmax_lse = attn_fn(query, key, value)
    else:
        attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd")
        output = attn_fn(query, key, value)
        softmax_lse = None
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER_SAGE_V2)
def _aiter_sage_v2_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    # Contiguous is needed for Sage v2 in older AITER versions.
    # This has been fixed in newer version of AITER, meaning the
    # contiguous calls can be removed in the future.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if AITER_SAGE_V2_SUPPORTS_RING and get_ring_parallel_world_size() > 1:
        attn_fn = functools.partial(fav3_sage_mxfp4_wrapper, layout="bhsd", hadamard_rotation=True, R=HADAMARD_MATRIX[query.device], return_lse=True)
        output, softmax_lse = attn_fn(query, key, value, causal=is_causal)
    else:
        attn_fn = functools.partial(fav3_sage_mxfp4_wrapper, layout="bhsd", hadamard_rotation=True, R=HADAMARD_MATRIX[query.device])
        output = attn_fn(query, key, value, causal=is_causal)
        softmax_lse = None
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

@torch.library.custom_op("xfuser::flex_block_attn", mutates_args=())
def flex_block_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int,
    block_n: int,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    return flex_block_attn_func(q, k, v, block_m, block_n, block_mask)

@flex_block_attn_op.register_fake
def _(q, k, v, block_m, block_n, block_mask):
    return torch.empty_like(q)

@register_attention_function(AttentionBackendType.FLEX_BLOCK_ATTN)
def _flex_block_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    attention_kwargs["sp_size"] = get_ulysses_parallel_world_size()
    block_size = math.prod(attention_kwargs["tile_size"])
    q, k, v, mask_config, ssta_state = setup_ssta(query, key, value, attention_kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=attention_kwargs["attn_sparse_type"])
    output = flex_block_attn_op(q, k, v, block_size, block_size, block_mask)
    output = untile_ssta_output(output, ssta_state, attention_kwargs["encoder_sequence_length"], attention_kwargs["sp_size"])
    return output, None

@register_attention_function(AttentionBackendType.AITER_SPARSE_SAGE)
def _aiter_sparse_sage_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    attention_kwargs["sp_size"] = get_ulysses_parallel_world_size()
    block_size = math.prod(attention_kwargs["tile_size"])
    config = get_sage_fwd_configs()
    config["BLOCK_M"] = _TRITON_SSTA_BLOCK_SIZE
    config["BLOCK_N"] = _TRITON_SSTA_BLOCK_SIZE
    attn_fn = functools.partial(fav3_sage_wrapper_func, layout="bhsd", config=config)
    q, k, v, mask_config, ssta_state = setup_ssta(query, key, value, attention_kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=attention_kwargs["attn_sparse_type"])
    if block_size != _TRITON_SSTA_BLOCK_SIZE:
        block_mask = expand_block_mask(block_mask, factor=block_size // _TRITON_SSTA_BLOCK_SIZE)
    block_lut = block_attn_mask_to_ragged_lut(block_mask, num_heads=q.shape[1])
    output = attn_fn(q, k, v, block_lut=block_lut)
    output = untile_ssta_output(output, ssta_state, attention_kwargs["encoder_sequence_length"], attention_kwargs["sp_size"])
    return output, None


@register_attention_function(AttentionBackendType.AITER_SPARSE_SAGE_V2)
def _aiter_sparse_sage_v2_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    attention_kwargs["sp_size"] = get_ulysses_parallel_world_size()
    block_size = math.prod(attention_kwargs["tile_size"])
    config = get_sage_fwd_configs_mxfp4()
    config["BLOCK_M"] = _TRITON_SSTA_BLOCK_SIZE
    config["BLOCK_N"] = _TRITON_SSTA_BLOCK_SIZE
    attn_fn = functools.partial(fav3_sage_mxfp4_wrapper, layout="bhsd", hadamard_rotation=True, R=HADAMARD_MATRIX[query.device], config=config)
    q, k, v, mask_config, ssta_state = setup_ssta(query, key, value, attention_kwargs)
    block_mask = get_sparse_mask(mask_config, sparse_type=attention_kwargs["attn_sparse_type"])
    if block_size != _TRITON_SSTA_BLOCK_SIZE:
        block_mask = expand_block_mask(block_mask, factor=block_size // _TRITON_SSTA_BLOCK_SIZE)
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
def _nvte_fp8_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
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


def _read_sparge_kwargs(attention_kwargs):
    attn_kwargs = attention_kwargs or {}
    simthreshd1 = attn_kwargs.get("spargeattn_simthreshold",0.3)
    cdfthreshd = attn_kwargs.get("spargeattn_cdfthreshold", 0.92)
    return (
        simthreshd1,
        cdfthreshd,
        attn_kwargs.get("spargeattn_reorder_sequence", True),
        attn_kwargs.get("use_spargeattn_static_block_mask", True),
        attn_kwargs.get("thw"),
        attn_kwargs.get("encoder_sequence_length", 0),
    )

def _build_sparge_block_mask(query, key, value, is_causal, attention_kwargs, config, pad_block_divisible=False):
    simthreshd1, cdfthreshd, reorder, use_static, thw, esl = _read_sparge_kwargs(attention_kwargs)
    q, k, v, state, static_mask = setup_sparge(
        query, key, value,
        thw=thw,
        sp_size=get_ulysses_parallel_world_size(),
        encoder_sequence_length=esl,
        reorder_sequence=reorder,
        use_static_block_mask=use_static,
        block_m=config["BLOCK_M"], block_n=config["BLOCK_N"],
        pad_block_divisible=pad_block_divisible,
    )
    block_mask = compute_sparge_block_mask(
        q, k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        is_causal=is_causal,
        static_block_mask=static_mask,
        text_len=state.text_len + state.tail_pad,
        block_m=config["BLOCK_M"], block_n=config["BLOCK_N"],
    )
    block_mask = mask_padded_kv_blocks(block_mask, state, config["BLOCK_N"])
    num_heads = q.shape[1]
    # Per-head selected-block cost for the Ulysses head-balancer. USP injects a
    # scratch "cost sink" tensor into attention_kwargs only when balancing is
    # active.
    cost_sink = (attention_kwargs or {}).get(COST_SINK_KEY)
    if cost_sink is not None:
        cost_sink.copy_(block_mask.to(torch.float32).sum(dim=(0, 2, 3)))
    return q, k, v, state, block_mask, num_heads

@register_attention_function(AttentionBackendType.AITER_SPARGE)
def _aiter_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    config = get_sage_fwd_configs()
    q, k, v, state, block_mask, num_heads = _build_sparge_block_mask(query, key, value, is_causal, attention_kwargs, config)
    block_lut = block_attn_mask_to_ragged_lut(block_mask, num_heads=num_heads)
    output = fav3_sage_wrapper_func(
        q, k, v, block_lut=block_lut, layout="bhsd", config=config,
    )
    return restore_sparge_output(output, state), None


@register_attention_function(AttentionBackendType.AITER_SPARGE_V2)
def _aiter_sparge_v2_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    config = get_sage_fwd_configs_mxfp4()
    q, k, v, state, block_mask, num_heads = _build_sparge_block_mask(query, key, value, is_causal, attention_kwargs, config)
    block_lut = block_attn_mask_to_ragged_lut(block_mask, num_heads=num_heads)
    output = fav3_sage_mxfp4_wrapper(
        q, k, v, causal=is_causal, block_lut=block_lut,
        layout="bhsd", hadamard_rotation=True,
        R=HADAMARD_MATRIX[query.device], config=config,
    )
    return restore_sparge_output(output, state), None

@register_attention_function(AttentionBackendType.FLEX_BLOCK_SPARGE)
def _flex_block_sparge_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    config = {"BLOCK_M": 256, "BLOCK_N": 256}
    q, k, v, state, block_mask, num_heads = _build_sparge_block_mask(
        query, key, value, is_causal, attention_kwargs, config, pad_block_divisible=True
    )
    output = flex_block_attn_op(
        q.contiguous(), k.contiguous(), v.contiguous(),
        config["BLOCK_M"], config["BLOCK_N"], block_mask,
    )
    return restore_sparge_output(output, state), None

def _aiter_flydsl_dispatch(query, key, value, dropout_p, is_causal, attention_kwargs, attn_op):
    # Layout here is [B, H, S, D]. Self-attn and non-causal cross-attn both hit the
    # kernel: causal masks padded cols via col > q_row; non-causal masks them via
    # seq_len_real (aligned) or the tail mask (unaligned); cross-attn loads K/V on
    # their own length. Anything the kernel would reject with a ValueError takes SDPA
    # instead: causal-cross (ambiguous alignment; never occurs in diffusion), GQA or
    # head_dim mismatches (the kernel is single-NUM_HEADS MHA), and head_dim outside
    # its >=64, %32==0 tile constraint.
    is_cross = query.shape[2] != key.shape[2]
    head_dim = query.shape[3]
    if (
        (is_cross and is_causal)
        or query.shape[1] != key.shape[1]
        or head_dim != key.shape[3]
        or head_dim < 64
        or head_dim % 32 != 0
    ):
        return _sdpa_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs)
    if not AITER_FLYDSL_GENERALIZED:
        if is_cross:
            return _sdpa_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs)
        if not is_causal:
            seq_len = query.shape[2]
            pad = (-seq_len) % 128
            # 199*pad > seq_len is pad/(seq_len+pad) > 0.005 (0.5%) in integer arithmetic.
            if pad > 0 and 199 * pad > seq_len:
                return _sdpa_flash_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs)
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output = attn_op(query, key, value, is_causal)
    output = torch.permute(output, [0, 2, 1, 3])
    return output, None


@register_attention_function(AttentionBackendType.AITER_FLYDSL)
def _aiter_flydsl_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    return _aiter_flydsl_dispatch(
        query, key, value, dropout_p, is_causal, attention_kwargs, torch.ops.xfuser.flydsl_attn
    )


@register_attention_function(AttentionBackendType.AITER_FLYDSL_FP8)
def _aiter_flydsl_fp8_attn_call(query, key, value, dropout_p, is_causal, attention_kwargs=None):
    return _aiter_flydsl_dispatch(
        query, key, value, dropout_p, is_causal, attention_kwargs, torch.ops.xfuser.flydsl_attn_fp8
    )
