# This file implements USP with torch version >= '2.5.0'
import functools
import inspect
from typing import Tuple

import torch
import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    model_parallel_is_initialized,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_ulysses_parallel_rank,
    get_runtime_state,
)

from xfuser.compat import version_at_least
from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.logger import init_logger
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import VarlenPacking
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.fp8_comms import (
    fp8_attention_kwargs,
    fp8_comms_input_all_to_all,
    fp8_comms_output_all_to_all,
    fp8_observe_output,
)
from xfuser.core.sparge_attention.head_balance import (
    apply_head_balance,
    revert_head_balance,
)
from xfuser.model_executor.layers.fused_a2a_integration import (
    fused_a2a_input,
    fused_a2a_input_role,
    get_fused_a2a_codecs,
    get_fused_a2a_mode,
    get_fused_a2a_profile,
    launch_attention_a2a_packed,
    use_fused_a2a_interleave,
    use_fused_a2a_packed,
)

# Sparge backends whose kernel cost can be load-balanced across Ulysses ranks.
# These all build a block mask via _build_sparge_block_mask and write the
# per-head cost into the head-balance "cost sink". Non-sparge backends are
# excluded so head balancing is a clean no-op for them.
_HEAD_BALANCE_BACKENDS = attention_registry.types_where(head_balanced=True)

_FP8_NCCL_NEEDS_VIEW = not version_at_least(torch.__version__, "2.11.0")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
_warned_fp8_comms_missing_attn = False
logger = init_logger(__name__)
_CUSTOM_OP_OPTIONS = (
    {"tags": (torch.Tag.cudagraph_unsafe,)}
    if hasattr(torch, "Tag") and "tags" in inspect.signature(torch.library.custom_op).parameters
    else {}
)


def _warn_fp8_comms_missing_attn():
    global _warned_fp8_comms_missing_attn
    if _warned_fp8_comms_missing_attn:
        return
    runtime_state = get_runtime_state()
    if runtime_state.fp8_comms is None or get_ulysses_parallel_world_size() <= 1:
        return
    _warned_fp8_comms_missing_attn = True
    logger.warning(
        "use_fp8_comms is enabled but USP was called without attn_layer "
        "(or a module with fp8 scale buffers). FP8 all-to-all will not run on this path."
    )


# A backend that needs per-head tensors of its own alongside query/key/value
# lists their ``attention_kwargs`` keys under this one, and USP carries them
# through the same Ulysses exchange without knowing what they mean.
ULYSSES_EXTRA_INPUTS_KEY = "ulysses_extra_inputs"


def ring_attn(
    attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None
):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "joint_attn_kwargs": joint_attn_kwargs,
        "attention_kwargs": attention_kwargs,
    }
    if version_at_least(torch.__version__, "2.6.0"):
        from torch.distributed.tensor.experimental._attention import _cp_options

        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.flatten()
    # NCCL does not support FP8 collectives before PyTorch 2.11, view as uint8 (same width) for the transfer.
    if _FP8_NCCL_NEEDS_VIEW and x_dtype in _FP8_DTYPES:
        x = x.view(torch.uint8)
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.view(x_dtype).reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _combined_qkv_all_to_all(q, k, v, *extra):
    """Concatenate query, key, value tensors and perform a single all-to-all communication.

    Extra tensors shaped like q ride the same exchange and are returned after v.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return (q, k, v, *extra)

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    n = 3 + len(extra)
    # [n, b, h, s, d]
    qkv = torch.stack([q, k, v, *extra], dim=0)
    # [n, b, P, h/P, s, d]
    qkv = qkv.view(n, b, world_size, h // world_size, s, d)
    # [P, n, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [n, b, h/P, P*s, d]  — reshape directly avoids the intermediate
    # contiguous copy that the separate permute+view required.
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(n, b, h // world_size, -1, d)

    return torch.unbind(qkv, dim=0)


def _combined_gqa_qkv_all_to_all(q, k, v, *extra):
    """Exchange GQA Q/K/V in one collective without expanding KV heads.

    Every destination receives its contiguous query-head shard and the
    corresponding compact KV-head shard from every source rank. Query-shaped
    extra tensors can share the same collective.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return (q, k, v, *extra)

    tensors = (q, k, v, *extra)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("GQA all-to-all inputs must all have four dimensions.")
    if k.shape != v.shape:
        raise ValueError(
            f"GQA key and value tensors must have identical shapes, got {tuple(k.shape)} and {tuple(v.shape)}."
        )

    batch_size, _, _, head_dim = q.shape
    if any(tensor.shape[0] != batch_size or tensor.shape[-1] != head_dim for tensor in tensors):
        raise ValueError("GQA all-to-all inputs must share batch and head dimensions.")
    if any(tensor.shape[1] % world_size != 0 for tensor in tensors):
        raise ValueError("Every GQA head count must be divisible by the Ulysses world size.")
    if any(tensor.shape != q.shape for tensor in extra):
        raise ValueError("Extra GQA all-to-all inputs must match the query shape.")

    packed_chunks = []
    metadata = []
    for tensor in tensors:
        _, heads, sequence_length, tensor_head_dim = tensor.shape
        local_heads = heads // world_size
        # Match _ft_c_input_all_to_all's destination-major layout, then pack
        # unequal Q and KV payloads into one equally split collective.
        packed_chunks.append(tensor.permute(1, 0, 2, 3).contiguous().reshape(world_size, -1))
        metadata.append((local_heads, sequence_length, tensor_head_dim))

    chunk_sizes = [chunk.shape[1] for chunk in packed_chunks]
    exchanged = _sdpa_all_to_all_single(torch.cat(packed_chunks, dim=1))

    outputs = []
    for chunk, (local_heads, sequence_length, tensor_head_dim) in zip(exchanged.split(chunk_sizes, dim=1), metadata):
        outputs.append(
            chunk.view(
                world_size,
                local_heads,
                batch_size,
                sequence_length,
                tensor_head_dim,
            )
            .permute(2, 1, 0, 3, 4)
            .reshape(batch_size, local_heads, -1, tensor_head_dim)
        )
    return tuple(outputs)


def _repeat_kv_heads(key, value, repeats):
    if repeats == 1:
        return key, value
    return (
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
    )


def _validate_gqa_params(
    query,
    key,
    value,
    kv_head_repeat,
    joint_strategy,
):
    if isinstance(kv_head_repeat, bool) or not isinstance(kv_head_repeat, int):
        raise TypeError("kv_head_repeat must be an integer.")
    if kv_head_repeat < 1:
        raise ValueError("kv_head_repeat must be at least 1.")
    if kv_head_repeat == 1:
        return

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("GQA query, key, and value must have four dimensions.")
    if key.shape[1] != value.shape[1]:
        raise ValueError("GQA key and value must have the same head count.")
    if query.shape[1] != key.shape[1] * kv_head_repeat:
        raise ValueError(
            f"Query heads ({query.shape[1]}) must equal KV heads "
            f"({key.shape[1]}) times kv_head_repeat ({kv_head_repeat})."
        )

    ulysses_world_size = get_ulysses_parallel_world_size()
    if ulysses_world_size > 1 and key.shape[1] % ulysses_world_size != 0:
        raise ValueError(
            f"KV heads ({key.shape[1]}) must be divisible by the Ulysses world size ({ulysses_world_size})."
        )
    if joint_strategy is not None:
        raise NotImplementedError("GQA KV repetition does not support joint tensors.")


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = joint_key.shape[1] // ulysses_world_size
    joint_key = joint_key.transpose(1, 2)
    joint_value = joint_value.transpose(1, 2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
        :,
    ].transpose(1, 2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
        :,
    ].transpose(1, 2)
    return joint_key, joint_value


def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor


def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value


def _has_kv_cache(attn_layer) -> bool:
    """Return whether PipeFusion registered a KV cache for this attention layer."""
    return attn_layer is not None and get_cache_manager().has_cache_entry(attn_layer)


def _serves_packed_keys(backend, key, attention_kwargs) -> bool:
    """Whether the selected backend will honour the packing itself."""
    spec = attention_registry.find(backend)
    if spec is None:
        return False
    probe = AttnCall(
        varlen=VarlenPacking.from_kwargs(attention_kwargs),
        attention_kwargs=attention_kwargs,
    )
    return spec.rejects(key, key, key, probe) is None


def _trim_trailing_kv_padding(key, value, attention_kwargs, backend=None):
    """Slice a uniform padded K/V suffix while retaining every query row.

    Returns the kwargs the backend should see: once the pad has been sliced
    the packing has been honoured, so it is cleared -- left in place, it would
    describe keys that are no longer there. Keys are set to None rather than
    removed, because torch.compile guards on this dict's key set.
    """
    kwargs = attention_kwargs or {}
    valid_kv_len = kwargs.get("valid_kv_len")
    if valid_kv_len is None:
        return key, value, attention_kwargs
    if kwargs.get("indices_k") is not None and _serves_packed_keys(backend, key, kwargs):
        # A backend that accepts packed keys handles the pad itself, and
        # slicing first would leave its indices pointing past the end of K.
        return key, value, attention_kwargs
    if not 0 < valid_kv_len <= key.shape[2]:
        raise ValueError(f"valid_kv_len must be in [1, {key.shape[2]}], got {valid_kv_len}.")

    consumed = kwargs
    if kwargs.get("indices_k") is not None:
        consumed = dict(kwargs)
        consumed["indices_k"] = None
        consumed["cu_seqlens_k"] = None
        consumed["max_seqlen_k"] = None
    return key[:, :, :valid_kv_len], value[:, :, :valid_kv_len], consumed


def _allocate_compact_attention_a2a_kv(
    key,
    value,
    k_scales,
    v_scales,
    valid_kv_len,
    *,
    profile,
    copy_values,
):
    """Fallback rebuild for tiled K/V that AITER cannot emit compactly.

    The ordinary USP path delegates trailing padding and arbitrary varlen keys
    to the backend Spec. Pre-quantized A2A operands are different: MXFP4/MXFP6
    K and FP6-P V encode the sequence tile count in their head strides, so a
    slice that removes a whole tile would no longer satisfy MHA-v4's packed
    layout. AITER emits direct compact K/V when both roles have token-local
    scales; this copy remains for recipes with a sequence-wide K or V scale.
    """
    from aiter.ops.mha_v4_quant import (
        MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
        MHA_V4_KV_TILE_ROWS,
        MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_TILE_BYTES,
        block_scale_storage,
        mxfp4_k_raw_buffer_size,
        mxfp4_k_view,
        mxfp4_v_raw_buffer_size,
        mxfp4_v_view,
        mxfp6_v_raw_buffer_size,
    )
    from aiter.ops.triton.quant.mxfp6_fmha_pack import (
        FP6_K_TILE_BYTES,
        fp6_k_lds_order_views_from_raw,
        fp6_k_raw_buffer_sizes,
    )

    b, source_sequence, heads, _ = key.shape
    config = AttentionA2AConfig(profile=profile)
    qk_codec, _, v_codec = config.codecs
    blocks = 4
    source_tiles = (source_sequence + 127) // 128
    target_tiles = (valid_kv_len + 127) // 128

    def flat_storage(tensor):
        return tensor.as_strided(
            (tensor.untyped_storage().nbytes() // tensor.element_size(),),
            (1,),
            0,
        )

    def copy_tiles(source, target, tile_bytes):
        source_main = flat_storage(source)[: b * heads * source_tiles * tile_bytes]
        target_main = target[: b * heads * target_tiles * tile_bytes]
        target_main.view(b, heads, target_tiles, tile_bytes).copy_(
            source_main.view(b, heads, source_tiles, tile_bytes)[:, :, :target_tiles]
        )

    def empty_with_zeroed_slack(reference, size, main_size):
        storage = reference.new_empty((size,))
        if size > main_size:
            storage[main_size:].zero_()
        return storage

    if qk_codec == "mxfp4":
        k_main_size = b * heads * target_tiles * 8192
        k_raw = empty_with_zeroed_slack(
            key,
            mxfp4_k_raw_buffer_size(b, valid_kv_len, heads),
            k_main_size,
        )
        compact_k_scales = block_scale_storage(
            key,
            b,
            valid_kv_len,
            heads,
            blocks,
            MHA_V4_KV_TILE_ROWS,
            lookahead_rows=MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
            extra=MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
        )
        compact_key = mxfp4_k_view(k_raw, compact_k_scales)
        if copy_values:
            copy_tiles(key, k_raw, 8192)
            compact_k_scales.copy_(k_scales[:, :valid_kv_len])
    elif qk_codec == "mxfp6":
        k_size, k_scale_size = fp6_k_raw_buffer_sizes(b, valid_kv_len, heads)
        k_raw = empty_with_zeroed_slack(
            key,
            k_size,
            b * heads * target_tiles * FP6_K_TILE_BYTES,
        )
        k_scale_raw = empty_with_zeroed_slack(
            k_scales,
            k_scale_size,
            b * valid_kv_len * heads * blocks,
        )
        compact_key, compact_k_scales = fp6_k_lds_order_views_from_raw(
            k_raw,
            k_scale_raw,
            b,
            valid_kv_len,
            heads,
        )
        if copy_values:
            copy_tiles(key, k_raw, FP6_K_TILE_BYTES)
            compact_k_scales.copy_(k_scales[:, :valid_kv_len])
    else:
        compact_key = key.new_empty((b, valid_kv_len, heads, key.shape[-1]))
        compact_k_scales = k_scales.new_empty(k_scales.shape)
        if copy_values:
            compact_key.copy_(key[:, :valid_kv_len])
            compact_k_scales.copy_(k_scales)

    if v_codec == "mxfp4":
        v_main_size = b * heads * target_tiles * 8192
        v_raw = empty_with_zeroed_slack(
            value,
            mxfp4_v_raw_buffer_size(b, valid_kv_len, heads),
            v_main_size,
        )
        scale_elements = b * heads * target_tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
        scale_storage = empty_with_zeroed_slack(
            v_scales,
            scale_elements + MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
            scale_elements,
        )
        compact_v_scales = scale_storage[:scale_elements].view(
            b,
            heads,
            target_tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES,
        )
        compact_value = mxfp4_v_view(v_raw, compact_v_scales, valid_kv_len)
        v_tile_bytes = 8192
    elif v_codec == "mxfp6_p":
        v_main_size = b * heads * target_tiles * 12288
        v_raw = empty_with_zeroed_slack(
            value,
            mxfp6_v_raw_buffer_size(b, valid_kv_len, heads),
            v_main_size,
        )
        compact_v_scales = v_scales.new_empty((b, heads, target_tiles * 512))
        compact_value = torch.as_strided(
            v_raw,
            (b, valid_kv_len, heads, 128),
            (heads * target_tiles * 12288, 96, target_tiles * 12288, 1),
        )
        v_tile_bytes = 12288
    else:
        compact_value = value.new_empty((b, valid_kv_len, heads, value.shape[-1]))
        compact_v_scales = v_scales.new_empty(v_scales.shape)
        v_tile_bytes = None

    if copy_values:
        if v_tile_bytes is None:
            compact_value.copy_(value[:, :valid_kv_len])
            compact_v_scales.copy_(v_scales)
        else:
            copy_tiles(value, v_raw, v_tile_bytes)
            compact_v_scales.copy_(v_scales[:, :, : target_tiles * 512])

    return compact_key, compact_value, compact_k_scales, compact_v_scales


@torch.library.custom_op(
    "xfuser::compact_attention_a2a_tiled_kv",
    mutates_args=(),
    **_CUSTOM_OP_OPTIONS,
)
def _compact_attention_a2a_tiled_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    valid_kv_len: int,
    profile: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _allocate_compact_attention_a2a_kv(
        key,
        value,
        k_scales,
        v_scales,
        valid_kv_len,
        profile=profile,
        copy_values=True,
    )


@_compact_attention_a2a_tiled_kv.register_fake
def _compact_attention_a2a_tiled_kv_fake(
    key,
    value,
    k_scales,
    v_scales,
    valid_kv_len,
    profile,
):
    return _allocate_compact_attention_a2a_kv(
        key,
        value,
        k_scales,
        v_scales,
        valid_kv_len,
        profile=profile,
        copy_values=False,
    )


def _trim_packed_attention_a2a_padding(
    key,
    value,
    packed_scales,
    valid_kv_len,
    codecs,
    profile,
):
    """Consume Wan's uniform trailing pad without reimplementing varlen packing."""
    if valid_kv_len is None:
        return key, value, packed_scales
    source_sequence = key.shape[1]
    if value.shape[1] != source_sequence:
        raise ValueError("packed Attention A2A K/V sequence lengths must match")
    if not 0 < valid_kv_len <= source_sequence:
        raise ValueError(f"valid_kv_len must be in [1, {source_sequence}], got {valid_kv_len}.")
    if valid_kv_len == source_sequence:
        return key, value, packed_scales

    qk_codec, _, v_codec = codecs
    source_tiles = (source_sequence + 127) // 128
    target_tiles = (valid_kv_len + 127) // 128
    tiled_layout = qk_codec in ("mxfp4", "mxfp6") or v_codec in (
        "mxfp4",
        "mxfp6_p",
    )
    if tiled_layout and source_tiles != target_tiles:
        compact_key, compact_value, k_scales, v_scales = _compact_attention_a2a_tiled_kv(
            key,
            value,
            packed_scales[1],
            packed_scales[2],
            valid_kv_len,
            profile,
        )
        return (
            compact_key,
            compact_value,
            (packed_scales[0], k_scales, v_scales),
        )

    # Plain byte layouts, and tiled layouts that retain the same tile count,
    # remain valid views. Only token-indexed K scales need the matching slice.
    key = key[:, :valid_kv_len]
    value = value[:, :valid_kv_len]
    k_scales = packed_scales[1]
    if k_scales.ndim == 4 and k_scales.shape[1] == source_sequence:
        k_scales = k_scales[:, :valid_kv_len]
    return key, value, (packed_scales[0], k_scales, packed_scales[2])


def _attention_a2a_packed_attn_call(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    profile: str,
    softmax_scale: float,
) -> torch.Tensor:
    out = launch_attention_a2a_packed(
        q_packed,
        k_packed,
        v_packed,
        q_scales,
        k_scales,
        v_scales,
        profile,
        softmax_scale,
    )
    return out.transpose(1, 2)


def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend

    spec = attention_registry.find(attention_backend)
    if spec is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(_spec_adapter(spec))


def _parallel_degrees() -> Tuple[int, int]:
    """(ulysses, ring) sequence-parallel degrees, or the single-rank default.

    `attention()` is the entry point for calls that need no sequence
    parallelism, and it is reached before -- or entirely without -- an
    initialised process group. Asking for the degrees there must not be what
    fails.
    """
    if not model_parallel_is_initialized():
        return 1, 1
    return get_ulysses_parallel_world_size(), get_ring_parallel_world_size()


def _spec_adapter(spec):
    """Present a Spec with the calling convention ring_attn and the direct
    call sites already use.

    This is also where the two things kernels used to fetch for themselves get
    supplied: the varlen packing, previously rebuilt inside every kernel from
    attention_kwargs, and the sequence-parallel degrees, previously read from
    globals. Both now arrive on the call, which is what makes a backend
    testable without an initialised process group.
    """

    def call(query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=None):
        kwargs = attention_kwargs if attention_kwargs is not None else {}
        ulysses_world_size, ring_world_size = _parallel_degrees()
        return spec.run(
            query,
            key,
            value,
            AttnCall(
                dropout_p=dropout_p,
                is_causal=is_causal,
                varlen=VarlenPacking.from_kwargs(kwargs),
                ulysses_world_size=ulysses_world_size,
                ring_world_size=ring_world_size,
                attention_kwargs=kwargs,
            ),
        )

    return call


def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1  # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, attention_kwargs=attention_kwargs)

    return wrapper


def _ulysses_extra_inputs(attention_kwargs, query):
    """Return the (name, tensor) pairs the backend asked to join the Ulysses exchange."""
    if not attention_kwargs:
        return []

    extras = []
    for name in attention_kwargs.get(ULYSSES_EXTRA_INPUTS_KEY) or ():
        tensor = attention_kwargs.get(name)
        if tensor is None:
            continue
        if tensor.shape != query.shape:
            raise ValueError(
                f"attention_kwargs['{name}'] must match the query shape to be "
                f"exchanged with it, got {tuple(tensor.shape)} vs {tuple(query.shape)}."
            )
        extras.append((name, tensor))
    return extras


def usp_fused_a2a_input_role(
    input,
    role,
    pending=None,
    valid_kv_len=None,
):
    """Submit one sequence-major Wan role to the active Ulysses A2A input hop."""
    return fused_a2a_input_role(
        input,
        role,
        PROCESS_GROUP.ULYSSES_PG,
        get_ulysses_parallel_rank(),
        pending,
        valid_kv_len,
    )


def usp_fused_a2a_input_q(input, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        0,
        valid_kv_len=valid_kv_len,
    )


def usp_fused_a2a_input_k(input, pending, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        1,
        pending,
        valid_kv_len,
    )


def usp_fused_a2a_input_v(input, pending, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        2,
        pending,
        valid_kv_len,
    )


def USP(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    joint_query: torch.Tensor | None = None,
    joint_key: torch.Tensor | None = None,
    joint_value: torch.Tensor | None = None,
    joint_strategy: str | None = None,
    attn_layer=None,
    combine_qkv_a2a: bool | None = None,
    backend=None,
    attention_kwargs: dict | None = None,
    head_balance_layer=None,
    kv_head_repeat: int = 1,
    attention_a2a_enabled: bool = False,
    attention_a2a_pending=None,
):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.

    ``attn_layer`` (optional): the attention module. Used to resolve Ulysses FP8
    communication state and to update the PipeFusion KV cache for modules that
    registered one. Callers only pass the module; USP no-ops when FP8 comms is
    off, the module has no scales, or ``joint_strategy`` is set, and it skips
    the KV cache outside PipeFusion, where no entry exists.

    ``head_balance_layer`` (optional): a stable per-layer handle (e.g. the
    attention module). When provided and --use_spargeattn_head_balance is set, the
    Ulysses head dimension is permuted so each rank gets a cost-balanced subset
    of heads (block-sparse load balancing); the permutation is inverted on the
    output. No-op for non-sparse backends (no cost is published) and for ring/
    joint paths. Also used as the FP8-comms module when ``attn_layer`` is None
    (KV cache already updated by the caller).

    ``kv_head_repeat`` keeps grouped-query attention K/V heads compact during
    the Ulysses input all-to-all, then repeats each local KV head immediately
    before attention. With ``combine_qkv_a2a=True``, unequal Q and KV payloads
    are packed into one collective.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False
    _validate_gqa_params(query, key, value, kv_head_repeat, joint_strategy)

    runtime_state = get_runtime_state()
    ulysses_world_size = get_ulysses_parallel_world_size()
    ring_world_size = get_ring_parallel_world_size()
    use_attention_a2a = attention_a2a_enabled and get_fused_a2a_mode() > 0
    if attention_a2a_pending is not None and not (use_attention_a2a and use_fused_a2a_interleave()):
        raise ValueError("interleaved input requires eligible Attention A2A transport")

    hb_backend = backend if backend is not None else runtime_state.attention_backend
    attention_a2a_profile = None
    if use_attention_a2a:
        if ulysses_world_size <= 1:
            raise NotImplementedError("Attention A2A requires Ulysses parallelism")
        if ring_world_size != 1:
            raise NotImplementedError("Attention A2A does not support ring parallelism")
        if joint_strategy is not None:
            raise NotImplementedError("Attention A2A does not support joint attention")
        if kv_head_repeat != 1:
            raise NotImplementedError("Attention A2A does not support grouped-query KV repetition")
        if runtime_state.runtime_config.use_spargeattn_head_balance:
            raise NotImplementedError("Attention A2A does not support sparse head balancing")
        if query.shape != key.shape or query.shape != value.shape:
            raise NotImplementedError("Attention A2A requires equal self-attention Q/K/V shapes")
        if not use_fused_a2a_packed():
            raise RuntimeError("Attention A2A requires packed per-role transport results")
        attention_a2a_profile = get_fused_a2a_profile()
        if attention_a2a_profile in ("none", "auto"):
            raise RuntimeError("Attention A2A runtime did not activate a concrete profile")

    # Packed Attention A2A calls MHA-v4 directly below. Resolving a registry
    # adapter here would build an unused closure in every attention block.
    attention_function = None if use_attention_a2a else _get_attention_function(backend=backend)

    fp8_module = attn_layer if attn_layer is not None else head_balance_layer
    fp8_comms = None
    if not use_attention_a2a and not joint_strategy:
        if fp8_module is None:
            _warn_fp8_comms_missing_attn()
        else:
            fp8_backend = backend if backend is not None else runtime_state.attention_backend
            fp8_comms = fp8_attention_kwargs(
                runtime_state.fp8_comms,
                fp8_module,
                query,
                key,
                value,
                False,
                fp8_backend,
            ).get("fp8_comms")

    if kv_head_repeat > 1 and fp8_comms is not None:
        raise NotImplementedError("GQA KV repetition does not support FP8 communication.")
    if use_attention_a2a and fp8_comms is not None:
        raise ValueError("Attention A2A and --use_fp8_comms are mutually exclusive")

    hb_uly = ulysses_world_size
    if use_attention_a2a:
        hb_applied = False
    else:
        query, key, value, hb_applied, attention_kwargs = apply_head_balance(
            query,
            key,
            value,
            head_balance_layer,
            enabled=(runtime_state.runtime_config.use_spargeattn_head_balance and kv_head_repeat == 1),
            ulysses_world_size=hb_uly,
            ring_world_size=ring_world_size,
            is_sparge_backend=hb_backend in _HEAD_BALANCE_BACKENDS,
            joint_strategy=joint_strategy,
            attention_kwargs=attention_kwargs,
        )

    if fp8_comms is not None and joint_strategy:
        # query is fp8-quantized before the all-to-all but joint_key/value stay bf16.
        raise NotImplementedError("fp8 comms does not support joint attention.")

    joint_attn_kwargs = None
    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),
        }

    extra_inputs = _ulysses_extra_inputs(attention_kwargs, query)
    if use_attention_a2a:
        unsupported = []
        if extra_inputs:
            unsupported.append("extra Ulysses inputs")
        if _has_kv_cache(attn_layer):
            unsupported.append("KV caching")
        if dropout_p not in (None, 0.0):
            unsupported.append("dropout")
        if is_causal:
            unsupported.append("causal masking")
        if (attention_kwargs or {}).get("indices_k") is not None:
            unsupported.append("arbitrary varlen key packing")
        if unsupported:
            raise NotImplementedError("packed Attention A2A does not support " + ", ".join(unsupported))

    qkv_amaxes = None
    packed_scales = None
    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len") if use_attention_a2a else None
    if ulysses_world_size > 1:
        if use_attention_a2a:
            (query, key, value), packed_scales = fused_a2a_input(
                query,
                key,
                value,
                PROCESS_GROUP.ULYSSES_PG,
                get_ulysses_parallel_rank(),
                softmax_scale=query.shape[-1] ** -0.5,
                pending=attention_a2a_pending,
                valid_kv_len=valid_kv_len,
            )
        elif fp8_comms is not None:
            if extra_inputs:
                raise NotImplementedError(
                    f"fp8 comms does not support extra Ulysses inputs: {', '.join(name for name, _ in extra_inputs)}."
                )
            fp8_comms_backend = backend if backend is not None else get_runtime_state().attention_backend
            query, key, value, attn_kwargs_update, qkv_amaxes = fp8_comms_input_all_to_all(
                query,
                key,
                value,
                fp8_comms.q_scale,
                fp8_comms.k_scale,
                fp8_comms.v_scale,
                fp8_comms_backend,
            )
            attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif combine_qkv_a2a and kv_head_repeat > 1:
            exchanged = _combined_gqa_qkv_all_to_all(query, key, value, *(tensor for _, tensor in extra_inputs))
            query, key, value = exchanged[:3]
            for (name, _), tensor in zip(extra_inputs, exchanged[3:]):
                attention_kwargs[name] = tensor
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            exchanged = _combined_qkv_all_to_all(query, key, value, *(tensor for _, tensor in extra_inputs))
            query, key, value = exchanged[:3]
            for (name, _), tensor in zip(extra_inputs, exchanged[3:]):
                attention_kwargs[name] = tensor
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)
            for name, tensor in extra_inputs:
                attention_kwargs[name] = _ft_c_input_all_to_all(tensor)

    if use_attention_a2a:
        key, value, packed_scales = _trim_packed_attention_a2a_padding(
            key,
            value,
            packed_scales,
            valid_kv_len,
            get_fused_a2a_codecs(),
            attention_a2a_profile,
        )

    if _has_kv_cache(attn_layer):
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    # Uniform trailing padding needs no mask or varlen packing. Keeping all Q
    # rows but slicing K/V is equivalent to masking those keys and lets dense
    # backends retain their optimized cross-attention path.
    if not use_attention_a2a:
        key, value, attention_kwargs = _trim_trailing_kv_padding(key, value, attention_kwargs, hb_backend)

    if kv_head_repeat > 1:
        key, value = _repeat_kv_heads(key, value, kv_head_repeat)

    if get_sequence_parallel_world_size() == 1:  # No SP
        out, _ = attention_function(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            joint_attn_kwargs=joint_attn_kwargs,
            attention_kwargs=attention_kwargs,
        )

    elif ulysses_world_size == 1:  # Ring only
        out = ring_attn(
            attention_function,
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            joint_attn_kwargs=joint_attn_kwargs,
            attention_kwargs=attention_kwargs,
        )

    else:
        if use_attention_a2a:
            out = _attention_a2a_packed_attn_call(
                query,
                key,
                value,
                *packed_scales,
                attention_a2a_profile,
                query.shape[-1] ** -0.5,
            )
        elif ring_world_size == 1:  # Ulysses only
            out, _ = attention_function(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                joint_attn_kwargs=joint_attn_kwargs,
                attention_kwargs=attention_kwargs,
            )
        else:  # USP
            out = ring_attn(
                attention_function,
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                joint_attn_kwargs=joint_attn_kwargs,
                attention_kwargs=attention_kwargs,
            )
        if fp8_comms is not None:
            out = fp8_comms_output_all_to_all(out, fp8_comms.o_scale, qkv_amaxes)
        else:
            out = _ft_c_output_all_to_all(out)
        if hb_applied:
            # Restore global head order on the output, gather this step's per-head
            # costs across the Ulysses group, and plan next step's permutation.
            out = revert_head_balance(out, attention_kwargs, head_balance_layer, hb_uly)

    if fp8_module is not None and not use_attention_a2a and not joint_strategy:
        fp8_observe_output(get_runtime_state().fp8_comms, fp8_module, out, False)

    return out


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend=None,
    attention_kwargs=None,
    head_balance_layer=None,
    attn_layer=None,
):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.

    ``attn_layer`` and ``head_balance_layer`` are accepted for call-site signature
    parity with ``USP`` but ignored here: with no Ulysses parallelism there is
    no head sharding or FP8 all-to-all.
    """
    attention_function = _get_attention_function(backend=backend)
    # Same rule as USP: a backend that cannot serve packed keys gets the pad
    # sliced instead, which is the same computation.
    resolved = backend if backend is not None else get_runtime_state().attention_backend
    key, value, attention_kwargs = _trim_trailing_kv_padding(key, value, attention_kwargs, resolved)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out
