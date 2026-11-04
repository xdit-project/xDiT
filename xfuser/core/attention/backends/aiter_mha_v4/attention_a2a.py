"""AITER MHA-v4 packed layouts used by quantized Attention A2A.

The distributed transport lives outside the attention backend, but its outputs
are AITER-specific packed buffers. Keep their views, padding compaction, and
fake-tensor metadata beside the MHA-v4 launcher rather than in generic USP.
"""

import inspect

import torch

from xfuser.config.attention_a2a import AttentionA2AConfig

_CUSTOM_OP_OPTIONS = (
    {"tags": (torch.Tag.cudagraph_unsafe,)}
    if hasattr(torch, "Tag") and "tags" in inspect.signature(torch.library.custom_op).parameters
    else {}
)


def _allocate_compact_aiter_mha_v4_kv(
    key,
    value,
    k_scales,
    v_scales,
    valid_kv_len,
    *,
    profile,
    copy_values,
):
    """Rebuild packed K/V when removing padding changes MHA-v4 tile strides."""
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
def _compact_aiter_mha_v4_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    valid_kv_len: int,
    profile: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _allocate_compact_aiter_mha_v4_kv(
        key,
        value,
        k_scales,
        v_scales,
        valid_kv_len,
        profile=profile,
        copy_values=True,
    )


@_compact_aiter_mha_v4_kv.register_fake
def _compact_aiter_mha_v4_kv_fake(
    key,
    value,
    k_scales,
    v_scales,
    valid_kv_len,
    profile,
):
    return _allocate_compact_aiter_mha_v4_kv(
        key,
        value,
        k_scales,
        v_scales,
        valid_kv_len,
        profile=profile,
        copy_values=False,
    )


def trim_aiter_mha_v4_a2a_padding(
    key,
    value,
    packed_scales,
    valid_kv_len,
    codecs,
    profile,
):
    """Consume Wan's uniform trailing padding from packed AITER operands."""
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
        compact_key, compact_value, k_scales, v_scales = _compact_aiter_mha_v4_kv(
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

    key = key[:, :valid_kv_len]
    value = value[:, :valid_kv_len]
    k_scales = packed_scales[1]
    if k_scales.ndim == 4 and k_scales.shape[1] == source_sequence:
        k_scales = k_scales[:, :valid_kv_len]
    return key, value, (packed_scales[0], k_scales, packed_scales[2])


def packed_output_views(
    outputs,
    scales,
    output_shape,
    codecs,
    valid_kv_len=None,
):
    """Rebuild role-specific logical MHA-v4 views over A2A raw buffers."""
    from aiter.ops.mha_v4 import (
        mxfp4_k_view,
        mxfp4_v_view,
        mxfp6_k_view,
    )

    b, sequence, heads, d = output_shape
    kv_sequence = sequence if valid_kv_len is None else valid_kv_len
    kv_shape = (b, kv_sequence, heads, d)
    q_raw, k_raw, v_raw = outputs
    q_scales, k_scales, v_scales = scales
    qk_codec, _, v_codec = codecs
    q_scale_shape = (*output_shape[:-1], d // 32)
    k_scale_shape = (*kv_shape[:-1], d // 32)
    tiles = (kv_sequence + 127) // 128

    if qk_codec == "bf16":
        q = q_raw.view(b, heads, sequence, d).transpose(1, 2)
        k = k_raw.view(b, heads, kv_sequence, d).transpose(1, 2)
    elif qk_codec in ("int8", "e4m3", "mxfp8"):
        q = q_raw.view(output_shape)
        k = k_raw.view(kv_shape)
        if qk_codec not in ("int8", "e4m3"):
            q_scales = q_scales.view(q_scale_shape)
            k_scales = k_scales.view(k_scale_shape)
    elif qk_codec == "mxfp4":
        q_scales = q_scales.view(q_scale_shape)
        k_scales = k_scales.view(k_scale_shape)
        q = q_raw.view(*output_shape[:-1], d // 2)
        k = mxfp4_k_view(k_raw, k_scales)
    elif qk_codec == "mxfp6":
        q_scales = q_scales.view(q_scale_shape)
        q = q_raw.view(*output_shape[:-1], d // 32 * 24)
        k, k_scales = mxfp6_k_view(
            k_raw,
            k_scales,
            b,
            kv_sequence,
            heads,
        )
    else:
        raise RuntimeError(f"unsupported packed Q/K codec {qk_codec!r}")

    if v_codec == "e4m3":
        v = v_raw.view(kv_shape)
    elif v_codec == "e4m3_pc":
        v = v_raw.view(kv_shape)
        v_scales = v_scales.view(b, heads, d)
    elif v_codec == "mxfp4":
        v_scales = v_scales.view(b, heads, tiles * 512)
        v = mxfp4_v_view(v_raw, v_scales, kv_sequence)
    elif v_codec == "mxfp6_p":
        v_scales = v_scales.view(b, heads, tiles * 512)
        v = torch.as_strided(
            v_raw,
            (b, kv_sequence, heads, d),
            (heads * tiles * 12288, 96, tiles * 12288, 1),
        )
    else:
        raise RuntimeError(f"unsupported packed V codec {v_codec!r}")

    return (
        (q, k, v),
        (q_scales, k_scales, v_scales),
    )


def fake_packed_raw_outputs(
    query,
    profile,
    world_size,
    valid_kv_len=None,
):
    """Describe raw transport outputs for fake-tensor and Inductor tracing."""
    config = AttentionA2AConfig(profile=profile)
    codecs = config.codecs
    b, h, s, d = query.shape
    shape = (b, s * world_size, h // world_size, d)
    from aiter.ops.mha_v4_quant import (
        mxfp4_k_raw_buffer_size,
        mxfp4_v_raw_buffer_size,
        mxfp6_v_raw_buffer_size,
    )
    from aiter.ops.triton.quant.mxfp6_fmha_pack import (
        fp6_k_raw_buffer_sizes,
    )

    sequence = shape[1]
    direct_valid_kv_len = None
    full_sequence = s * world_size
    if (
        valid_kv_len is not None
        and config.supports_direct_kv_output
        and full_sequence - s < valid_kv_len <= full_sequence
    ):
        direct_valid_kv_len = valid_kv_len
    kv_sequence = sequence if direct_valid_kv_len is None else direct_valid_kv_len
    heads = shape[2]
    q_numel = b * sequence * heads * d
    kv_numel = b * kv_sequence * heads * d
    qk_codec, _, v_codec = codecs

    if qk_codec in ("int8", "e4m3", "mxfp8"):
        q_raw = query.new_empty((q_numel,), dtype=torch.uint8)
        k_raw = query.new_empty((kv_numel,), dtype=torch.uint8)
        if qk_codec in ("int8", "e4m3"):
            q_scales = query.new_empty((1,), dtype=torch.float32)
            k_scales = query.new_empty((1,), dtype=torch.float32)
        else:
            q_scales = query.new_empty((q_numel // 32,), dtype=torch.uint8)
            k_scales = query.new_empty((kv_numel // 32,), dtype=torch.uint8)
    elif qk_codec in ("mxfp4", "mxfp6"):
        q_scales = query.new_empty((q_numel // 32,), dtype=torch.uint8)
        if qk_codec == "mxfp4":
            q_raw = query.new_empty((q_numel // 2,), dtype=torch.uint8)
            k_raw = query.new_empty(
                (mxfp4_k_raw_buffer_size(b, kv_sequence, heads),),
                dtype=torch.uint8,
            )
            k_scales = query.new_empty((kv_numel // 32,), dtype=torch.uint8)
        else:
            q_raw = query.new_empty((q_numel * 3 // 4,), dtype=torch.uint8)
            k_size, k_scale_size = fp6_k_raw_buffer_sizes(
                b,
                kv_sequence,
                heads,
            )
            k_raw = query.new_empty((k_size,), dtype=torch.uint8)
            k_scales = query.new_empty((k_scale_size,), dtype=torch.uint8)
    else:
        raise RuntimeError(f"unsupported packed Q/K codec {qk_codec!r}")

    if v_codec == "e4m3":
        v_raw = query.new_empty((kv_numel,), dtype=torch.uint8)
        v_scales = query.new_empty((1,), dtype=torch.float32)
    elif v_codec == "e4m3_pc":
        v_raw = query.new_empty((kv_numel,), dtype=torch.uint8)
        v_scales = query.new_empty((b, heads, d), dtype=torch.float32)
    elif v_codec in ("mxfp4", "mxfp6_p"):
        v_raw_size = (
            mxfp4_v_raw_buffer_size(b, kv_sequence, heads)
            if v_codec == "mxfp4"
            else mxfp6_v_raw_buffer_size(b, kv_sequence, heads)
        )
        v_raw = query.new_empty(
            (v_raw_size,),
            dtype=torch.uint8,
        )
        v_scales = query.new_empty(
            (b * heads * ((kv_sequence + 127) // 128) * 512,),
            dtype=torch.uint8,
        )
    else:
        raise RuntimeError(f"unsupported packed V codec {v_codec!r}")
    return (q_raw, k_raw, v_raw), (q_scales, k_scales, v_scales)
