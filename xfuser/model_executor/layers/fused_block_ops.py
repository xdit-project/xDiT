"""Fused FLUX.3 block kernels around the GEMMs: what Inductor cannot fuse across a custom
op, a collective or a separately compiled block.

- ``qk_norm_rope_seq_major_send``: the q/k RMSNorm and RoPE of a block's projected q, k, v,
  written straight into the Ulysses sequence-major send buffer ``[P, s, b, 3, h/P, d]``
  (see ``usp._seq_major_qkv_send``) with v copied alongside, instead of a norm kernel and a
  pack kernel.
- ``qk_norm_rope_seq_major_send_fp8``: the same send buffer in FP8 for FP8 Ulysses comms,
  with q and k Hadamard-rotated and q, k, v quantized by their calibrated per-tensor scales,
  instead of the rotation GEMMs, three quantization passes and three exchanges of USP's path.
- ``qk_norm_rope``: the same q and k as one ``[b, s, 2, h, d]`` buffer, for the paths that
  exchange q, k and v some other way, or not at all.
- ``gated_residual_layernorm_modulate`` and its ``_mxfp4_quant`` variant: the previous
  block's gated residual ``x + gate * (attn + ff)`` of one or two segments (or
  ``x + gate * out`` after a merged output GEMM), then
  ``(1 + scale) * layer_norm(.) + shift`` of the result, the next block's input, in bf16
  or quantized as ``mxfp4_fused_quant.layernorm_modulate_mxfp4_quant`` quantizes it.

The kernels are ``flux3_block_flydsl``'s; inputs they do not take (fp32 activations,
unaligned rows) run the same math in torch. The math follows upstream: q/k norm and RoPE in
fp32 with one rounding to the output dtype; the residual rounds once to the hidden-state
dtype, which the LayerNorm then reads, and the modulated norm rounds once more.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from xfuser.model_executor.layers import flux3_block_flydsl, mxfp4_fused_quant

FUSED_BLOCK_OPS_AVAILABLE = flux3_block_flydsl._HAS_FLYDSL


def _check_qk_inputs(query, key, value, q_weight, k_weight, cos, sin, heads):
    head_dim = query.shape[-1] // heads
    if head_dim * heads != query.shape[-1] or head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(f"{query.shape[-1]} columns are not {heads} heads of a power of 2 of at least 16.")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.shape != query.shape or tensor.stride() != query.stride():
            raise ValueError(f"{name} must match query's shape and strides, got {tensor.shape}, {tensor.stride()}.")
    if query.stride(-1) != 1:
        raise ValueError(f"q, k and v must be contiguous in their last dimension, got strides {query.stride()}.")
    for name, tensor in (("q_weight", q_weight), ("k_weight", k_weight)):
        if tensor.shape != (head_dim,) or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [{head_dim}] tensor, got {tuple(tensor.shape)}.")
    if cos.shape != (query.shape[1], head_dim) or cos.stride(-1) != 1 or sin.stride() != cos.stride():
        raise ValueError(f"cos and sin must be [{query.shape[1]}, {head_dim}] rows, got {tuple(cos.shape)}.")
    return head_dim


def _norm_rope(x, weight, eps, cos, sin, heads):
    """Upstream's fp32 RMSNorm and interleaved RoPE of ``[b, s, h * d]`` ``x``, ``[b, s, h, d]``."""
    x = x.float().unflatten(-1, (heads, -1))
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    rotated = torch.stack([-y[..., 1::2], y[..., 0::2]], dim=-1).flatten(-2)
    return y * cos.float()[:, None] + rotated * sin.float()[:, None]


def _qk_norm_rope(query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, seq_major, fp8=None):
    """The q/k kernel on ``[b, s, h * d]`` q, k, v into ``out``: the send buffer
    ``[P, s, b, 3, h/P, d]`` if ``seq_major`` (in FP8 with ``fp8``, the rotation and the q, k,
    v scales), else ``[b, s, 2, h, d]`` q and k."""
    head_dim = query.shape[-1] // heads
    fp8_dtype = None if fp8 is None else out.dtype
    if flux3_block_flydsl.qk_supported(query, cos, sin, q_weight, k_weight, heads, head_dim, fp8_dtype):
        return flux3_block_flydsl.launch_qk_norm_rope(
            query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, seq_major, fp8
        )
    parts = [_norm_rope(x, weight, eps, cos, sin, heads) for x, weight in ((query, q_weight), (key, k_weight))]
    if fp8 is not None:
        rotation, *scales = fp8
        parts = [(part.to(torch.bfloat16) @ rotation).float() for part in parts]
        parts.append(value.float().unflatten(-1, (heads, -1)))
        fp8_max = torch.finfo(out.dtype).max
        parts = [(part / scale).clamp(-fp8_max, fp8_max) for part, scale in zip(parts, scales)]
    elif seq_major:
        parts.append(value.unflatten(-1, (heads, -1)))
    for which, part in enumerate(parts):
        if seq_major:
            out[:, :, :, which] = part.unflatten(2, (out.shape[0], -1)).permute(2, 1, 0, 3, 4)
        else:
            out[:, :, which] = part
    return out


@torch.library.custom_op("xfuser::qk_norm_rope_seq_major_send", mutates_args=())
def qk_norm_rope_seq_major_send(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    world_size: int,
) -> torch.Tensor:
    """The ``[P, s, b, 3, h/P, d]`` Ulysses send buffer of ``[b, s, h * d]`` q, k, v (column
    slices of one GEMM output, so sharing strides): q and k RMSNormed by their weights and
    rotated by the ``[s, d]`` cos/sin rows, v as is."""
    head_dim = _check_qk_inputs(query, key, value, q_weight, k_weight, cos, sin, heads)
    if heads % world_size:
        raise ValueError(f"{heads} heads do not split over {world_size} ranks.")
    batch, seq_len, _ = query.shape
    out = query.new_empty(world_size, seq_len, batch, 3, heads // world_size, head_dim)
    return _qk_norm_rope(query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, seq_major=True)


@qk_norm_rope_seq_major_send.register_fake
def _(query, key, value, q_weight, k_weight, eps, cos, sin, heads, world_size):
    batch, seq_len, cols = query.shape
    return query.new_empty(world_size, seq_len, batch, 3, heads // world_size, cols // heads)


@torch.library.custom_op("xfuser::qk_norm_rope_seq_major_send_fp8", mutates_args=())
def qk_norm_rope_seq_major_send_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    world_size: int,
    rotation: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """``qk_norm_rope_seq_major_send`` in ``fp8_dtype`` for FP8 Ulysses comms: q and k also
    rotated by the bf16 ``[d, d]`` ``rotation`` (``hadamard.rotate_qk``), then q, k and v
    quantized with the one-element per-tensor scales, as ``fp8_comms`` quantizes them."""
    head_dim = _check_qk_inputs(query, key, value, q_weight, k_weight, cos, sin, heads)
    if heads % world_size:
        raise ValueError(f"{heads} heads do not split over {world_size} ranks.")
    if rotation.shape != (head_dim, head_dim) or rotation.dtype != torch.bfloat16 or not rotation.is_contiguous():
        raise ValueError(f"rotation must be a contiguous bf16 [{head_dim}, {head_dim}] matrix.")
    for name, scale in (("q_scale", q_scale), ("k_scale", k_scale), ("v_scale", v_scale)):
        if scale.numel() != 1 or scale.dtype != torch.float32:
            raise ValueError(f"{name} must be a one-element fp32 tensor, got {tuple(scale.shape)} {scale.dtype}.")
    batch, seq_len, _ = query.shape
    out = query.new_empty(world_size, seq_len, batch, 3, heads // world_size, head_dim, dtype=fp8_dtype)
    fp8 = (rotation, q_scale, k_scale, v_scale)
    return _qk_norm_rope(query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, True, fp8)


@qk_norm_rope_seq_major_send_fp8.register_fake
def _(
    query,
    key,
    value,
    q_weight,
    k_weight,
    eps,
    cos,
    sin,
    heads,
    world_size,
    rotation,
    q_scale,
    k_scale,
    v_scale,
    fp8_dtype,
):
    batch, seq_len, cols = query.shape
    return query.new_empty(world_size, seq_len, batch, 3, heads // world_size, cols // heads, dtype=fp8_dtype)


@torch.library.custom_op("xfuser::qk_norm_rope", mutates_args=())
def qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """The RMSNormed, rotated ``[b, s, h * d]`` q and k as one ``[b, s, 2, h, d]`` buffer."""
    head_dim = _check_qk_inputs(query, key, key, q_weight, k_weight, cos, sin, heads)
    batch, seq_len, _ = query.shape
    out = query.new_empty(batch, seq_len, 2, heads, head_dim)
    return _qk_norm_rope(query, key, key, q_weight, k_weight, eps, cos, sin, heads, out, seq_major=False)


@qk_norm_rope.register_fake
def _(query, key, q_weight, k_weight, eps, cos, sin, heads):
    batch, seq_len, cols = query.shape
    return query.new_empty(batch, seq_len, 2, heads, cols // heads)


def _gated_residual_reference(segments, branches, eps):
    """The joint hidden state and fp32 modulated norm of ``segments`` in torch."""
    hidden, normed, offset = [], [], 0
    for x, gate, shift, scale in segments:
        rows = slice(offset, offset + x.shape[1])
        branch = sum(output[:, rows].float() for output in branches)
        h = (x.float() + gate.float() * branch).to(x.dtype)
        hidden.append(h)
        normed.append((1 + scale.float()) * F.layer_norm(h.float(), (h.shape[-1],), eps=eps) + shift.float())
        offset += x.shape[1]
    return torch.cat(hidden, dim=1), torch.cat(normed, dim=1)


def _gated_residual_layernorm_modulate(xs, gates, shifts, scales, branches, eps, quantize):
    """The joint ``[b, sum s_i, k]`` hidden state, then the bf16 modulated norm or its MXFP4
    values and scales, of one or two segments ``xs[i]`` ``[b, s_i, k]`` laid end to end along
    the sequence, whose branch output is the sum of the matching rows of the one or two
    ``branches`` (a block's attention and feed-forward outputs, or their merged GEMM's)."""
    if not 1 <= len(xs) <= 2 or not len(xs) == len(gates) == len(shifts) == len(scales):
        raise ValueError(f"Expected one or two segments with a gate, shift and scale each, got {len(xs)} segments.")
    if not 1 <= len(branches) <= 2:
        raise ValueError(f"Expected one or two branch outputs, got {len(branches)}.")
    batch, cols = xs[0].shape[0], xs[0].shape[-1]
    rows_per_batch = sum(x.shape[1] for x in xs)
    if any(branch.shape != (batch, rows_per_batch, cols) for branch in branches):
        shapes = [tuple(branch.shape) for branch in branches]
        raise ValueError(f"The branch outputs must be [{batch}, {rows_per_batch}, {cols}], got {shapes}.")
    segments = []
    for x, gate, shift, scale in zip(xs, gates, shifts, scales):
        for name, tensor in (("x", x), ("gate", gate), ("shift", shift), ("scale", scale)):
            if tensor.stride(-1) != 1 or tensor.shape[-1] != cols:
                raise ValueError(
                    f"{name} must be rows of {cols} contiguous columns, got {tensor.shape}, {tensor.stride()}."
                )
        segments.append((x, *(tensor.expand_as(x) for tensor in (gate, shift, scale))))
    for branch in branches:
        if branch.stride(-1) != 1:
            raise ValueError(f"Branch outputs must be contiguous in their last dimension, got {branch.stride()}.")
    tensors = [tensor for segment in segments for tensor in segment] + list(branches)
    if not flux3_block_flydsl.rows_supported(cols, *tensors):
        hidden, normed = _gated_residual_reference(segments, branches, eps)
        if not quantize:
            return hidden, normed.to(hidden.dtype), None, None
        return (hidden, None, *mxfp4_fused_quant.hip_quant(normed.to(torch.bfloat16).flatten(0, 1)))
    hidden = xs[0].new_empty(batch, rows_per_batch, cols)
    if quantize:
        out = hidden.new_empty(0)
        fp4, quant_scale = mxfp4_fused_quant.mxfp4_buffers(batch * rows_per_batch, cols, hidden.device)
    else:
        out = torch.empty_like(hidden)
        fp4 = quant_scale = hidden.new_empty(0, dtype=torch.uint8)
    flux3_block_flydsl.launch_block_rows(segments, branches, eps, hidden, out, fp4, quant_scale, quantize)
    return hidden, out, fp4, quant_scale


@torch.library.custom_op("xfuser::gated_residual_layernorm_modulate", mutates_args=())
def gated_residual_layernorm_modulate(
    xs: List[torch.Tensor],
    gates: List[torch.Tensor],
    shifts: List[torch.Tensor],
    scales: List[torch.Tensor],
    branches: List[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The joint hidden state ``x + gate * sum(branches)`` of the segments ``xs`` and its
    modulated norm ``(1 + scale) * layer_norm(.) + shift``, both ``[b, sum s_i, k]``.
    ``gates``, ``shifts`` and ``scales`` are broadcast-shaped views per segment."""
    hidden, out, _, _ = _gated_residual_layernorm_modulate(xs, gates, shifts, scales, branches, eps, quantize=False)
    return hidden, out


@gated_residual_layernorm_modulate.register_fake
def _(xs, gates, shifts, scales, branches, eps):
    return xs[0].new_empty(branches[0].shape), xs[0].new_empty(branches[0].shape)


@torch.library.custom_op("xfuser::gated_residual_layernorm_modulate_mxfp4_quant", mutates_args=())
def gated_residual_layernorm_modulate_mxfp4_quant(
    xs: List[torch.Tensor],
    gates: List[torch.Tensor],
    shifts: List[torch.Tensor],
    scales: List[torch.Tensor],
    branches: List[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gated_residual_layernorm_modulate`` with the modulated norm returned as the MXFP4
    values and shuffled scales of its ``[b * sum s_i, k]`` rows (see ``mxfp4_fused_quant``)."""
    hidden, _, fp4, quant_scale = _gated_residual_layernorm_modulate(
        xs, gates, shifts, scales, branches, eps, quantize=True
    )
    return hidden, fp4, quant_scale


@gated_residual_layernorm_modulate_mxfp4_quant.register_fake
def _(xs, gates, shifts, scales, branches, eps):
    batch, rows, cols = branches[0].shape
    return (xs[0].new_empty(branches[0].shape), *mxfp4_fused_quant.mxfp4_buffers(batch * rows, cols, xs[0].device))
