"""MXFP4 activation quantization fused into the kernels that produce FP4 GEMM inputs.

Each op returns the packed E2M1 values and shuffled E8M0 scales that AITER's HIP per-1x32
quantization (``aiter.get_hip_quant(per_1x32)(y, shuffle=True)``, the one inside
``xfuser::mxfp4_gemm``) gives for the producer's bf16 output ``y``, so
``xfuser::mxfp4_gemm_quantized`` takes them as they are and ``y`` is never written:

- ``layernorm_modulate_mxfp4_quant``: ``(1 + scale) * layer_norm(x) + shift``, a FLUX.3 block input.
- ``swiglu_mxfp4_quant``: ``silu(x[..., :n]) * x[..., n:]``, the feed-forward hidden state.
- ``seq_major_attention_output_mxfp4_quant``: the ``[b * s, h * d]`` rows of a Ulysses output
  all-to-all receive buffer ``[P, s, b, h / P, d]``, without the reorder copy.
- ``attention_output_swiglu_mxfp4_quant``: those attention rows and the feed-forward hidden
  state side by side, the input of a block's merged output GEMM.

The kernels are FlyDSL's (``swiglu_mxfp4_flydsl``, ``flux3_block_flydsl``); rows they do not
take are computed in torch and quantized by the HIP kernel.

Both outputs are uint8: the values are ``[M, K / 2]`` and the scales are
``[ceil(M / 256) * 256, ceil(K / 256) * 8]`` in the HIP quantization's shuffled order. The
producers keep fp32 intermediates and round once to bf16 before quantizing, as Inductor's
fused kernels do.
"""

from typing import Optional, Tuple

import torch

from xfuser.model_executor.layers import flux3_block_flydsl, swiglu_mxfp4_flydsl

MXFP4_FUSED_QUANT_AVAILABLE = swiglu_mxfp4_flydsl._HAS_FLYDSL and flux3_block_flydsl._HAS_FLYDSL


def mxfp4_buffers(rows: int, cols: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uninitialized value and scale buffers for an ``[rows, cols]`` activation, shaped as
    the HIP quantization allocates them."""
    if cols % 32:
        raise ValueError(f"MXFP4 quantizes groups of 32 columns, got {cols} columns.")
    fp4 = torch.empty(rows, cols // 2, dtype=torch.uint8, device=device)
    scale = torch.empty(
        (rows + 255) // 256 * 256, flux3_block_flydsl.mxfp4_scale_cols(cols), dtype=torch.uint8, device=device
    )
    return fp4, scale


def hip_quant(rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The HIP per-1x32 quantization of bf16 ``[M, K]`` ``rows``, as uint8 buffers."""
    import aiter

    fp4, quant_scale = aiter.get_hip_quant(aiter.QuantType.per_1x32)(rows.contiguous(), shuffle=True)
    return fp4.view(torch.uint8), quant_scale.view(torch.uint8)


def _check_rows(name: str, tensor: torch.Tensor) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dimension, got strides {tensor.stride()}.")


@torch.library.custom_op("xfuser::layernorm_modulate_mxfp4_quant", mutates_args=())
def layernorm_modulate_mxfp4_quant(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    x_after: Optional[torch.Tensor] = None,
    shift_after: Optional[torch.Tensor] = None,
    scale_after: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``(1 + scale) * layer_norm(x) + shift`` of ``[b, s, k]`` ``x``, with
    ``shift`` and ``scale`` broadcast-shaped views (stride 0 where shared). With ``x_after``,
    each batch row's sequence continues with ``x_after``'s, modulated by its own pair, as if
    the two results were concatenated along the sequence."""
    segments = [(x, shift, scale)]
    if x_after is not None:
        segments.append((x_after, shift_after, scale_after))
    batch, cols = x.shape[0], x.shape[-1]
    rows_per_batch = sum(segment[0].shape[1] for segment in segments)
    for x_part, shift_part, scale_part in segments:
        for name, tensor in (("x", x_part), ("shift", shift_part), ("scale", scale_part)):
            _check_rows(name, tensor)
            if tensor.shape[-1] != cols:
                raise ValueError(f"{name} has {tensor.shape[-1]} columns, expected {cols}.")
    segments = [(x_part, None, *(t.expand_as(x_part) for t in rest)) for x_part, *rest in segments]
    if flux3_block_flydsl.rows_supported(cols, *(t for segment in segments for t in segment if t is not None)):
        fp4, quant_scale = mxfp4_buffers(batch * rows_per_batch, cols, x.device)
        empty = x.new_empty(0)
        flux3_block_flydsl.launch_block_rows(segments, [], eps, empty, empty, fp4, quant_scale, True)
        return fp4, quant_scale
    normed = [
        (1 + scale_part.float()) * torch.nn.functional.layer_norm(x_part.float(), (cols,), eps=eps) + shift_part.float()
        for x_part, _, shift_part, scale_part in segments
    ]
    return hip_quant(torch.cat(normed, dim=1).to(torch.bfloat16).flatten(0, 1))


@layernorm_modulate_mxfp4_quant.register_fake
def _(x, shift, scale, eps, x_after=None, shift_after=None, scale_after=None):
    rows = x.shape[1] + (0 if x_after is None else x_after.shape[1])
    return mxfp4_buffers(x.shape[0] * rows, x.shape[-1], x.device)


def _swiglu_rows(x: torch.Tensor) -> torch.Tensor:
    _check_rows("x", x)
    return x.flatten(0, -2) if x.ndim > 2 else x


def _swiglu(rows: torch.Tensor) -> torch.Tensor:
    gate, up = rows.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


@torch.library.custom_op("xfuser::swiglu_mxfp4_quant", mutates_args=())
def swiglu_mxfp4_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``silu(x[..., :n]) * x[..., n:]`` rows of ``x``, ``[..., 2n]``; the
    rows may be strided, e.g. a column slice of a wider GEMM output."""
    rows = _swiglu_rows(x)
    fp4, quant_scale = mxfp4_buffers(rows.shape[0], rows.shape[1] // 2, x.device)
    if swiglu_mxfp4_flydsl.supported(rows, fp4, 0):
        swiglu_mxfp4_flydsl.launch_swiglu_mxfp4(rows, fp4, quant_scale, 0)
        return fp4, quant_scale
    return hip_quant(_swiglu(rows))


@swiglu_mxfp4_quant.register_fake
def _(x):
    return mxfp4_buffers(x.numel() // x.shape[-1], x.shape[-1] // 2, x.device)


def _received_cols(received: torch.Tensor, descale: Optional[torch.Tensor]) -> int:
    world_size, _, _, heads, head_dim = received.shape
    if received.stride(4) != 1 or received.stride(3) != head_dim:
        raise ValueError(f"Each received chunk must hold contiguous heads, got strides {received.stride()}.")
    if (descale is None) != (received.element_size() != 1):
        raise ValueError(f"A descale goes with an FP8 receive buffer, got {received.dtype} and descale {descale}.")
    return world_size * heads * head_dim


def _attention_rows(received: torch.Tensor, descale: Optional[torch.Tensor]) -> torch.Tensor:
    """The bf16 ``[b * s, P * h * d]`` rows of a receive buffer, dequantized as ``fp8_comms`` does."""
    world_size, seq_len, batch, heads, head_dim = received.shape
    rows = received.permute(2, 1, 0, 3, 4).reshape(batch * seq_len, world_size * heads * head_dim)
    return rows if descale is None else (rows.float() * descale).to(torch.bfloat16)


@torch.library.custom_op("xfuser::seq_major_attention_output_mxfp4_quant", mutates_args=())
def seq_major_attention_output_mxfp4_quant(
    received: torch.Tensor, descale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``[b * s, P * h * d]`` rows of a ``[P, s, b, h, d]`` output receive
    buffer, chunk ``p`` holding heads ``p * h`` onwards for this rank's ``s`` tokens. An FP8
    buffer (FP8 Ulysses comms) comes with its one-element ``descale``."""
    _, seq_len, batch, _, _ = received.shape
    fp4, quant_scale = mxfp4_buffers(batch * seq_len, _received_cols(received, descale), received.device)
    if not flux3_block_flydsl.seq_major_output_supported(received, fp4):
        return hip_quant(_attention_rows(received, descale))
    flux3_block_flydsl.launch_seq_major_output_mxfp4(received, fp4, quant_scale, 0, descale)
    return fp4, quant_scale


@seq_major_attention_output_mxfp4_quant.register_fake
def _(received, descale=None):
    world_size, seq_len, batch, heads, head_dim = received.shape
    return mxfp4_buffers(batch * seq_len, world_size * heads * head_dim, received.device)


@torch.library.custom_op("xfuser::attention_output_swiglu_mxfp4_quant", mutates_args=())
def attention_output_swiglu_mxfp4_quant(
    received: torch.Tensor, ff_hidden: torch.Tensor, descale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``[b * s, P * h * d + n]`` rows of ``torch.cat([attention, swiglu],
    dim=-1)``: the ``received`` rows as ``seq_major_attention_output_mxfp4_quant`` reads
    them (with ``descale`` for an FP8 buffer), then ``swiglu_mxfp4_quant`` of the matching
    ``[b, s, 2n]`` ``ff_hidden`` rows. The attention width must be a multiple of the
    32-column group."""
    _, seq_len, batch, _, _ = received.shape
    attention_cols = _received_cols(received, descale)
    if attention_cols % 32:
        raise ValueError(f"The attention rows must fill whole 32-column groups, got {attention_cols} columns.")
    rows = _swiglu_rows(ff_hidden)
    if rows.shape[0] != batch * seq_len:
        raise ValueError(f"ff_hidden has {rows.shape[0]} rows, the attention output {batch * seq_len}.")
    fp4, quant_scale = mxfp4_buffers(batch * seq_len, attention_cols + rows.shape[1] // 2, received.device)
    if not (
        flux3_block_flydsl.seq_major_output_supported(received, fp4)
        and swiglu_mxfp4_flydsl.supported(rows, fp4, attention_cols)
    ):
        return hip_quant(torch.cat([_attention_rows(received, descale), _swiglu(rows)], dim=-1))
    flux3_block_flydsl.launch_seq_major_output_mxfp4(received, fp4, quant_scale, 0, descale)
    swiglu_mxfp4_flydsl.launch_swiglu_mxfp4(rows, fp4, quant_scale, attention_cols)
    return fp4, quant_scale


@attention_output_swiglu_mxfp4_quant.register_fake
def _(received, ff_hidden, descale=None):
    world_size, seq_len, batch, heads, head_dim = received.shape
    cols = world_size * heads * head_dim + ff_hidden.shape[-1] // 2
    return mxfp4_buffers(batch * seq_len, cols, received.device)
