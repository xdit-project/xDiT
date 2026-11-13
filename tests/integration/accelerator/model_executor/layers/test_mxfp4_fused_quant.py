"""The fused MXFP4 producers must quantize exactly as the HIP quantization in ``xfuser::mxfp4_gemm``.

The attention-output op only reorders, so its bytes must match the HIP quantization of the
reordered rows. The SwiGLU and LayerNorm producers compute their bf16 result with FlyDSL
math, whose last fp32 bit can differ from torch's and so, rarely, its bf16 rounding; their
quantization of that result must still match the HIP one.
"""

import pytest
import torch

aiter = pytest.importorskip("aiter")
mxfp4_fused_quant = pytest.importorskip("xfuser.model_executor.layers.mxfp4_fused_quant")
from xfuser.model_executor.layers import mxfp4_linear  # noqa: E402,F401  registers the GEMM ops

if not mxfp4_fused_quant.MXFP4_FUSED_QUANT_AVAILABLE:
    pytest.skip("FlyDSL or AITER's FlyDSL MX helpers are unavailable.", allow_module_level=True)


def _hip_quant(rows):
    fp4, scale = aiter.get_hip_quant(aiter.QuantType.per_1x32)(rows.contiguous(), shuffle=True)
    return fp4.view(torch.uint8), scale.view(torch.uint8)


def _scale_positions(rows, cols, scale_cols, device):
    """The shuffled-buffer position of every (row, group) scale, as the HIP kernel writes it."""
    x = torch.arange(rows, device=device)[:, None]
    y = torch.arange(cols // 32, device=device)[None, :]
    return (
        (x // 32 * scale_cols) * 32 + (y // 8) * 256 + (y % 4) * 64 + (x % 16) * 4 + (y % 8) // 4 * 2 + (x % 32) // 16
    )


def _mismatches(fused, reference, rows, cols):
    """Fractions of differing fp4 bytes and of differing scales over the written entries."""
    (fp4, scale), (ref_fp4, ref_scale) = fused, reference
    assert fp4.shape == ref_fp4.shape and scale.shape == ref_scale.shape
    positions = _scale_positions(rows, cols, scale.shape[1], scale.device).flatten()
    scale_diff = scale.flatten()[positions] != ref_scale.flatten()[positions]
    return (fp4 != ref_fp4).float().mean().item(), scale_diff.float().mean().item()


def _activations(*shape, generator):
    x = torch.randn(*shape, generator=generator, device="cuda") * 3
    x[..., ::97] *= 40  # outlier channels, as in real activations
    x[..., :32] = 0  # an all-zero group in every row
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq_len", [65, 256])
def test_attention_output_quant_matches_hip_quant_bitwise(batch, seq_len):
    generator = torch.Generator(device="cuda").manual_seed(0)
    world_size, heads, head_dim = 8, 3, 128
    received = _activations(world_size, seq_len, batch, heads, head_dim, generator=generator)
    rows = received.permute(2, 1, 0, 3, 4).reshape(batch * seq_len, -1)

    fused = torch.ops.xfuser.seq_major_attention_output_mxfp4_quant(received)

    assert _mismatches(fused, _hip_quant(rows), *rows.shape) == (0.0, 0.0)


@pytest.mark.parametrize("world_size", [1, 8])
def test_fp8_attention_output_quant_dequantizes_as_fp8_comms(world_size):
    """An FP8 receive buffer (FP8 Ulysses comms) quantizes as its ``fp8_comms`` dequantization."""
    from xfuser.core.distributed.fp8_comms import _per_tensor_quant

    generator = torch.Generator(device="cuda").manual_seed(0)
    batch, seq_len, heads = 2, 261, 24 // world_size
    received = _activations(world_size, seq_len, batch, heads, 128, generator=generator)
    projected = _activations(batch, seq_len, 3 * 3072 + 2 * 9216, generator=generator)
    ff_hidden = projected[..., 3 * 3072 :]
    fp8, descale = _per_tensor_quant(received, torch.tensor([0.37], device="cuda"))
    dequantized = (fp8.float() * descale).to(torch.bfloat16)

    for op, args, cols in (
        (torch.ops.xfuser.seq_major_attention_output_mxfp4_quant, (), 3072),
        (torch.ops.xfuser.attention_output_swiglu_mxfp4_quant, (ff_hidden,), 3072 + 9216),
    ):
        fused = op(fp8, *args, descale)
        reference = op(dequantized, *args)
        assert _mismatches(fused, reference, batch * seq_len, cols) == (0.0, 0.0)
    with pytest.raises(ValueError, match="descale"):
        torch.ops.xfuser.seq_major_attention_output_mxfp4_quant(fp8)


def test_quantized_gemm_matches_the_quantizing_gemm():
    generator = torch.Generator(device="cuda").manual_seed(0)
    received = _activations(8, 66, 1, 3, 128, generator=generator)
    rows = received.permute(2, 1, 0, 3, 4).reshape(66, -1)
    layer = mxfp4_linear.xFuserMXFP4Linear(rows.shape[1], 512, bias=False, device="cuda", dtype=torch.bfloat16)
    layer._quantize_weights()

    fused = layer.forward_quantized(*torch.ops.xfuser.seq_major_attention_output_mxfp4_quant(received), (1, 66))

    torch.testing.assert_close(fused, layer(rows).view(1, 66, 512), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [80, 522, 2058])
def test_swiglu_quant_matches_hip_quant_of_swiglu(rows):
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = 9216
    # The feed-forward input is a column slice of the block's wider input GEMM output.
    projected = _activations(rows, 3 * 3072 + 2 * hidden, generator=generator)
    x = projected[:, 3 * 3072 :]
    gate, up = x.float().chunk(2, dim=-1)
    reference = _hip_quant((torch.nn.functional.silu(gate) * up).to(torch.bfloat16))

    fused = torch.ops.xfuser.swiglu_mxfp4_quant(x)

    fp4_diff, scale_diff = _mismatches(fused, reference, rows, hidden)
    assert fp4_diff < 1e-4 and scale_diff < 1e-3


@pytest.mark.parametrize("rows", [80, 522, 2058])
def test_flydsl_swiglu_quant_fills_columns_from_an_offset(rows):
    from xfuser.model_executor.layers import swiglu_mxfp4_flydsl

    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden, col_offset = 9216, 3072
    projected = _activations(rows, 3 * 3072 + 2 * hidden, generator=generator)
    x = projected[:, 3 * 3072 :]
    fp4, scale = mxfp4_fused_quant.mxfp4_buffers(rows, col_offset + hidden, x.device)
    if not swiglu_mxfp4_flydsl.supported(x, fp4, col_offset):
        pytest.skip("FlyDSL is unavailable.")
    alone_fp4, alone_scale = torch.ops.xfuser.swiglu_mxfp4_quant(x)

    swiglu_mxfp4_flydsl.launch_swiglu_mxfp4(x, fp4, scale, col_offset)

    assert torch.equal(fp4[:, col_offset // 2 :], alone_fp4)
    positions = _scale_positions(rows, col_offset + hidden, scale.shape[1], x.device)[:, col_offset // 32 :]
    alone_positions = _scale_positions(rows, hidden, alone_scale.shape[1], x.device)
    assert torch.equal(scale.flatten()[positions], alone_scale.flatten()[alone_positions])


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("rows", [80, 522])
def test_attention_output_swiglu_quant_matches_its_two_halves(world_size, rows):
    # The merged output GEMM's input: attention rows, then the feed-forward hidden state.
    generator = torch.Generator(device="cuda").manual_seed(0)
    batch, seq_len, heads, hidden = 2, rows // 2, 24 // world_size, 9216
    received = _activations(world_size, seq_len, batch, heads, 128, generator=generator)
    projected = _activations(batch, seq_len, 3 * 3072 + 2 * hidden, generator=generator)
    ff_hidden = projected[..., 3 * 3072 :]
    attention = torch.ops.xfuser.seq_major_attention_output_mxfp4_quant(received)
    swiglu = torch.ops.xfuser.swiglu_mxfp4_quant(ff_hidden)

    fp4, scale = torch.ops.xfuser.attention_output_swiglu_mxfp4_quant(received, ff_hidden)

    attention_cols = 3072
    assert fp4.shape == (rows, (attention_cols + hidden) // 2)
    assert torch.equal(fp4[:, : attention_cols // 2], attention[0])
    assert torch.equal(fp4[:, attention_cols // 2 :], swiglu[0])
    positions = _scale_positions(rows, attention_cols + hidden, scale.shape[1], scale.device)
    written = scale.flatten()[positions]
    for part, (_, part_scale), cols in (
        (written[:, : attention_cols // 32], attention, attention_cols),
        (written[:, attention_cols // 32 :], swiglu, hidden),
    ):
        assert torch.equal(part, part_scale.flatten()[_scale_positions(rows, cols, part_scale.shape[1], scale.device)])


@pytest.mark.parametrize("text_len", [0, 80])
def test_layernorm_modulate_quant_matches_hip_quant_of_the_modulated_norm(text_len):
    generator = torch.Generator(device="cuda").manual_seed(0)
    batch, content_len, dim, eps = 2, 522, 3072, 1e-6
    text, content = (_activations(batch, length, dim, generator=generator) for length in (text_len, content_len))
    # Text shares one modulation row per batch entry (stride 0); content rows each have their own.
    text_mod = _activations(batch, 1, 3 * dim, generator=generator).expand(-1, text_len, -1)
    content_mod = _activations(batch, content_len, 3 * dim, generator=generator)
    (text_shift, text_scale, _), (shift, scale, _) = text_mod.chunk(3, dim=-1), content_mod.chunk(3, dim=-1)

    def modulated(x, shift, scale):
        normed = torch.nn.functional.layer_norm(x.float(), (dim,), eps=eps)
        return (1 + scale.float()) * normed + shift.float()

    joint = torch.cat([modulated(text, text_shift, text_scale), modulated(content, shift, scale)], dim=1)
    reference = _hip_quant(joint.to(torch.bfloat16).flatten(0, 1))

    if text_len:
        fused = torch.ops.xfuser.layernorm_modulate_mxfp4_quant(
            text, text_shift, text_scale, eps, content, shift, scale
        )
    else:
        fused = torch.ops.xfuser.layernorm_modulate_mxfp4_quant(content, shift, scale, eps)

    fp4_diff, scale_diff = _mismatches(fused, reference, batch * (text_len + content_len), dim)
    assert fp4_diff < 1e-3 and scale_diff < 1e-3
