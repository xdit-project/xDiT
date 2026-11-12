"""SP pad rows across Z-Image's software-pipelined block stack.

``_run_modulated_stack`` rotates block ``i + 1``'s opening norm into block
``i``'s closing kernel.  That rotation spans a block boundary, which is where
``zero_sp_pad_rows_hook`` would otherwise clear the trailing SP pad rows -- and
the hook rides on ``layer.forward``, which the pipelined path never calls.  The
runner clears them itself instead, and these tests pin the two claims that makes
it correct:

* a pad row's reference value at such a boundary is exactly zero in *both*
  outputs, so clearing after the fused call lands on the same values as
  clearing between the two halves;
* every block in the stack actually gets cleared -- each block's attention sees
  zeroed pad rows, which is the quantity the hook exists to protect (per-tensor
  FP8 query scales are taken across it).

These need a GPU: the fused path is Triton + bf16, and on CPU the wrapper falls
through to the eager reference.
"""

import pytest
import torch

pytestmark = pytest.mark.accelerator

from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

from xfuser.model_executor.layers.fused_gated_residual_norm_zimage import (
    _reference,
    fused_gated_residual_norm,
)
from xfuser.model_executor.models.transformers.transformer_z_image import (
    _fused_block_forward,
    _run_modulated_stack,
)
from xfuser.model_executor.models.transformers.transformers_utils import (
    zero_sp_pad_rows,
    zero_sp_pad_rows_hook,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DIM = 256
HEADS = 4
BATCH = 2
SEQ = 64
PAD = 3


def _rms_norm(dim, device):
    return torch.nn.RMSNorm(dim, eps=1e-5, dtype=torch.bfloat16, device=device)


@requires_gpu
def test_a_pad_rows_reference_value_at_a_boundary_is_zero_in_both_outputs():
    """The claim the whole change rests on.

    At a block boundary the unfused pair is: close block i, let the hook zero
    its output, then open block i + 1 by norming what the hook left.  Since
    ``RMSNorm(0) * scale`` is zero, *both* tensors are zero on a pad row -- so
    computing them from stale data and overwriting afterwards is not an
    approximation of the hook, it is the same value.
    """
    device = "cuda"
    torch.manual_seed(0)
    shape = (BATCH, SEQ, DIM)
    res = torch.randn(shape, device=device, dtype=torch.bfloat16)
    y = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate = torch.randn(BATCH, 1, DIM, device=device, dtype=torch.bfloat16).tanh()
    scale = 1.0 + torch.randn(BATCH, 1, DIM, device=device, dtype=torch.bfloat16)
    norm_post, norm_pre = _rms_norm(DIM, device), _rms_norm(DIM, device)

    with torch.no_grad():
        # Reference: the hook fires *between* the two halves.
        x1_ref = res + gate * norm_post(y)
        zero_sp_pad_rows(x1_ref, PAD)
        h_ref = norm_pre(x1_ref) * scale

    zeros = torch.zeros(BATCH, PAD, DIM, device=device, dtype=torch.bfloat16)
    # Not assumed -- asserted: norming a zeroed row leaves it zero.
    assert torch.equal(h_ref[:, -PAD:, :], zeros)

    with torch.no_grad():
        x1, h = fused_gated_residual_norm(res, y, gate, norm_post, norm_pre, scale)
        # Kept to compare against after clearing. Both calls must be inside
        # no_grad: the wrapper is inference-only and a grad-enabled call takes
        # the eager fallback instead of the kernel.
        x1_uncleared, h_uncleared = x1.clone(), h.clone()
        zero_sp_pad_rows(x1, PAD)
        zero_sp_pad_rows(h, PAD)

    assert torch.equal(x1[:, -PAD:, :], zeros)
    assert torch.equal(h[:, -PAD:, :], zeros)
    # Clearing touches the pad rows and nothing else.
    assert torch.equal(x1[:, :-PAD, :], x1_uncleared[:, :-PAD, :])
    assert torch.equal(h[:, :-PAD, :], h_uncleared[:, :-PAD, :])


@requires_gpu
def test_b_clearing_after_the_fused_call_matches_clearing_between_the_halves():
    """Post-hoc clearing reproduces the hook's values on the real rows too.

    The fused pair and the eager pair differ by the fusion's known drift, so
    the real rows are compared loosely; the pad rows must agree exactly.
    """
    device = "cuda"
    torch.manual_seed(1)
    shape = (BATCH, SEQ, DIM)
    res = torch.randn(shape, device=device, dtype=torch.bfloat16)
    y = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate = torch.randn(BATCH, 1, DIM, device=device, dtype=torch.bfloat16).tanh()
    scale = 1.0 + torch.randn(BATCH, 1, DIM, device=device, dtype=torch.bfloat16)
    norm_post, norm_pre = _rms_norm(DIM, device), _rms_norm(DIM, device)

    with torch.no_grad():
        x1_ref, _ = _reference(res, y, gate, norm_post, norm_pre, scale)
        zero_sp_pad_rows(x1_ref, PAD)
        h_ref = norm_pre(x1_ref) * scale

        x1, h = fused_gated_residual_norm(res, y, gate, norm_post, norm_pre, scale)
        zero_sp_pad_rows(x1, PAD)
        zero_sp_pad_rows(h, PAD)

    torch.testing.assert_close(x1.float(), x1_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(h.float(), h_ref.float(), rtol=2e-2, atol=2e-2)
    assert torch.equal(x1[:, -PAD:, :], x1_ref[:, -PAD:, :])
    assert torch.equal(h[:, -PAD:, :], h_ref[:, -PAD:, :])


def _stack(n_layers, device):
    torch.manual_seed(2)
    layers = [
        ZImageTransformerBlock(i, DIM, HEADS, HEADS, 1e-5, True, modulation=True).to(
            device=device, dtype=torch.bfloat16
        )
        for i in range(n_layers)
    ]
    for layer in layers:
        layer.forward = _fused_block_forward.__get__(layer, type(layer))
    return layers


def _stack_inputs(device):
    torch.manual_seed(3)
    x = torch.randn(BATCH, SEQ, DIM, device=device, dtype=torch.bfloat16)
    # Realistic: the stack's input arrives with its pad rows already cleared,
    # by whatever produced it.
    zero_sp_pad_rows(x, PAD)
    attn_mask = torch.ones(BATCH, SEQ, dtype=torch.bool, device=device)
    attn_mask[:, -PAD:] = False
    freqs = torch.randn(BATCH, SEQ, DIM // HEADS // 2, device=device)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    adaln_input = torch.randn(BATCH, DIM, device=device, dtype=torch.bfloat16)
    return x, attn_mask, freqs_cis, adaln_input


@requires_gpu
def test_c_every_blocks_attention_sees_zeroed_pad_rows():
    """The plumbing test: no boundary may be left uncleared.

    A block's attention input is exactly what the hook protects -- Sage and
    AITER FP8 take one query scale across the whole tensor, so a stale pad row
    there coarsens every real token.  With the stack input already cleared,
    every block's attention input must be exactly zero on the pad rows.
    """
    device = "cuda"
    layers = _stack(4, device)
    x, attn_mask, freqs_cis, adaln_input = _stack_inputs(device)

    seen = []
    for layer in layers:
        layer.attention.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))

    with torch.no_grad():
        out = _run_modulated_stack(layers, x, attn_mask, freqs_cis, adaln_input, pad_rows=PAD)

    assert len(seen) == len(layers)
    zeros = torch.zeros(BATCH, PAD, DIM, device=device, dtype=torch.bfloat16)
    for i, q in enumerate(seen):
        assert torch.equal(q[:, -PAD:, :], zeros), f"block {i} attention saw non-zero pad rows"
    # The peeled epilogue is the last block's output and gets cleared too.
    assert torch.equal(out[:, -PAD:, :], zeros)


@requires_gpu
def test_d_pipelined_stack_tracks_the_hooked_per_layer_loop():
    """End to end against the path this replaces under padding.

    The per-layer loop goes through layer.forward, so the hook fires and owns
    the pad rows -- that is exactly what the runner used to fall back to.  Both
    share the within-block fusion, so the only difference is the rotation, and
    the real rows should track within the fusion's drift.
    """
    device = "cuda"
    x, attn_mask, freqs_cis, adaln_input = _stack_inputs(device)

    hooked = _stack(4, device)
    sp_pad = {"local_rows": PAD}
    for layer in hooked:
        layer.register_forward_hook(zero_sp_pad_rows_hook(sp_pad))
    with torch.no_grad():
        want = x
        for layer in hooked:
            want = layer(want, attn_mask, freqs_cis, adaln_input)

    pipelined = _stack(4, device)
    with torch.no_grad():
        got = _run_modulated_stack(pipelined, x, attn_mask, freqs_cis, adaln_input, pad_rows=PAD)

    zeros = torch.zeros(BATCH, PAD, DIM, device=device, dtype=torch.bfloat16)
    assert torch.equal(want[:, -PAD:, :], zeros)
    assert torch.equal(got[:, -PAD:, :], zeros)
    torch.testing.assert_close(got.float(), want.float(), rtol=5e-2, atol=5e-2)
