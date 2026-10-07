"""CPU cover for the neox (``rotate_half``) variant of the fused QK-norm+RoPE layer.

The FlyDSL kernel itself needs a GPU, so what is checkable here is everything
*around* it -- which is where this variant's correctness actually lives:

* the table fold in :func:`prepare_neox_rope_tables`, which is what lets the
  kernel body be branch-free, and is pure arithmetic;
* :func:`_pick_block`'s ``half`` divisibility rule, the invariant the kernel's
  partner-reload relies on to never straddle the rotary half boundary;
* :func:`_reference_neox`, the fallback taken whenever flydsl is importable but
  the shape is out of envelope, including its handling of the 4-tuple
  ``rotary_emb`` the MiniMax-H3 wrapper now sends down;
* the envelope guards, which must route to that reference rather than compute
  something subtly wrong.

These tests do not exercise the emitted kernel. ``_supported`` rejects CPU
tensors, so every call through the public entry point here takes the fallback.
MiniMax-H3's real geometry (``head_dim=128``, ``rotary_dim=96``) is used
throughout so the parameters under test are the ones the model actually hits.
"""

import pytest
import torch

from xfuser.model_executor.layers import fused_qk_rope_flydsl as _module
from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    _pick_block,
    _reference_neox,
    flydsl_fused_qk_norm_rope,
    prepare_neox_rope_tables,
)

# MiniMax-H3: attention_head_dim=128, rotary_dim = 2 * 3 * rope_freq_dim = 96.
HEAD_DIM = 128
ROTARY_DIM = 96


def _rope_tables(seq_len, rotary_dim, *, seed=0):
    """fp32 cos/sin shaped like ``MiniMaxH3RotaryPosEmbed.forward`` returns."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(seq_len, rotary_dim, generator=generator) * 6.0
    return angles.cos(), angles.sin()


def _qk(seq_len, heads, head_dim, *, seed=0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, seq_len, heads, head_dim)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype),
        torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype),
    )


def _rotate_half(hidden_states, cos, sin, rotary_dim):
    """The rotation the fold is supposed to reproduce, written out literally."""
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)


def _apply_folded(hidden_states, cos_pad, sin_fold, rotary_dim):
    """The branch-free expression the kernel evaluates, one channel at a time.

    ``out[c] = x[c] * cos_pad[c] + x[(c + half) % rot] * sin_fold[c]``. The
    kernel reaches the partner by re-loading a VEC block at ``p_choff``; here a
    gather over the same index expresses the same thing.
    """
    head_dim = hidden_states.shape[-1]
    half = rotary_dim // 2
    partner = (torch.arange(head_dim) + half) % rotary_dim
    cos_pad = cos_pad[None, :, None, :]
    sin_fold = sin_fold[None, :, None, :]
    return hidden_states * cos_pad + hidden_states[..., partner] * sin_fold


def test_reference_neox_is_norm_then_the_diffusers_rotation():
    """The fallback must be diffusers' rotate, composed in the model's order.

    It is also handed the 4-tuple the MiniMax-H3 wrapper builds -- the folded
    tables ride along behind the originals, and the reference has to keep
    reading the originals and ignore the rest.
    """
    diffusers_rotate = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")._apply_rotary_emb

    cos, sin = _rope_tables(64, ROTARY_DIM)
    query, key = _qk(64, 4, HEAD_DIM)
    norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    norm_k = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    four_tuple = (cos, sin) + prepare_neox_rope_tables(cos, sin, HEAD_DIM)

    with torch.no_grad():
        got_q, got_k = _reference_neox(query, key, norm_q, norm_k, four_tuple)
        want_q = diffusers_rotate(norm_q(query), cos, sin)
        want_k = diffusers_rotate(norm_k(key), cos, sin)

    assert torch.equal(got_q, want_q)
    assert torch.equal(got_k, want_k)


@pytest.mark.parametrize(
    ("head_dim", "rotary_dim"),
    [
        (HEAD_DIM, ROTARY_DIM),  # MiniMax-H3: a 32-channel passthrough tail
        (64, 64),  # no tail at all, so pad == 0
    ],
)
def test_folded_tables_reproduce_rotate_half_exactly(head_dim, rotary_dim):
    """The fold is a sign flip and a mask, so it must be bit-exact, not close.

    Both sides are evaluated in float64 off the same bf16-rounded angles, which
    takes arithmetic rounding out of the comparison entirely: what remains is
    only whether the fold reassociates the rotation correctly.
    """
    cos, sin = _rope_tables(32, rotary_dim)
    cos_pad, sin_fold = prepare_neox_rope_tables(cos, sin, head_dim)
    hidden_states, _ = _qk(32, 2, head_dim)

    hidden_states = hidden_states.to(torch.float64)
    # prepare_neox_rope_tables rounds to bf16 to match the reference's
    # ``cos.to(hidden_states.dtype)``, so the reference has to see the same
    # rounded angles for this to isolate the fold.
    cos_bf = cos.to(torch.bfloat16).to(torch.float64)
    sin_bf = sin.to(torch.bfloat16).to(torch.float64)

    expected = _rotate_half(hidden_states, cos_bf, sin_bf, rotary_dim)
    got = _apply_folded(
        hidden_states,
        cos_pad.to(torch.float64),
        sin_fold.to(torch.float64),
        rotary_dim,
    )

    assert torch.equal(got, expected)


def test_folded_tables_are_the_identity_over_the_passthrough_tail():
    cos, sin = _rope_tables(8, ROTARY_DIM)

    cos_pad, sin_fold = prepare_neox_rope_tables(cos, sin, HEAD_DIM)

    assert cos_pad.shape == sin_fold.shape == (8, HEAD_DIM)
    assert cos_pad.dtype == sin_fold.dtype == torch.bfloat16
    assert torch.equal(cos_pad[:, ROTARY_DIM:], torch.ones_like(cos_pad[:, ROTARY_DIM:]))
    assert torch.equal(sin_fold[:, ROTARY_DIM:], torch.zeros_like(sin_fold[:, ROTARY_DIM:]))
    # The sign fold is exact in both halves -- negation never rounds.
    half = ROTARY_DIM // 2
    assert torch.equal(sin_fold[:, :half], -sin.to(torch.bfloat16)[:, :half])
    assert torch.equal(sin_fold[:, half:ROTARY_DIM], sin.to(torch.bfloat16)[:, half:])


def test_passthrough_tail_normalizes_negative_zero_to_positive_zero():
    """The one documented inexactness of the fold, pinned so it stays documented.

    ``own * 1 + partner * 0`` turns ``-0.0`` into ``+0.0``. Harmless for an
    activation, but it is the only way the fold is not bit-exact, and a silent
    change here would mean the fold stopped being a pure mask.
    """
    cos, sin = _rope_tables(1, ROTARY_DIM)
    cos_pad, sin_fold = prepare_neox_rope_tables(cos, sin, HEAD_DIM)
    hidden_states = torch.zeros(1, 1, 1, HEAD_DIM, dtype=torch.float32)
    hidden_states[..., ROTARY_DIM] = -0.0

    folded = _apply_folded(hidden_states, cos_pad.to(torch.float32), sin_fold.to(torch.float32), ROTARY_DIM)

    assert torch.signbit(hidden_states[0, 0, 0, ROTARY_DIM])
    assert not torch.signbit(folded[0, 0, 0, ROTARY_DIM])


@pytest.mark.parametrize("wave_size", [64, 32])
def test_pick_block_keeps_a_lane_inside_one_rotary_half(wave_size):
    """The partner-reload is only correct while no VEC block straddles ``half``.

    ``vec | half`` is what guarantees that, and -- because ``vec | d`` too --
    that the partner block is itself VEC-aligned. wave32 is covered because the
    kernel caps BLOCK_THREADS at the device wave, and gfx12xx is wave32.
    """
    half = ROTARY_DIM // 2

    block_threads, vec = _pick_block(HEAD_DIM, wave_size, half)

    assert block_threads * vec == HEAD_DIM
    assert block_threads <= wave_size
    assert half % vec == 0
    assert vec % 2 == 0


def test_pick_block_rejects_a_rotary_half_no_lane_width_divides():
    # head_dim 128 with wave 64 forces vec=2, then 4, then 8 ...; an odd half
    # divides none of them, so there is no legal lane assignment.
    assert _pick_block(HEAD_DIM, 64, 7) is None


@pytest.fixture
def envelope_probe(monkeypatch):
    """Record whether a call got past the neox guards to the envelope check.

    Without this the fallback assertions below are vacuous: ``_supported``
    rejects CPU tensors on its own, so deleting every neox guard would leave
    the results unchanged and the tests still green. Stubbing it out makes the
    guard decision observable -- a rejected shape must never reach here.
    """
    calls = []

    def probe(*args):
        calls.append(args)
        return False

    monkeypatch.setattr(_module, "_supported", probe)
    return calls


def test_neox_entry_point_matches_the_unfused_reference(envelope_probe):
    """Happy path: every guard passes, and the result is still the reference."""
    cos, sin = _rope_tables(64, ROTARY_DIM)
    query, key = _qk(64, 4, HEAD_DIM)
    norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    norm_k = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)

    with torch.no_grad():
        got_q, got_k = flydsl_fused_qk_norm_rope(
            query,
            key,
            norm_q,
            norm_k,
            (cos, sin),
            rope_style="neox",
            rotary_dim=ROTARY_DIM,
            neox_tables=prepare_neox_rope_tables(cos, sin, HEAD_DIM),
        )
        want_q, want_k = _reference_neox(query, key, norm_q, norm_k, (cos, sin))

    # MiniMax-H3's own geometry has to survive the guards, or the fused path is
    # dead code in the only model that uses it.
    assert len(envelope_probe) == 1
    # The width-D folded table is what reaches the envelope check, not cos.
    assert envelope_probe[0][2].shape == (64, HEAD_DIM)
    assert envelope_probe[0][3] == ROTARY_DIM // 2
    assert torch.equal(got_q, want_q)
    assert torch.equal(got_k, want_k)


@pytest.mark.parametrize(
    ("rotary_dim", "with_tables", "reason"),
    [
        (ROTARY_DIM, False, "neox without the folded tables"),
        (95, True, "odd rotary_dim has no half"),
        (HEAD_DIM + 8, True, "rotary_dim wider than the head"),
        (32, True, "passthrough tail wider than half"),
    ],
)
def test_out_of_envelope_neox_requests_fall_back(rotary_dim, with_tables, reason, envelope_probe):
    """Every rejected shape bails early and still produces the reference result.

    A guard that bails is fine; a guard that is missing computes a wrong
    rotation with no error, which is the failure mode worth testing for.
    """
    cos, sin = _rope_tables(64, ROTARY_DIM)
    query, key = _qk(64, 4, HEAD_DIM)
    norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    norm_k = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    tables = prepare_neox_rope_tables(cos, sin, HEAD_DIM) if with_tables else None

    with torch.no_grad():
        got_q, got_k = flydsl_fused_qk_norm_rope(
            query,
            key,
            norm_q,
            norm_k,
            (cos, sin),
            rope_style="neox",
            rotary_dim=rotary_dim,
            neox_tables=tables,
        )
        want_q, want_k = _reference_neox(query, key, norm_q, norm_k, (cos, sin))

    assert envelope_probe == [], reason
    assert torch.equal(got_q, want_q), reason
    assert torch.equal(got_k, want_k), reason


def test_an_unknown_rope_style_falls_back_to_the_interleaved_reference():
    apply_rotary_emb = pytest.importorskip("diffusers.models.embeddings").apply_rotary_emb
    cos, sin = _rope_tables(64, HEAD_DIM)
    query, key = _qk(64, 4, HEAD_DIM)
    norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)

    with torch.no_grad():
        got_q, _ = flydsl_fused_qk_norm_rope(query, key, norm_q, norm_q, (cos, sin), rope_style="not-a-rotation")
        want_q = apply_rotary_emb(norm_q(query), (cos, sin), sequence_dim=1)

    assert torch.equal(got_q, want_q)


def test_default_rope_style_is_still_the_interleaved_rotation():
    """Guards FLUX/Qwen/Z-Image: adding neox must not move the default."""
    apply_rotary_emb = pytest.importorskip("diffusers.models.embeddings").apply_rotary_emb
    cos, sin = _rope_tables(64, HEAD_DIM)
    query, key = _qk(64, 4, HEAD_DIM)
    norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)
    norm_k = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, dtype=torch.bfloat16)

    with torch.no_grad():
        got_q, got_k = flydsl_fused_qk_norm_rope(query, key, norm_q, norm_k, (cos, sin))
        want_q = apply_rotary_emb(norm_q(query), (cos, sin), sequence_dim=1)
        want_k = apply_rotary_emb(norm_k(key), (cos, sin), sequence_dim=1)

    assert torch.equal(got_q, want_q)
    assert torch.equal(got_k, want_k)
    # ... and that is a different rotation from neox, so the switch is real.
    neox_q, _ = flydsl_fused_qk_norm_rope(
        query,
        key,
        norm_q,
        norm_k,
        (cos, sin),
        rope_style="neox",
        rotary_dim=HEAD_DIM,
        neox_tables=prepare_neox_rope_tables(cos, sin, HEAD_DIM),
    )
    assert not torch.equal(neox_q, got_q)
