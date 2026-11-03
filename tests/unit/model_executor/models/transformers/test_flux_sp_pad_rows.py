"""FLUX.1 keeps the sequence-parallel pad rows of every block's image output at zero.

Attention drops the pad keys, but the blocks still write the pad rows, and those
rows enter statistics taken over a whole local tensor (FP8-comms calibration, a
step cache's residual difference, quantization scales shared across query rows).
A forward hook on every double and single block zeroes them on the ranks that
hold any; the prompt rows and the real image rows must not change.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers.models.transformers.transformer_flux")

from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.models.transformers import transformer_flux as flux  # noqa: E402

SP_DEGREE = 2


def _local_usp(query, key, value, *, joint_query=None, joint_key=None, joint_value=None, attention_kwargs=None, **_):
    """This rank's share of USP: attention to its own keys minus the declared pad, behind any joint prompt."""
    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
    pad = 0 if valid_kv_len is None else SP_DEGREE * key.shape[2] - valid_kv_len
    key, value = key[:, :, : key.shape[2] - pad], value[:, :, : value.shape[2] - pad]
    if joint_query is not None:
        query = torch.cat([joint_query, query], dim=2)
        key = torch.cat([joint_key, key], dim=2)
        value = torch.cat([joint_value, value], dim=2)
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture
def last_rank(monkeypatch):
    """Run as the last of SP_DEGREE ranks, with the collectives and USP stubbed at the boundary."""
    runtime = SimpleNamespace(
        fp8_comms=None,
        split_text_embed_in_sp=True,
        attention_backend=AttentionBackendType.SDPA,
        num_pipeline_patch=1,
        max_condition_sequence_length=0,
    )
    monkeypatch.setattr(flux, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(flux, "get_sequence_parallel_world_size", lambda: SP_DEGREE)
    monkeypatch.setattr(flux, "get_sequence_parallel_rank", lambda: SP_DEGREE - 1)
    monkeypatch.setattr(flux, "get_ulysses_parallel_world_size", lambda: SP_DEGREE)
    monkeypatch.setattr(flux, "get_ring_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(flux, "get_sp_group", lambda: SimpleNamespace(all_gather=lambda sample, dim: sample))
    monkeypatch.setattr(flux, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda sample, dim: sample))
    monkeypatch.setattr(flux, "USP", _local_usp)
    monkeypatch.setattr(flux, "_HAS_FLYDSL", False)
    torch.manual_seed(0)
    model = flux.xFuserFlux1Transformer2DWrapper(
        in_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    return model


def _blocks(model):
    return [*model.transformer_blocks, *model.single_transformer_blocks]


def _forward(model, image_tokens, text_tokens):
    """Run a forward; return the sample and every block's (text, image) output, cloned as the block left it."""
    outputs = []
    handles = [
        block.register_forward_hook(lambda module, args, out: outputs.append(tuple(t.clone() for t in out)))
        for block in _blocks(model)
    ]
    generator = torch.Generator().manual_seed(1)
    try:
        with torch.no_grad():
            (sample,) = model(
                torch.randn(1, image_tokens, 8, generator=generator),
                encoder_hidden_states=torch.randn(1, text_tokens, 12, generator=generator),
                pooled_projections=torch.randn(1, 8, generator=generator),
                timestep=torch.tensor([0.5]),
                img_ids=torch.randn(image_tokens, 3, generator=generator),
                txt_ids=torch.randn(text_tokens, 3, generator=generator),
                return_dict=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    return sample, outputs


def test_every_double_and_single_block_carries_the_pad_row_hook(last_rank):
    blocks = _blocks(last_rank)
    assert len(blocks) == 4
    for block in blocks:
        assert len(block._forward_hooks) == 1


@pytest.mark.parametrize("text_tokens", [pytest.param(8, id="sharded-prompt"), pytest.param(7, id="replicated-prompt")])
def test_a_padded_forward_zeroes_only_the_image_pad_rows(last_rank, text_tokens):
    # 15 image tokens pad to 16: the last rank holds 7 real rows and 1 pad row.
    sample, hooked = _forward(last_rank, image_tokens=15, text_tokens=text_tokens)
    assert last_rank._sp_pad["local_rows"] == 1

    for block in _blocks(last_rank):
        block._forward_hooks.clear()
    reference_sample, unhooked = _forward(last_rank, image_tokens=15, text_tokens=text_tokens)

    assert len(hooked) == len(unhooked) == 4
    for index, ((text, image), (reference_text, reference_image)) in enumerate(zip(hooked, unhooked)):
        assert image.shape[1] == 8, f"block {index}: unexpected local image length"
        assert not image[:, -1:].any(), f"block {index}: pad row kept"
        assert reference_image[:, -1:].any(), f"block {index}: the reference pad row is already zero"
        assert torch.equal(image[:, :-1], reference_image[:, :-1]), f"block {index}: real image rows changed"
        assert torch.equal(text, reference_text), f"block {index}: prompt rows changed"
    assert torch.equal(sample, reference_sample)


def test_an_unpadded_forward_declares_no_pad_rows(last_rank):
    last_rank._sp_pad["local_rows"] = 3  # left over from an earlier padded request
    _, hooked = _forward(last_rank, image_tokens=16, text_tokens=8)
    assert last_rank._sp_pad["local_rows"] == 0
    assert all(image[:, -1:].any() for _, image in hooked)
