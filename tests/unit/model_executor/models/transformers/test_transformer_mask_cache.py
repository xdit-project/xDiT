"""Krea-2 and LTX-2 build key-padding metadata once per mask and reuse it
across denoising steps. A later request must never be served the metadata of
an earlier one, even when its mask lands at the address the earlier mask was
freed from -- which the caching allocator makes the common case."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.models.transformers import transformer_krea2, transformer_ltx2


def _at_same_address(mask, values):
    """A new tensor over ``mask``'s storage holding ``values``: what the caching
    allocator commonly hands the next request once the previous mask is freed."""
    reused = torch.empty(0, dtype=mask.dtype).set_(mask.untyped_storage(), 0, mask.shape)
    return reused.copy_(values)


def _masked_attention(query, key, value, attention_kwargs=None, **_):
    """Stand-in for the sequence-parallel attention dispatch on one rank."""
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_kwargs["attn_mask"])


@pytest.fixture
def krea2():
    torch.manual_seed(0)
    model = transformer_krea2.xFuserKrea2Transformer2DWrapper(
        in_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=2,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    single_rank = SimpleNamespace(all_gather=lambda x, dim: x)
    with (
        mock.patch.object(transformer_krea2, "get_sequence_parallel_world_size", return_value=1),
        mock.patch.object(transformer_krea2, "get_sequence_parallel_rank", return_value=0),
        mock.patch(
            "xfuser.model_executor.models.transformers.transformers_utils.get_sp_group",
            return_value=single_rank,
        ),
        mock.patch.object(transformer_krea2, "USP", _masked_attention),
    ):
        yield model


def _krea2_inputs(text_len=6, image_len=4):
    torch.manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, image_len, 4),
        encoder_hidden_states=torch.randn(1, text_len, 2, 8),
        timestep=torch.tensor([0.5]),
        position_ids=torch.randint(0, 4, (text_len + image_len, 3)),
    )


@torch.no_grad()
def test_krea2_second_request_is_not_served_the_first_requests_mask(krea2):
    inputs = _krea2_inputs()
    request_b = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)

    mask_a = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    krea2(**inputs, encoder_attention_mask=mask_a)
    mask_b = _at_same_address(mask_a, request_b)
    assert mask_b.data_ptr() == mask_a.data_ptr()

    out = krea2(**inputs, encoder_attention_mask=mask_b).sample

    # A distinct live tensor with request B's content, so nothing cached matches it.
    expected = krea2(**inputs, encoder_attention_mask=request_b).sample
    torch.testing.assert_close(out, expected)


@pytest.fixture
def ltx2():
    return transformer_ltx2.xFuserLTX2VideoTransformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        num_layers=1,
        caption_channels=16,
    )


def test_ltx2_second_request_is_not_served_the_first_requests_mask(ltx2):
    mask_a = torch.tensor([[1, 1, 0, 0, 0, 0]])
    transformer_ltx2._get_mask_meta(ltx2._enc_mask_cache, mask_a)
    mask_b = _at_same_address(mask_a, torch.tensor([[1, 1, 1, 1, 1, 0]]))
    assert mask_b.data_ptr() == mask_a.data_ptr()

    meta = transformer_ltx2._get_mask_meta(ltx2._enc_mask_cache, mask_b)

    assert meta.indices_k.tolist() == [0, 1, 2, 3, 4]
    assert meta.max_seqlen_k == 5
