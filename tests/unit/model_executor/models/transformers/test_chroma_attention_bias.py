"""Chroma's attention bias must reproduce what the diffusers blocks apply.

diffusers' ChromaPipeline hands the transformer a mask in the model dtype and each
block forms ``mask[:, None, None, :] * mask[:, None, :, None]``. SDPA treats a
floating point mask as an additive bias, so the reference adds +1 to the logits of
valid-valid pairs and still attends masked pad tokens. These tests pin that
behaviour, the Ulysses token order, and the exclusion of sequence-parallel padding.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers.models.transformers.transformer_chroma")

from xfuser.model_executor.models.transformers.transformer_chroma import (  # noqa: E402
    chroma_attention_bias,
)


def _pipeline_mask(valid_text, num_txt, num_img, dtype=torch.bfloat16):
    """The mask ChromaPipeline passes: prompt tokens plus one pad, then all image tokens."""
    text = (torch.arange(num_txt) <= valid_text).to(dtype)[None]
    return torch.cat([text, torch.ones(1, num_img, dtype=torch.bool)], dim=1)


def _diffusers_block_mask(mask):
    return mask[:, None, None, :] * mask[:, None, :, None]


def _ulysses_sequence(tokens, num_txt, sp_world_size):
    """What the all-to-all assembles: each rank's local [text shard, image shard]."""
    text, image = tokens[:, :num_txt], tokens[:, num_txt:]
    shards = [
        torch.cat([t, i], dim=1) for t, i in zip(text.chunk(sp_world_size, dim=1), image.chunk(sp_world_size, dim=1))
    ]
    return torch.cat(shards, dim=1)


def test_single_rank_bias_is_the_diffusers_block_mask():
    mask = _pipeline_mask(valid_text=3, num_txt=8, num_img=6)

    bias = chroma_attention_bias(mask, num_txt=8, num_img=6)

    expected = _diffusers_block_mask(mask)
    assert bias.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(bias, expected)
    # Masked pad tokens keep a zero bias rather than being excluded.
    assert torch.isfinite(bias).all()


def test_no_mask_and_no_padding_leaves_attention_unmasked():
    assert chroma_attention_bias(None, num_txt=8, num_img=8, sp_world_size=2) is None


@pytest.mark.parametrize("sp_world_size", [2, 4])
def test_bias_follows_the_ulysses_token_order(sp_world_size):
    num_txt, num_img = 8, 16
    mask = _pipeline_mask(valid_text=2, num_txt=num_txt, num_img=num_img, dtype=torch.float32)
    # Distinct per-token values so any misplaced row or column shows up.
    weights = mask * torch.linspace(1.0, 2.0, num_txt + num_img)

    bias = chroma_attention_bias(weights, num_txt, num_img, sp_world_size=sp_world_size)

    ordered = _ulysses_sequence(weights, num_txt, sp_world_size)
    assert torch.equal(bias, _diffusers_block_mask(ordered))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sequence_parallel_padding_does_not_change_real_tokens(dtype):
    """Pad tokens added for an uneven split must be invisible to the real ones."""
    torch.manual_seed(0)
    num_txt, num_img, txt_pad, img_pad, sp = 5, 7, 1, 1, 2
    heads, head_dim = 2, 8
    mask = _pipeline_mask(valid_text=1, num_txt=num_txt, num_img=num_img, dtype=dtype)
    q, k, v = (torch.randn(1, heads, num_txt + num_img, head_dim, dtype=dtype) for _ in range(3))
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=_diffusers_block_mask(mask))

    def pad(x, text_fill=7.0, image_fill=-3.0):
        # Garbage pad tokens: nothing may depend on their values.
        text, image = x[:, :, :num_txt], x[:, :, num_txt:]
        text_pad = torch.full_like(text[:, :, :txt_pad], text_fill)
        image_pad = torch.full_like(image[:, :, :img_pad], image_fill)
        padded = torch.cat([text, text_pad, image, image_pad], dim=2)
        return _ulysses_sequence(padded.transpose(1, 2), num_txt + txt_pad, sp).transpose(1, 2)

    bias = chroma_attention_bias(mask, num_txt, num_img, txt_pad, img_pad, sp)
    out = F.scaled_dot_product_attention(pad(q), pad(k), pad(v), attn_mask=bias)
    assert torch.isfinite(out).all()

    # Undo the layout: locate each real token in the padded, reordered sequence.
    positions = torch.arange(num_txt + num_img, dtype=torch.float32)[None, None, :, None]
    located = pad(positions, text_fill=-1.0, image_fill=-1.0)[0, 0, :, 0]
    order = [int((located == p).nonzero()) for p in range(num_txt + num_img)]
    torch.testing.assert_close(out[:, :, order], reference, rtol=0, atol=2e-2 if dtype == torch.bfloat16 else 1e-6)


def test_padding_without_a_mask_hides_only_the_pad_keys():
    bias = chroma_attention_bias(None, num_txt=3, num_img=5, txt_pad=1, img_pad=1, sp_world_size=2)

    # Ulysses order: [txt0 txt1 | img0 img1 img2] then [txt2 PAD | img3 img4 PAD].
    expected_keys = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1, 1, 0], dtype=torch.bool)
    assert bias.dtype == torch.bool
    assert torch.equal(bias.reshape(-1), expected_keys)


def test_mask_must_cover_the_joint_sequence():
    with pytest.raises(ValueError, match="must cover the text and image tokens"):
        chroma_attention_bias(torch.ones(1, 4), num_txt=4, num_img=4)
