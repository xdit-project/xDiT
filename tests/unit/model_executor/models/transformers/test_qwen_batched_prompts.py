"""Per-row key packing for a Qwen-Image batch whose prompts are padded to one length."""

import pytest
import torch

from xfuser.model_executor.models.transformers import transformer_qwen as qwen
from xfuser.model_executor.models.transformers.transformer_qwen import per_sample_key_packing
from xfuser.model_executor.models.transformers.transformers_utils import local_sp_pad_rows


def _gathered_keys(row, text_len, text_seq_len, image_seq_len, sp_size):
    """The labels of the keys USP gathers for one row, built the way the wrapper lays them out.

    Labels: ("text", i) and ("image", i) for real tokens, None for prompt padding and SP pad.
    """
    text = [("text", i) if i < text_len else None for i in range(text_seq_len)]
    image = [("image", i) for i in range(image_seq_len)]
    image_pad = (-image_seq_len) % sp_size
    text_pad = (-text_seq_len) % sp_size
    sharded_padding = bool(image_pad or text_pad)

    def chunks(labels, pad):
        # chunk_and_pad_sequence: append the SP pad, then split into equal shards.
        padded = labels + [None] * pad
        size = len(padded) // sp_size
        return [padded[rank * size : (rank + 1) * size] for rank in range(sp_size)]

    gathered = []
    for rank, (text_chunk, image_chunk) in enumerate(zip(chunks(text, text_pad), chunks(image, image_pad))):
        if sharded_padding:
            image_rows = local_sp_pad_rows(image_seq_len, len(image_chunk), rank)
            text_rows = local_sp_pad_rows(text_seq_len, len(text_chunk), rank)
            real_image, pad_image = (
                image_chunk[: len(image_chunk) - image_rows],
                image_chunk[len(image_chunk) - image_rows :],
            )
            real_text, pad_text = text_chunk[: len(text_chunk) - text_rows], text_chunk[len(text_chunk) - text_rows :]
            gathered += real_image + real_text + pad_image + pad_text
        else:
            gathered += text_chunk + image_chunk
    return gathered


@pytest.mark.parametrize(
    ("text_lens", "image_seq_len", "sp_size"),
    [
        ([12, 7], 64, 4),  # no SP pad: [text, image] per rank
        ([12, 7], 66, 4),  # image SP pad: reordered layout
        ([13, 7], 64, 4),  # prompt SP pad: reordered layout
        ([14, 5], 70, 8),
        ([40, 34], 4096, 8),
        ([9, 9], 30, 1),  # no sequence parallelism
    ],
)
def test_packing_selects_each_rows_real_tokens_in_gathered_order(text_lens, image_seq_len, sp_size):
    text_seq_len = max(text_lens)
    packing = per_sample_key_packing(text_lens, text_seq_len, image_seq_len, sp_size)
    rows = [_gathered_keys(r, n, text_seq_len, image_seq_len, sp_size) for r, n in enumerate(text_lens)]
    gathered = len(rows[0])
    flat = [label for row in rows for label in row]

    selected = [flat[i] for i in packing["indices_k"].tolist()]
    cu = packing["cu_seqlens_k"].tolist()
    for r, row in enumerate(rows):
        expected = [label for label in row if label is not None]
        assert selected[cu[r] : cu[r + 1]] == expected
        assert all(r * gathered <= i < (r + 1) * gathered for i in packing["indices_k"][cu[r] : cu[r + 1]].tolist())
    assert packing["max_seqlen_k"] == image_seq_len + text_seq_len
    assert packing["cu_seqlens_k"].dtype == torch.int32


def test_the_replicated_prompt_layout_gets_no_packing():
    # 5 image tokens at degree 8 pad over several shards, so the prompt is replicated.
    assert per_sample_key_packing([8, 3], 8, 5, 8) is None


def _model():
    return qwen.xFuserQwenImageTransformerWrapper(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )


def test_forward_hands_the_packing_to_attention_and_drops_the_uniform_pad(monkeypatch):
    from types import SimpleNamespace

    model = _model()
    seen = {}

    def block(*args, **kwargs):
        seen.update(model._self_attn_kwargs)
        return kwargs["encoder_hidden_states"], kwargs["hidden_states"]

    model.transformer_blocks[0].forward = block
    monkeypatch.setattr(qwen, "get_sequence_parallel_world_size", lambda: 4)
    monkeypatch.setattr(qwen, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(qwen, "get_runtime_state", lambda: SimpleNamespace(fp8_comms=None))
    monkeypatch.setattr(qwen, "gather_and_unpad", lambda x, pad_amount, dim: x)
    packing = per_sample_key_packing([7, 3], 7, 8, 4)
    model(
        torch.ones(2, 8, 16),
        encoder_hidden_states=torch.ones(2, 7, 16),
        timestep=torch.tensor([0.5, 0.5]),
        img_shapes=[[(1, 1, 8)]] * 2,
        attention_kwargs=dict(packing),
    )
    assert seen["indices_k"] is packing["indices_k"]
    assert seen["max_seqlen_k"] == 15
    assert seen["valid_kv_len"] is None
    assert seen["sharded_padding"] is True
