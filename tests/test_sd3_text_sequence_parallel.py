from types import SimpleNamespace
from unittest.mock import patch

import torch

from xfuser.model_executor.layers.attention_processor import (
    _joint_sp_padding_attention_kwargs,
)
from xfuser.model_executor.layers.usp import _trim_trailing_kv_padding
from xfuser.model_executor.pipelines.pipeline_stable_diffusion_3 import (
    xFuserStableDiffusion3Pipeline,
)


def _chunk_text(prompt_embeds, *, fp8_comms=None, rank=0):
    runtime_state = SimpleNamespace(
        fp8_comms=fp8_comms,
        split_text_embed_in_sp=True,
        text_embed_sp_pad=0,
    )
    module = "xfuser.model_executor.pipelines.pipeline_stable_diffusion_3"
    with (
        patch(f"{module}.get_runtime_state", return_value=runtime_state),
        patch(f"{module}.get_sequence_parallel_world_size", return_value=2),
        patch(f"{module}.get_sequence_parallel_rank", return_value=rank),
    ):
        result = xFuserStableDiffusion3Pipeline._chunk_text_for_sp(
            None, prompt_embeds
        )
    return result, runtime_state


def test_non_divisible_text_keeps_joint_attention_without_fp8_comms():
    prompt_embeds = torch.randn(1, 5, 4)

    result, runtime_state = _chunk_text(prompt_embeds)

    assert result is prompt_embeds
    assert runtime_state.text_embed_sp_pad == 0
    assert runtime_state.split_text_embed_in_sp is False


def test_non_divisible_text_is_padded_and_records_mask_length_for_fp8():
    prompt_embeds = torch.randn(1, 5, 4)

    result, runtime_state = _chunk_text(
        prompt_embeds, fp8_comms=object(), rank=1
    )

    assert result.shape == (1, 3, 4)
    torch.testing.assert_close(result[:, -1], torch.zeros(1, 4))
    assert runtime_state.text_embed_sp_pad == 1
    assert runtime_state.split_text_embed_in_sp is True


def test_divisible_text_is_chunked_without_padding():
    prompt_embeds = torch.randn(1, 6, 4)

    result, runtime_state = _chunk_text(prompt_embeds, rank=1)

    torch.testing.assert_close(result, prompt_embeds[:, 3:])
    assert runtime_state.text_embed_sp_pad == 0
    assert runtime_state.split_text_embed_in_sp is True


def test_joint_attention_describes_valid_kv_prefix():
    query = torch.randn(1, 4, 2, 8)
    encoder_query = torch.randn(1, 3, 2, 8)
    runtime_state = SimpleNamespace(text_embed_sp_pad=1)
    module = "xfuser.model_executor.layers.attention_processor"

    with (
        patch(f"{module}.get_runtime_state", return_value=runtime_state),
        patch(f"{module}.get_ulysses_parallel_world_size", return_value=2),
        patch(f"{module}.get_ring_parallel_world_size", return_value=1),
    ):
        kwargs = _joint_sp_padding_attention_kwargs(query, encoder_query)

    assert kwargs == {"valid_kv_len": 13}


def test_trailing_kv_padding_uses_views():
    key = torch.randn(1, 2, 14, 8)
    value = torch.randn_like(key)

    trimmed_key, trimmed_value = _trim_trailing_kv_padding(
        key, value, {"valid_kv_len": 13}
    )

    assert trimmed_key.shape == (1, 2, 13, 8)
    assert trimmed_value.shape == (1, 2, 13, 8)
    assert trimmed_key.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    assert trimmed_value.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
