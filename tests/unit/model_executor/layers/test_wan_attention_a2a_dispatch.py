from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.model_executor.models.transformers import transformer_wan


def test_wan_self_attention_dispatches_to_dedicated_a2a_path():
    processor = transformer_wan.xFuserWanAttnProcessor()
    hidden_states = torch.empty(1)
    expected = object()

    with (
        mock.patch.object(transformer_wan, "get_fused_a2a_mode", return_value=1),
        mock.patch.object(processor, "_call_attention_a2a", return_value=expected) as a2a,
        mock.patch.object(processor, "_call_standard") as standard,
    ):
        result = processor(object(), hidden_states)

    assert result is expected
    a2a.assert_called_once()
    standard.assert_not_called()


def test_wan_cross_attention_stays_on_standard_path():
    processor = transformer_wan.xFuserWanAttnProcessor(is_cross_attention=True)
    hidden_states = torch.empty(1)
    expected = object()

    with (
        mock.patch.object(transformer_wan, "get_fused_a2a_mode", return_value=1),
        mock.patch.object(processor, "_call_attention_a2a") as a2a,
        mock.patch.object(processor, "_call_standard", return_value=expected) as standard,
    ):
        result = processor(object(), hidden_states)

    assert result is expected
    standard.assert_called_once()
    a2a.assert_not_called()


def test_wan_a2a_does_not_fall_back_without_flydsl_preprocessing():
    processor = transformer_wan.xFuserWanAttnProcessor()
    attn = SimpleNamespace(add_k_proj=None)

    with (
        mock.patch.object(transformer_wan, "_HAS_FLYDSL", False),
        pytest.raises(RuntimeError, match="requires the Wan FlyDSL preprocessing path"),
    ):
        processor._call_attention_a2a(
            attn,
            torch.empty(1),
            None,
            (torch.empty(1), torch.empty(1)),
        )
