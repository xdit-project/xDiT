from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from xfuser.model_executor.models.transformers import transformer_wan
from xfuser.model_executor.models.transformers.transformer_wan import (
    xFuserWanTransformer3DWrapper,
)

_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=12,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=8,
    ffn_dim=32,
    num_layers=2,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=32,
)


def _dense_attention(query, key, value, **_):
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture
def single_rank():
    """One sequence-parallel rank, with attention run as plain SDPA on CPU."""
    module = transformer_wan.__name__
    runtime_state = SimpleNamespace(
        increment_step_counter=lambda: None,
        get_cross_attention_backend=lambda: None,
    )
    with (
        mock.patch(f"{module}.get_sequence_parallel_rank", return_value=0),
        mock.patch(f"{module}.get_sequence_parallel_world_size", return_value=1),
        mock.patch(f"{module}.get_runtime_state", return_value=runtime_state),
        mock.patch(f"{module}.get_sp_group", return_value=SimpleNamespace(all_gather=lambda x, dim: x)),
    ):
        yield


def _models():
    torch.manual_seed(0)
    stock = WanTransformer3DModel(**_CONFIG).eval()
    wrapper = xFuserWanTransformer3DWrapper(**_CONFIG).eval()
    wrapper.load_state_dict(stock.state_dict())
    for block in wrapper.blocks:
        block.attn1.processor.attention_function = _dense_attention
        block.attn2.processor.attention_function = _dense_attention
    return stock, wrapper


def _inputs():
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(1, 4, 3, 4, 6, generator=generator)
    encoder_hidden_states = torch.randn(1, 7, 16, generator=generator)
    return hidden_states, torch.tensor([500]), encoder_hidden_states


def test_blocks_receive_a_contiguous_residual_stream(single_rank):
    _, wrapper = _models()
    layouts = []
    wrapper.blocks[0].register_forward_pre_hook(lambda module, args: layouts.append(args[0].is_contiguous()))

    with torch.no_grad():
        wrapper(*_inputs())

    # A strided residual stream gives the same values but sends every residual add and
    # dtype cast in every block through slow uncoalesced elementwise kernels.
    assert layouts == [True]


def test_forward_matches_diffusers_on_one_rank(single_rank):
    stock, wrapper = _models()

    with torch.no_grad():
        expected = stock(*_inputs(), return_dict=False)[0]
        actual = wrapper(*_inputs(), return_dict=False)[0]

    torch.testing.assert_close(actual, expected)
