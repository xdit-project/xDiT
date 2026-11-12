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


class _QuantizedLinear(torch.nn.Linear):
    """Stands in for a quantized GEMM module that replaces nn.Linear."""


@pytest.mark.parametrize(
    ("change", "counted"),
    [(None, True), ("cross-attention", False), ("quantized", False), ("fp32", False)],
)
def test_sdma_q_k_first_is_counted_only_where_it_can_run(change, counted):
    """The runner's count of Q/K-first modules matches the projections the path accepts."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    from xfuser.model_executor.layers import usp_sdma

    processor = transformer_wan.xFuserWanAttnProcessor(is_cross_attention=change == "cross-attention")
    attn = WanAttention(64, heads=4, dim_head=16, processor=processor).to(torch.bfloat16)
    if change == "quantized":
        attn.to_q = _QuantizedLinear(64, 64, dtype=torch.bfloat16)
    if change == "fp32":
        attn.float()
    # The runner's preparation turns the processor into its registered SDMA subclass.
    sdma = usp_sdma._PROCESSORS[transformer_wan.xFuserWanAttnProcessor]
    processor.__class__ = sdma
    assert processor.sdma_qk_first(attn) == counted
