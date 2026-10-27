"""Ideogram 4 publishes its real sequence length to USP and keeps the sequence-parallel pad rows at zero.

When the text + image token count does not divide the sequence-parallel degree, the wrapper zero-pads
the sequence. Attention drops the pad keys, but every block still writes the pad rows. Quantizing
attention backends share a Q scale across rows, so those rows would set it, and the sibling padded
wrappers clear them too. A forward hook on every block zeroes them; the real rows must not change.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers.models.transformers.transformer_ideogram4")

from diffusers.models.transformers.transformer_ideogram4 import (  # noqa: E402
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Transformer2DModel,
)

from xfuser.model_executor.models.transformers import transformer_ideogram4  # noqa: E402
from xfuser.model_executor.models.transformers import transformers_utils  # noqa: E402

SP_DEGREE = 2
_CONFIG = dict(
    in_channels=8,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    intermediate_size=64,
    adaln_dim=16,
    llm_features_dim=24,
    rope_theta=10_000,
    mrope_section=(2, 1, 1),
)
_IMAGE_TOKENS = 9


def _inputs(text_tokens):
    length = text_tokens + _IMAGE_TOKENS
    indicator = torch.full((1, length), OUTPUT_IMAGE_INDICATOR)
    indicator[:, :text_tokens] = LLM_TOKEN_INDICATOR
    generator = torch.Generator().manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, length, _CONFIG["in_channels"], generator=generator),
        timestep=torch.tensor([0.3]),
        encoder_hidden_states=torch.randn(1, length, _CONFIG["llm_features_dim"], generator=generator),
        position_ids=torch.arange(length).view(1, length, 1).repeat(1, 1, 3),
        segment_ids=torch.ones(1, length, dtype=torch.long),
        indicator=indicator,
        return_dict=False,
    )


def _build(monkeypatch, usp, all_gather):
    """The wrapper as the last of SP_DEGREE ranks, with USP and the gather stubbed at the boundary."""
    monkeypatch.setattr(transformer_ideogram4, "USP", usp)
    monkeypatch.setattr(transformers_utils, "get_sp_group", lambda: SimpleNamespace(all_gather=all_gather))
    torch.manual_seed(0)
    wrapper_class = transformer_ideogram4.get_ideogram4_transformer_wrapper_class()
    model = wrapper_class.from_config(Ideogram4Transformer2DModel(**_CONFIG).config).eval()
    model._sp_rank, model._sp_size = SP_DEGREE - 1, SP_DEGREE
    return model


@pytest.mark.parametrize(("text_tokens", "valid_kv_len"), [(4, 13), (5, None)], ids=["padded", "divisible"])
def test_padding_is_published_to_the_usp_calls_through_the_shared_dict(monkeypatch, text_tokens, valid_kv_len):
    """The wrapper pads, publishes the real length, and every USP call receives the shared dict."""
    calls = []

    def usp_recorder(query, key, value, **kwargs):
        shared = kwargs.get("attention_kwargs")
        calls.append((shared, dict(shared or {}), key.shape[2]))
        return query

    wrapper = _build(monkeypatch, usp_recorder, lambda x, dim: torch.cat([x, x], dim=dim))
    with torch.no_grad():
        wrapper(**_inputs(text_tokens))

    length = text_tokens + _IMAGE_TOKENS
    assert len(calls) == _CONFIG["num_layers"]
    for shared, published, local_length in calls:
        assert shared is wrapper._self_attn_kwargs
        assert published == {"valid_kv_len": valid_kv_len}
        assert local_length == (length + 1) // 2


def _local_usp(query, key, value, *, attention_kwargs=None, **_):
    """This rank's share of USP: attention to its own keys minus the declared pad."""
    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
    keep = key.shape[2] - (0 if valid_kv_len is None else SP_DEGREE * key.shape[2] - valid_kv_len)
    return F.scaled_dot_product_attention(query, key[:, :, :keep], value[:, :, :keep])


@pytest.fixture
def last_rank(monkeypatch):
    return _build(monkeypatch, _local_usp, lambda x, dim: x)


def _forward(model, text_tokens):
    """Run a forward; return the sample and every block's output, cloned as the block left it."""
    outputs = []
    handles = [
        layer.register_forward_hook(lambda module, args, out: outputs.append(out.clone())) for layer in model.layers
    ]
    try:
        with torch.no_grad():
            (sample,) = model(**_inputs(text_tokens))
    finally:
        for handle in handles:
            handle.remove()
    return sample, outputs


def test_every_block_carries_one_pad_row_hook_however_the_model_is_loaded(tmp_path):
    wrapper_class = transformer_ideogram4.get_ideogram4_transformer_wrapper_class()
    reference = Ideogram4Transformer2DModel(**_CONFIG)
    reference.save_pretrained(tmp_path)
    # from_pretrained builds through from_config and installs again; so does the FP8 loader.
    from_pretrained = wrapper_class.from_pretrained(tmp_path)
    from_config = wrapper_class.from_config(reference.config)
    from_config._install_xfuser_processors()
    for model in (from_pretrained, from_config):
        assert len(model.layers) == _CONFIG["num_layers"]
        for layer in model.layers:
            assert len(layer._forward_hooks) == 1


def test_a_padded_forward_zeroes_only_the_pad_rows(last_rank):
    # 4 text + 9 image tokens pad to 14: the last rank holds 6 real rows and 1 pad row.
    sample, hooked = _forward(last_rank, text_tokens=4)
    assert last_rank._sp_pad["local_rows"] == 1

    for layer in last_rank.layers:
        layer._forward_hooks.clear()
    reference_sample, unhooked = _forward(last_rank, text_tokens=4)

    assert len(hooked) == len(unhooked) == _CONFIG["num_layers"]
    for index, (output, reference) in enumerate(zip(hooked, unhooked)):
        assert output.shape[1] == 7, f"block {index}: unexpected local length"
        assert not output[:, -1:].any(), f"block {index}: pad row kept"
        assert reference[:, -1:].any(), f"block {index}: the reference pad row is already zero"
        assert torch.equal(output[:, :-1], reference[:, :-1]), f"block {index}: real rows changed"
    assert torch.equal(sample, reference_sample)


def test_an_unpadded_forward_declares_no_pad_rows(last_rank):
    last_rank._sp_pad["local_rows"] = 3  # left over from an earlier padded request
    _, hooked = _forward(last_rank, text_tokens=5)
    assert last_rank._sp_pad["local_rows"] == 0
    assert all(output[:, -1:].any() for output in hooked)
