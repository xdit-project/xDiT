from types import SimpleNamespace

import pytest
import torch

qwenimage21 = pytest.importorskip(
    "diffusers.models.transformers.transformer_qwenimage21",
    reason="installed diffusers does not include Qwen-Image-2.1",
)

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import usp
from xfuser.model_executor.models.transformers import transformer_qwenimage21 as xfuser_qwenimage21


NUM_LAYERS = 2


def _tiny_model():
    torch.manual_seed(0)
    return qwenimage21.QwenImage21Transformer2DModel(
        num_layers=NUM_LAYERS,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        in_channels=4,
        out_channels=4,
        axes_dims_rope=(4, 6, 6),
    ).eval()


def _tiny_inputs():
    # Joint layout: 3 text, one 2x2 condition image (one VLM slot), 3 text, then the 4x4 target (four slots).
    generator = torch.Generator().manual_seed(1)
    img_mask = torch.tensor([[False] * 3 + [True] + [False] * 3 + [True] * 4])
    return {
        "hidden_states": torch.randn(1, 4 + 16, 4, generator=generator),
        "encoder_hidden_states": torch.randn(1, 7, 8, generator=generator),
        "img_shapes": [[(1, 2, 2), (1, 4, 4)]],
        "img_mask": img_mask,
    }


def _prefill_then_decode(model, encoder_hidden_states_mask=None):
    inputs = _tiny_inputs()
    kv_cache = qwenimage21.QwenImage21KVCache(NUM_LAYERS)
    outputs = []
    for timestep, mode in ((0.9, "extract"), (0.4, "cached")):
        with torch.no_grad():
            outputs.append(
                model(
                    **inputs,
                    timestep=torch.tensor([timestep]),
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    kv_cache=kv_cache,
                    kv_cache_mode=mode,
                ).sample
            )
    return outputs


@pytest.fixture
def xfuser_sdpa_calls(monkeypatch):
    monkeypatch.setattr(usp, "get_runtime_state", lambda: SimpleNamespace(attention_backend=AttentionBackendType.SDPA))
    monkeypatch.setattr(xfuser_qwenimage21, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(xfuser_qwenimage21, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(xfuser_qwenimage21, "get_sequence_parallel_rank", lambda: 0)
    calls = []
    real_attention = xfuser_qwenimage21.attention

    def counting_attention(*args, **kwargs):
        calls.append(args[0].shape)
        return real_attention(*args, **kwargs)

    monkeypatch.setattr(xfuser_qwenimage21, "attention", counting_attention)
    return calls


def _set_processor(model, processor_cls):
    for block in model.transformer_blocks:
        block.attn.set_processor(processor_cls())


@pytest.mark.parametrize(
    "processor_name",
    ["xFuserQwenImage21AttnProcessor", "xFuserQwenImage21FlexAttnProcessor"],
)
def test_decode_runs_on_xfuser_backend_and_matches_diffusers(xfuser_sdpa_calls, processor_name):
    processor_cls = getattr(xfuser_qwenimage21, processor_name)
    if processor_name == "xFuserQwenImage21FlexAttnProcessor" and not qwenimage21._FLEX_AVAILABLE:
        pytest.skip("flex_attention unavailable")

    model = _tiny_model()
    reference = _prefill_then_decode(model)

    _set_processor(model, processor_cls)
    actual = _prefill_then_decode(model)

    # Only the decode step reaches xDiT: one call per block, target-image queries only.
    assert len(xfuser_sdpa_calls) == NUM_LAYERS
    assert all(shape[2] == 16 for shape in xfuser_sdpa_calls)
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_padded_prompt_decode_falls_back_to_diffusers(xfuser_sdpa_calls):
    model = _tiny_model()
    mask = torch.ones(1, 7, dtype=torch.bool)
    mask[0, -1] = False
    reference = _prefill_then_decode(model, encoder_hidden_states_mask=mask)

    _set_processor(model, xfuser_qwenimage21.xFuserQwenImage21AttnProcessor)
    actual = _prefill_then_decode(model, encoder_hidden_states_mask=mask)

    assert xfuser_sdpa_calls == []
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "processor_name",
    ["xFuserQwenImage21AttnProcessor", "xFuserQwenImage21FlexAttnProcessor"],
)
def test_wrapper_forward_matches_diffusers(xfuser_sdpa_calls, processor_name):
    """The wrapper copies diffusers' forward; at Ulysses degree 1 it must match it exactly."""
    if processor_name == "xFuserQwenImage21FlexAttnProcessor" and not qwenimage21._FLEX_AVAILABLE:
        pytest.skip("flex_attention unavailable")
    stock = _tiny_model()
    reference = _prefill_then_decode(stock)

    wrapper = xfuser_qwenimage21.xFuserQwenImage21TransformerWrapper.from_config(stock.config).eval()
    wrapper.load_state_dict(stock.state_dict())
    _set_processor(wrapper, getattr(xfuser_qwenimage21, processor_name))
    actual = _prefill_then_decode(wrapper)

    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
