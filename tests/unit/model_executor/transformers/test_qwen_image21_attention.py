from types import SimpleNamespace

import pytest
import torch

qwenimage21 = pytest.importorskip(
    "diffusers.models.transformers.transformer_qwenimage21",
    reason="installed diffusers does not include Qwen-Image-2.1",
)

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import fused_qk_rope_flydsl, usp
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


DECODE_STEPS = 2


def _prefill_then_decode(model, encoder_hidden_states_mask=None):
    inputs = _tiny_inputs()
    kv_cache = qwenimage21.QwenImage21KVCache(NUM_LAYERS)
    generator = torch.Generator().manual_seed(2)
    outputs = []
    for timestep, mode in ((0.9, "extract"), (0.6, "cached"), (0.3, "cached")):
        with torch.no_grad():
            outputs.append(
                model(
                    **(inputs | {"hidden_states": torch.randn(inputs["hidden_states"].shape, generator=generator)}),
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


def _xfuser_model(stock, processor_cls):
    wrapper = xfuser_qwenimage21.xFuserQwenImage21TransformerWrapper.from_config(stock.config).eval()
    wrapper.load_state_dict(stock.state_dict())
    for block in wrapper.transformer_blocks:
        block.attn.set_processor(processor_cls())
    return wrapper


@pytest.mark.parametrize("key_in_cache", [False, True])
def test_cached_step_writes_kv_into_the_cache(key_in_cache):
    """A cached step's K/V equal prefix + new tokens, held in the cache filled by the extract step."""
    prefix_k, prefix_v, key, value = (torch.randn(1, n, 2, 4) for n in (3, 3, 5, 5))
    layer_cache = qwenimage21.QwenImage21KVLayerCache()
    xfuser_qwenimage21._update_kv_cache(
        torch.cat([prefix_k, torch.zeros_like(key)], dim=1),
        torch.cat([prefix_v, torch.zeros_like(value)], dim=1),
        layer_cache,
        "extract",
        slice(0, 3),
    )
    if key_in_cache:
        key = layer_cache.k[:, 3:].copy_(key)

    out_k, out_v = xfuser_qwenimage21._update_kv_cache(
        key, value, layer_cache, "cached", None, key_in_cache=key_in_cache
    )

    assert out_k is layer_cache.k and out_v is layer_cache.v
    torch.testing.assert_close(out_k, torch.cat([prefix_k, key], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(out_v, torch.cat([prefix_v, value], dim=1), rtol=0, atol=0)


def test_flydsl_qk_norm_rope_gets_tables_matching_diffusers(xfuser_sdpa_calls, monkeypatch):
    """With FlyDSL available, QK-norm + RoPE goes to its fused kernel, given (cos, sin) tables that reproduce diffusers.

    The kernel needs ROCm; on CPU the wrapper runs its diffusers reference, which takes the same tables.
    """
    fused_calls = []

    def fused_reference(query, key, norm_q, norm_k, rotary_emb, key_out=None):
        fused_calls.append(key_out is not None)
        return fused_qk_rope_flydsl.flydsl_fused_qk_norm_rope(query, key, norm_q, norm_k, rotary_emb, key_out=key_out)

    monkeypatch.setattr(xfuser_qwenimage21, "_HAS_FLYDSL", True)
    monkeypatch.setattr(xfuser_qwenimage21, "flydsl_fused_qk_norm_rope", fused_reference)
    stock = _tiny_model()
    reference = _prefill_then_decode(stock)
    actual = _prefill_then_decode(_xfuser_model(stock, xfuser_qwenimage21.xFuserQwenImage21AttnProcessor))

    # Cached steps write their keys straight into the KV cache.
    assert fused_calls == [False] * NUM_LAYERS + [True] * (DECODE_STEPS * NUM_LAYERS)
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "processor_name",
    ["xFuserQwenImage21AttnProcessor", "xFuserQwenImage21FlexAttnProcessor"],
)
def test_decode_runs_on_xfuser_backend_and_matches_diffusers(xfuser_sdpa_calls, processor_name):
    """At Ulysses degree 1 the wrapper and its processors must match stock diffusers."""
    if processor_name == "xFuserQwenImage21FlexAttnProcessor" and not qwenimage21._FLEX_AVAILABLE:
        pytest.skip("flex_attention unavailable")

    stock = _tiny_model()
    reference = _prefill_then_decode(stock)
    actual = _prefill_then_decode(_xfuser_model(stock, getattr(xfuser_qwenimage21, processor_name)))

    # The prefill's target rows and every decode step reach xDiT: one call per block, target-image queries only.
    assert len(xfuser_sdpa_calls) == (1 + DECODE_STEPS) * NUM_LAYERS
    assert all(shape[2] == 16 for shape in xfuser_sdpa_calls)
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "processor_name",
    ["xFuserQwenImage21AttnProcessor", "xFuserQwenImage21FlexAttnProcessor"],
)
def test_padded_prompt_decode_falls_back_to_diffusers(xfuser_sdpa_calls, processor_name):
    if processor_name == "xFuserQwenImage21FlexAttnProcessor" and not qwenimage21._FLEX_AVAILABLE:
        pytest.skip("flex_attention unavailable")

    stock = _tiny_model()
    mask = torch.ones(1, 7, dtype=torch.bool)
    mask[0, -1] = False
    reference = _prefill_then_decode(stock, encoder_hidden_states_mask=mask)
    actual = _prefill_then_decode(
        _xfuser_model(stock, getattr(xfuser_qwenimage21, processor_name)), encoder_hidden_states_mask=mask
    )

    assert xfuser_sdpa_calls == []
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def _count_calls(modules):
    """Forward calls of each of `modules`, counted from now on."""
    counts = [0] * len(modules)

    def counter(index):
        def count(module, args):
            counts[index] += 1

        return count

    for index, module in enumerate(modules):
        module.register_forward_pre_hook(counter(index))
    return counts


def _weight_bytes(model):
    """Bytes of the distinct storages behind a model's parameters."""
    storages = {
        parameter.untyped_storage().data_ptr(): parameter.untyped_storage().nbytes() for parameter in model.parameters()
    }
    return sum(storages.values())


def test_fused_projections_replace_the_separate_gemms_and_keep_the_output(xfuser_sdpa_calls):
    stock = _tiny_model()
    processor_cls = xfuser_qwenimage21.xFuserQwenImage21AttnProcessor
    reference = _prefill_then_decode(_xfuser_model(stock, processor_cls))
    model = _xfuser_model(stock, processor_cls)

    assert model.fuse_projections_for_inference() == {"qkv": NUM_LAYERS, "swiglu": NUM_LAYERS}

    blocks = model.transformer_blocks
    separate = _count_calls(
        [
            module
            for block in blocks
            for module in (
                block.attn.to_q,
                block.attn.to_k,
                block.attn.to_v,
                block.img_mlp.gate_layer,
                block.img_mlp.proj,
            )
        ]
    )
    fused = _count_calls([module for block in blocks for module in (block.attn.to_qkv, block.img_mlp.gate_proj)])
    actual = _prefill_then_decode(model)

    assert separate == [0] * (5 * NUM_LAYERS)
    assert fused == [1 + DECODE_STEPS] * (2 * NUM_LAYERS)
    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_fusing_projections_costs_no_memory():
    model = _xfuser_model(_tiny_model(), xfuser_qwenimage21.xFuserQwenImage21AttnProcessor)
    before = _weight_bytes(model)

    model.fuse_projections_for_inference()

    assert _weight_bytes(model) == before


def test_the_separate_swiglu_projections_compute_the_same_from_the_fused_weights():
    model = _xfuser_model(_tiny_model(), xfuser_qwenimage21.xFuserQwenImage21AttnProcessor)
    mlp = model.transformer_blocks[0].img_mlp
    sample = torch.randn(3, mlp.proj.in_features, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = mlp.gate_layer(sample), mlp.proj(sample)

    model.fuse_projections_for_inference()

    with torch.no_grad():
        assert torch.equal(mlp.gate_layer(sample), expected[0])
        assert torch.equal(mlp.proj(sample), expected[1])


def test_a_feed_forward_whose_projections_are_not_plain_linears_keeps_its_forward():
    class WrappedLinear(torch.nn.Linear):
        pass

    model = _xfuser_model(_tiny_model(), xfuser_qwenimage21.xFuserQwenImage21AttnProcessor)
    wrapped, plain = (block.img_mlp for block in model.transformer_blocks)
    wrapped.proj.__class__ = WrappedLinear

    assert xfuser_qwenimage21.fuse_swiglu_projections(model) == NUM_LAYERS - 1

    assert type(wrapped) is qwenimage21.QwenImage21SwiGLUFeedForward and not hasattr(wrapped, "gate_proj")
    assert type(plain) is xfuser_qwenimage21._FusedSwiGLUFeedForward
    assert plain.gate_proj.weight.dtype == plain.out.weight.dtype
    # Fusing twice leaves the fused feed-forward as it is.
    assert xfuser_qwenimage21.fuse_swiglu_projections(model) == 0
