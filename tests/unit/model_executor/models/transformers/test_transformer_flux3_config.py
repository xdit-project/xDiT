"""The FLUX.3 wrapper must build from a checkpoint config exactly as the stock transformer does.

diffusers picks the config keys it passes to ``__init__`` from the class signature, so a
wrapper that hid upstream's parameters would build every checkpoint with defaults; a
flow-map (step-distilled) checkpoint would lose its target-timestep embedder.
"""

from types import SimpleNamespace

import pytest

transformer_flux3 = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_flux3")

TINY = {
    "in_channels": 8,
    "cond_in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "timestep_channels": 32,
    "axes_dims_rope": (4, 4, 4, 4),
    "rope_theta": 2000,
}


@pytest.mark.parametrize("flow_map", [True, False])
def test_from_config_keeps_every_checkpoint_setting(flow_map):
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

    stock = Flux3Transformer2DModel(**TINY, flow_map=flow_map)

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper.from_config(stock.config)

    assert dict(wrapper.config) == dict(stock.config)
    assert (wrapper.target_timestep_embedder is not None) == flow_map
    assert wrapper.state_dict().keys() == stock.state_dict().keys()


@pytest.mark.parametrize(
    "name",
    [
        "XFUSER_FLUX3_FUSED_MXFP4_QUANT",
        "XFUSER_FLUX3_FUSED_QK_NORM_ROPE",
        "XFUSER_FLUX3_FUSED_BLOCK_RESIDUAL",
        "XFUSER_FLUX3_SEQ_MAJOR_ULYSSES",
    ],
)
def test_environment_switches_each_fusion_off(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert transformer_flux3._env_flag(name)
    for value, enabled in [("1", True), (" on ", True), ("0", False), ("false", False), ("OFF", False)]:
        monkeypatch.setenv(name, value)
        assert transformer_flux3._env_flag(name) is enabled


def test_each_transformer_call_advances_the_step_schedule(monkeypatch):
    # The hybrid GEMM schedule switches precision per step from this counter.
    calls = []
    state = SimpleNamespace(increment_step_counter=lambda: calls.append(1))
    monkeypatch.setattr(transformer_flux3, "get_runtime_state", lambda: state)
    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)

    for hook in wrapper._forward_pre_hooks.values():
        hook(wrapper, ())

    assert len(calls) == 1


def test_fusing_stacks_the_four_input_projections_of_every_block():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    block = wrapper.single_transformer_blocks[0]
    layers = (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.ff.linear_in)
    x = torch.randn(3, layers[0].in_features)
    expected = torch.cat([layer(x) for layer in layers], dim=-1)

    assert transformer_flux3.fuse_block_input_projections(wrapper) == 4

    assert type(block).__name__ == "_FusedFlux3SingleTransformerBlock"
    assert block.fused_in_split == [layer.out_features for layer in layers]
    assert not hasattr(block.attn, "to_q") and not hasattr(block.ff, "linear_in")
    torch.testing.assert_close(block.fused_in(x), expected)
    assert type(wrapper.mode_blocks["image"][0]).__name__ == "_FusedFlux3ModeTransformerBlock"


def test_fusing_joins_the_attention_and_feed_forward_outputs_of_fused_blocks():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    block = wrapper.mode_blocks["image"][0]
    to_out, linear_out = block.attn.to_out[0], block.ff.linear_out
    attention, hidden = torch.randn(3, to_out.in_features), torch.randn(3, linear_out.in_features)
    expected = to_out(attention) + linear_out(hidden)

    assert transformer_flux3.fuse_block_output_projections(wrapper) == 0  # needs the input fusion first
    transformer_flux3.fuse_block_input_projections(wrapper)
    assert transformer_flux3.fuse_block_output_projections(wrapper) == 4
    assert transformer_flux3.fuse_block_output_projections(wrapper) == 0

    assert not hasattr(block.attn, "to_out") and not hasattr(block.ff, "linear_out")
    torch.testing.assert_close(block.fused_out(torch.cat([attention, hidden], dim=-1)), expected)


def test_fusing_skips_blocks_whose_layers_cannot_be_stacked():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    block = wrapper.txt_mode_blocks[0]
    block.attn.to_k = torch.nn.Linear(block.attn.to_k.in_features, block.attn.to_k.out_features, bias=True)

    assert transformer_flux3.fuse_block_input_projections(wrapper) == 3
    assert type(block).__name__ == "Flux3ModeTransformerBlock"
    assert hasattr(block.attn, "to_q")


@pytest.mark.parametrize(
    ("start", "length", "expected_rows"),
    [
        (0, 4, [0, 0, 0, 0]),  # image only
        (4, 4, [0, 0, 1, 1]),  # straddles image and reference
        (8, 4, [1, 1, 1, 1]),  # reference, then SP pad
        (2, 0, []),  # a rank holding only text
    ],
)
def test_content_modulation_gives_each_position_its_stream_row(start, length, expected_rows):
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    rows = [torch.full((1, 1, 3), float(index)) for index in range(2)]

    mod = wrapper._content_modulation(rows, [6, 4], start=start, length=length)

    assert mod.shape == (1, length, 3)
    assert mod[0, :, 0].tolist() == expected_rows


def test_block_inputs_keep_one_address_per_shape():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    first = wrapper._block_input("joint.cos", torch.ones(4, 2))
    second = wrapper._block_input("joint.cos", torch.zeros(4, 2))
    other_shape = wrapper._block_input("joint.cos", torch.zeros(5, 2))

    assert second.data_ptr() == first.data_ptr() and torch.equal(second, torch.zeros(4, 2))
    assert other_shape.data_ptr() != first.data_ptr()
    assert getattr(first, "_dynamo_static_input_type", None) == "unguarded"


def test_step_constants_recompute_for_a_new_or_modified_source():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    calls = []

    def compute(x):
        calls.append(1)
        return x * 2

    ids = torch.ones(3)
    with torch.no_grad():
        first = wrapper._step_constant("ids", ids, compute)
        assert wrapper._step_constant("ids", ids, compute) is first and len(calls) == 1
        ids.add_(1)
        assert torch.equal(wrapper._step_constant("ids", ids, compute), torch.full((3,), 4.0)) and len(calls) == 2
        wrapper._step_constant("ids", torch.ones(3), compute)
        assert len(calls) == 3
    wrapper._step_constant("ids", ids, compute)
    wrapper._step_constant("ids", ids, compute)
    assert len(calls) == 5
    with torch.inference_mode():
        inference_ids = torch.ones(3)
        first = wrapper._step_constant("ids", inference_ids, compute)
        assert wrapper._step_constant("ids", inference_ids, compute) is first and len(calls) == 6


@pytest.mark.parametrize(("change", "asks"), [({}, True), ({"local": True}, False), ({"disabled": True}, False)])
def test_the_processor_takes_usps_seq_major_plan_unless_local_or_switched_off(monkeypatch, change, asks):
    plan = object()
    calls = []
    monkeypatch.setattr(
        transformer_flux3,
        "usp_seq_major_plan",
        lambda attn, heads, attention_kwargs: calls.append((attn, heads, attention_kwargs)) or plan,
    )
    processor = transformer_flux3.xFuserFlux3AttnProcessor()
    processor.local = change.get("local", False)
    if change.get("disabled"):
        processor.seq_major_ulysses = False
    attn = SimpleNamespace(heads=8)
    attention_kwargs = {"valid_kv_len": 5}

    assert processor._seq_major_plan(attn, attention_kwargs) is (plan if asks else None)
    assert calls == ([(attn, 8, attention_kwargs)] if asks else [])
