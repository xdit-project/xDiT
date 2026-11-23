"""Runners that opt in compute Q, K and V with one fused projection per attention layer.

Fusing swaps three GEMMs for one three times as wide and must not change what the
transformer computes; it stays off where the projections are not plain linears on this
rank (quantized or sharded weights, CPU offload) and where the runner does not opt in.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers.models.transformers.transformer_flux")

from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.layers import plain_linear as plain_linear_module  # noqa: E402
from xfuser.model_executor.models.runner_models import base_model, flux as flux_runner  # noqa: E402
from xfuser.model_executor.models.runner_models.flux import (  # noqa: E402
    xFuserFlux2Model,
    xFuserFluxKontextModel,
    xFuserFluxModel,
)
from xfuser.model_executor.models.transformers import transformer_flux as flux  # noqa: E402


def _local_usp(query, key, value, **_):
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    """The runner logs from the last rank, read from the launcher's environment."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.fixture
def transformer(monkeypatch):
    """A tiny FLUX.1 transformer on one rank, with USP stubbed at the boundary."""
    runtime = SimpleNamespace(
        fp8_comms=None,
        split_text_embed_in_sp=True,
        attention_backend=AttentionBackendType.SDPA,
        num_pipeline_patch=1,
        max_condition_sequence_length=0,
    )
    monkeypatch.setattr(flux, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(flux, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(flux, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_ring_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(flux, "get_sp_group", lambda: SimpleNamespace(all_gather=lambda sample, dim: sample))
    monkeypatch.setattr(flux, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda sample, dim: sample))
    monkeypatch.setattr(flux, "USP", _local_usp)
    monkeypatch.setattr(flux, "_HAS_FLYDSL", False)
    torch.manual_seed(0)
    return flux.xFuserFlux1Transformer2DWrapper(
        in_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 6, 6),
    ).eval()


def _forward(model):
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        (sample,) = model(
            torch.randn(1, 6, 8, generator=generator),
            encoder_hidden_states=torch.randn(1, 4, 12, generator=generator),
            pooled_projections=torch.randn(1, 8, generator=generator),
            timestep=torch.tensor([0.5]),
            img_ids=torch.randn(6, 3, generator=generator),
            txt_ids=torch.randn(4, 3, generator=generator),
            return_dict=False,
        )
    return sample


def _runner(model_cls, transformer=None, **config):
    runner = object.__new__(model_cls)
    runner.config = SimpleNamespace(
        fully_shard_degree=1,
        memory_efficient_sharding=False,
        enable_group_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
        pipefusion_parallel_degree=1,
    )
    vars(runner.config).update(config)
    runner.pipe = SimpleNamespace(transformer=transformer)
    return runner


def _attention_layers(model):
    return [block.attn for block in (*model.transformer_blocks, *model.single_transformer_blocks)]


def test_fusing_keeps_the_transformer_output(transformer):
    expected = _forward(transformer)

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    layers = _attention_layers(transformer)
    assert [layer.fused_projections for layer in layers] == [True] * 4
    # Double blocks also project the prompt stream.
    assert [hasattr(layer, "to_added_qkv") for layer in layers] == [True, True, False, False]
    # Single blocks compute their MLP projection with Q/K/V.
    assert [block.to_qkv_mlp.out_features for block in transformer.single_transformer_blocks] == [7 * 32] * 2
    assert torch.allclose(_forward(transformer), expected, atol=1e-5)


def test_a_single_block_whose_mlp_projection_is_not_a_plain_linear_keeps_its_forward(transformer):
    class QuantizedLinear(torch.nn.Linear):
        pass

    expected = _forward(transformer)
    first, second = transformer.single_transformer_blocks
    quantized = QuantizedLinear(first.proj_mlp.in_features, first.proj_mlp.out_features)
    quantized.load_state_dict(first.proj_mlp.state_dict())
    first.proj_mlp = quantized

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    assert not hasattr(first, "to_qkv_mlp") and "forward" not in vars(first)
    assert first.attn.fused_projections
    assert hasattr(second, "to_qkv_mlp")
    assert torch.allclose(_forward(transformer), expected, atol=1e-5)


def test_single_blocks_keep_their_forward_while_q_k_and_v_are_separate(transformer):
    assert flux.fold_mlp_into_qkv_projections(transformer) == 0
    assert not any("forward" in vars(block) for block in transformer.single_transformer_blocks)


def _weight_bytes(model):
    """Bytes of the distinct storages behind a model's parameters."""
    storages = {
        parameter.untyped_storage().data_ptr(): parameter.untyped_storage().nbytes() for parameter in model.parameters()
    }
    return sum(storages.values())


def test_fusing_costs_no_memory(transformer):
    before = _weight_bytes(transformer)

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    assert _weight_bytes(transformer) == before


def test_the_separate_projections_compute_the_same_from_the_fused_weights(transformer):
    layer = transformer.transformer_blocks[0].attn
    names = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
    sample = torch.randn(3, layer.to_q.in_features, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = [getattr(layer, name)(sample) for name in names]

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    with torch.no_grad():
        for name, output in zip(names, expected):
            assert torch.equal(getattr(layer, name)(sample), output), name


def test_a_layer_whose_projections_are_not_plain_linears_stays_separate(transformer):
    class QuantizedLinear(torch.nn.Linear):
        pass

    first = transformer.transformer_blocks[0].attn
    first.add_k_proj = QuantizedLinear(first.add_k_proj.in_features, first.add_k_proj.out_features)

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    assert [layer.fused_projections for layer in _attention_layers(transformer)] == [False, True, True, True]


@pytest.mark.parametrize("kind", ["quantized-in-place", "sharded"])
def test_linears_whose_weights_are_not_plain_tensors_are_left_alone(monkeypatch, transformer, kind):
    """torchao quantizes a weight in place and FSDP shards it: the module stays an ``nn.Linear`` either way."""
    double, single = transformer.transformer_blocks[0], transformer.single_transformer_blocks[0]
    expected = _forward(transformer)
    special = (double.attn.to_k, single.proj_mlp)
    if kind == "sharded":
        for linear in special:
            linear.weight.placements = ("shard",)
    else:
        monkeypatch.setattr(
            plain_linear_module,
            "_weight_is_torchao_quantized",
            lambda module: any(module is linear for linear in special),
        )

    _runner(xFuserFluxModel, transformer)._fuse_qkv_projections()

    assert [layer.fused_projections for layer in _attention_layers(transformer)] == [False, True, True, True]
    assert not hasattr(single, "to_qkv_mlp") and "forward" not in vars(single)
    assert torch.allclose(_forward(transformer), expected, atol=1e-5)


@pytest.mark.parametrize(
    "config",
    [
        {"fully_shard_degree": 2},
        {"memory_efficient_sharding": True},
        {"enable_group_cpu_offload": True},
        {"enable_sequential_cpu_offload": True},
        {"enable_model_cpu_offload": True},
    ],
)
def test_sharded_or_offloaded_weights_are_not_fused(transformer, config):
    _runner(xFuserFluxModel, transformer, **config)._fuse_qkv_projections()

    assert not any(layer.fused_projections for layer in _attention_layers(transformer))


@pytest.mark.parametrize(
    ("model_cls", "config", "rdna4", "expected"),
    [
        (xFuserFluxModel, {}, False, True),
        (xFuserFluxModel, {"pipefusion_parallel_degree": 2}, False, False),
        (xFuserFluxModel, {}, True, False),
        (xFuserFluxKontextModel, {}, False, True),
        (xFuserFluxKontextModel, {}, True, False),
        # FLUX.2 keeps the base default.
        (xFuserFlux2Model, {}, False, False),
    ],
)
def test_which_runs_fuse(monkeypatch, model_cls, config, rdna4, expected):
    monkeypatch.setattr(flux_runner.PACKAGES_CHECKER, "_on_rdna4", lambda: rdna4)

    assert _runner(model_cls, **config)._fuses_qkv_projections() is expected


@pytest.mark.parametrize(("model_cls", "fused"), [(xFuserFluxModel, True), (xFuserFlux2Model, False)])
def test_the_post_load_step_fuses_once_the_weights_are_placed(monkeypatch, transformer, model_cls, fused):
    monkeypatch.setattr(flux_runner.PACKAGES_CHECKER, "_on_rdna4", lambda: False)
    monkeypatch.setattr(base_model, "get_sequence_parallel_world_size", lambda: 1)
    runner = _runner(
        model_cls,
        transformer,
        use_hybrid_attn_schedule=False,
        use_hybrid_gemm_schedule=False,
        solattn_beta_schedule=None,
        use_vae_channels_last_format=False,
    )
    placed = []
    runner.loader = SimpleNamespace(
        materialize_pipeline=lambda: placed.append(
            any(layer.fused_projections for layer in _attention_layers(transformer))
        )
    )

    runner._post_load_and_state_initialization({})

    assert placed == [False]
    assert [layer.fused_projections for layer in _attention_layers(transformer)] == [fused] * 4
