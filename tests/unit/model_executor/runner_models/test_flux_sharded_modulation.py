"""FLUX.1 runners share the blocks' timestep modulation out over the sequence-parallel ranks.

Each rank projects a share of the modulation and the ranks gather it, which must not change
what the transformer computes; it stays off where a rank does not run the whole transformer
on whole, plain weights.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers.models.transformers.transformer_flux")

from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.layers import plain_linear as plain_linear_module, sharded_modulation  # noqa: E402
from xfuser.model_executor.layers.sharded_modulation import SlicedModulation  # noqa: E402
from xfuser.model_executor.models.runner_models import base_model  # noqa: E402
from xfuser.model_executor.models.runner_models.flux import (  # noqa: E402
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
    """A tiny FLUX.1 transformer on one rank, with USP and the gather stubbed at the boundary."""
    runtime = SimpleNamespace(
        fp8_comms=None,
        split_text_embed_in_sp=True,
        attention_backend=AttentionBackendType.SDPA,
        num_pipeline_patch=1,
        max_condition_sequence_length=0,
    )
    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor)
    monkeypatch.setattr(flux, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(flux, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(flux, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_ring_parallel_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(flux, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(flux, "get_sp_group", lambda: group)
    monkeypatch.setattr(flux, "get_cfg_group", lambda: group)
    monkeypatch.setattr(flux, "USP", _local_usp)
    monkeypatch.setattr(flux, "_HAS_FLYDSL", False)
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(sharded_modulation, "get_sp_group", lambda: group)
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


def _forward(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        (sample,) = model(
            torch.randn(1, 6, 8, generator=generator),
            encoder_hidden_states=torch.randn(1, 4, 12, generator=generator),
            pooled_projections=torch.randn(1, 8, generator=generator),
            timestep=torch.rand(1, generator=generator),
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
        cache_method=None,
    )
    vars(runner.config).update(config)
    runner.pipe = SimpleNamespace(transformer=transformer)
    return runner


@pytest.fixture
def eight_ranks(monkeypatch):
    monkeypatch.setattr(base_model, "get_sequence_parallel_world_size", lambda: 8)
    monkeypatch.setattr(base_model.PACKAGES_CHECKER, "_on_rdna4", lambda: False)


def test_sharding_keeps_the_transformer_output(transformer):
    expected = [_forward(transformer, seed) for seed in (1, 2)]
    state = set(transformer.state_dict())

    # Two double blocks project the image and the prompt stream's modulation, two single blocks one each.
    assert transformer.shard_modulation() == 6

    assert all(type(linear) is SlicedModulation for linear in flux.flux_modulation_linears(transformer))
    # A second embedding must reach the blocks too: the row is projected on every forward.
    for seed, output in zip((1, 2), expected):
        assert torch.allclose(_forward(transformer, seed), output, atol=1e-5)
    assert set(transformer.state_dict()) == state


@pytest.mark.parametrize("kind", ["replaced", "quantized-in-place", "sharded"])
def test_modulation_that_is_not_all_plain_linears_stays_as_it_is(monkeypatch, transformer, kind):
    """Quantized GEMMs replace the linear; torchao quantizes its weight in place and FSDP shards it."""

    class QuantizedLinear(torch.nn.Linear):
        pass

    norm = transformer.single_transformer_blocks[0].norm
    if kind == "replaced":
        norm.linear = QuantizedLinear(norm.linear.in_features, norm.linear.out_features)
    elif kind == "sharded":
        norm.linear.weight.placements = ("shard",)
    else:
        monkeypatch.setattr(plain_linear_module, "_weight_is_torchao_quantized", lambda module: module is norm.linear)

    assert transformer.shard_modulation() == 0
    assert not any(type(linear) is SlicedModulation for linear in flux.flux_modulation_linears(transformer))
    assert not hasattr(transformer, "sharded_modulation")


@pytest.mark.parametrize("model_cls", [xFuserFluxModel, xFuserFluxKontextModel])
def test_the_post_load_step_shards_once_the_weights_are_placed(monkeypatch, eight_ranks, transformer, model_cls):
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
        materialize_pipeline=lambda: placed.append(hasattr(transformer, "sharded_modulation"))
    )

    runner._post_load_and_state_initialization({})

    assert placed == [False]
    assert hasattr(transformer, "sharded_modulation")


@pytest.mark.parametrize(
    "config",
    [
        {"pipefusion_parallel_degree": 2},
        {"fully_shard_degree": 2},
        {"memory_efficient_sharding": True},
        {"enable_group_cpu_offload": True},
        {"enable_sequential_cpu_offload": True},
        {"enable_model_cpu_offload": True},
        {"cache_method": "dbcache"},
    ],
)
def test_runs_without_the_whole_transformer_on_whole_weights_do_not_shard(eight_ranks, transformer, config):
    _runner(xFuserFluxModel, transformer, **config)._shard_modulation()

    assert not hasattr(transformer, "sharded_modulation")


def test_one_rank_or_rdna4_does_not_shard(monkeypatch, transformer):
    monkeypatch.setattr(base_model.PACKAGES_CHECKER, "_on_rdna4", lambda: False)
    monkeypatch.setattr(base_model, "get_sequence_parallel_world_size", lambda: 1)
    _runner(xFuserFluxModel, transformer)._shard_modulation()
    assert not hasattr(transformer, "sharded_modulation")

    monkeypatch.setattr(base_model.PACKAGES_CHECKER, "_on_rdna4", lambda: True)
    monkeypatch.setattr(base_model, "get_sequence_parallel_world_size", lambda: 8)
    _runner(xFuserFluxModel, transformer)._shard_modulation()
    assert not hasattr(transformer, "sharded_modulation")
