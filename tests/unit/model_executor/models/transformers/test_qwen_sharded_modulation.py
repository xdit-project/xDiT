"""Qwen-Image shares the blocks' timestep modulation out over the sequence-parallel ranks.

Each rank projects a share of the modulation and the ranks gather it, which must not change
what the transformer computes. The runner turns it on for Qwen-Image only.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.layers import sharded_modulation
from xfuser.model_executor.layers.sharded_modulation import SlicedModulation
from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.qwen import (
    xFuserQwenImageEditModel,
    xFuserQwenImageModel,
)
from xfuser.model_executor.models.transformers import transformer_qwen as qwen


def _local_usp(query, key, value, **_):
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture(autouse=True)
def one_rank(monkeypatch):
    """One rank, with USP and the gathers stubbed at the boundary."""
    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(qwen, "get_runtime_state", lambda: SimpleNamespace(fp8_comms=None, num_pipeline_patch=1))
    monkeypatch.setattr(qwen, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(qwen, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(qwen, "gather_and_unpad", lambda x, pad_amount, dim: x)
    monkeypatch.setattr(qwen, "USP", _local_usp)
    monkeypatch.setattr(qwen, "_HAS_FLYDSL", False)
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(sharded_modulation, "get_sp_group", lambda: group)


def _model(**config):
    torch.manual_seed(0)
    return qwen.xFuserQwenImageTransformerWrapper(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        **config,
    ).eval()


def _forward(model, seed, img_shapes):
    generator = torch.Generator().manual_seed(seed)
    tokens = sum(height * width for _, height, width in img_shapes)
    with torch.no_grad():
        (sample,) = model(
            torch.randn(1, tokens, 16, generator=generator),
            encoder_hidden_states=torch.randn(1, 4, 16, generator=generator),
            timestep=torch.rand(1, generator=generator),
            img_shapes=[img_shapes],
            return_dict=False,
        )
    return sample


@pytest.mark.parametrize(
    ("config", "img_shapes"),
    [
        pytest.param({}, [(1, 2, 3)], id="qwen_image"),
        # The edit transformer modulates the reference image's tokens at timestep 0: its image
        # linears project both embeddings, its prompt linears only the first.
        pytest.param({"zero_cond_t": True}, [(1, 2, 3), (1, 2, 2)], id="zero_cond_t"),
    ],
)
def test_sharding_keeps_the_transformer_output(config, img_shapes):
    model = _model(**config)
    expected = [_forward(model, seed, img_shapes) for seed in (1, 2)]
    state = set(model.state_dict())

    # Each of the two blocks projects the image and the prompt stream's modulation.
    assert model.shard_modulation() == 4

    assert all(type(linear) is SlicedModulation for linear in qwen.qwen_modulation_linears(model))
    # A second embedding must reach the blocks too: the row is projected on every forward.
    for seed, output in zip((1, 2), expected):
        assert torch.allclose(_forward(model, seed, img_shapes), output, atol=1e-5)
    assert set(model.state_dict()) == state


def _runner(model_cls, transformer):
    runner = object.__new__(model_cls)
    runner.config = SimpleNamespace(
        fully_shard_degree=1,
        memory_efficient_sharding=False,
        enable_group_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
        pipefusion_parallel_degree=1,
        cache_method=None,
        use_hybrid_attn_schedule=False,
        use_hybrid_gemm_schedule=False,
        solattn_beta_schedule=None,
        use_vae_channels_last_format=False,
    )
    runner.pipe = SimpleNamespace(transformer=transformer)
    return runner


@pytest.fixture
def eight_ranks(monkeypatch):
    monkeypatch.setattr(base_model, "get_sequence_parallel_world_size", lambda: 8)
    monkeypatch.setattr(base_model.PACKAGES_CHECKER, "_on_rdna4", lambda: False)


def test_qwen_image_shards_once_the_weights_are_placed(eight_ranks):
    model = _model()
    runner = _runner(xFuserQwenImageModel, model)
    placed = []
    runner.loader = SimpleNamespace(materialize_pipeline=lambda: placed.append(hasattr(model, "sharded_modulation")))

    runner._post_load_and_state_initialization({})

    assert placed == [False]
    assert hasattr(model, "sharded_modulation")


def test_qwen_image_edit_keeps_its_modulation(monkeypatch, eight_ranks):
    model = _model(zero_cond_t=True)
    runner = _runner(xFuserQwenImageEditModel, model)
    runner.loader = SimpleNamespace(materialize_pipeline=lambda: None)
    monkeypatch.setattr(xFuserQwenImageEditModel, "_batches_true_cfg", lambda self: False)

    runner._post_load_and_state_initialization({})

    assert not hasattr(model, "sharded_modulation")
