"""The SkyReels-V2 wrapper against the stock diffusers transformer, on CPU.

Sequence parallelism is faked at degree 1 by installing a stand-in SP group and
runtime state, so the wrapper runs its real split/attention code path without a
process group.
"""

from types import SimpleNamespace

import pytest
import torch

_skyreels = pytest.importorskip(
    "diffusers.models.transformers.transformer_skyreels_v2",
    reason="installed diffusers does not include SkyReels-V2",
)
if not hasattr(_skyreels, "SkyReelsV2AttnProcessor"):
    pytest.skip("installed diffusers predates SkyReelsV2Attention", allow_module_level=True)

from diffusers.models.transformers.transformer_skyreels_v2 import (  # noqa: E402
    SkyReelsV2Transformer3DModel,
)

from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.models.transformers.transformer_skyreels_v2 import (  # noqa: E402
    xFuserSkyReelsV2Transformer3DWrapper,
)

TEXT_DIM = 16
IMAGE_DIM = 8


def _config(i2v=False, **overrides):
    config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 8 if i2v else 4,
        "out_channels": 4,
        "text_dim": TEXT_DIM,
        "freq_dim": 16,
        "ffn_dim": 32,
        "num_layers": 2,
        "rope_max_seq_len": 32,
    }
    if i2v:
        config.update(image_dim=IMAGE_DIM, added_kv_proj_dim=24)
    config.update(overrides)
    return config


def _inputs(config, i2v=False, timestep=None):
    generator = torch.Generator().manual_seed(0)
    inputs = {
        "hidden_states": torch.randn(1, config["in_channels"], 3, 4, 6, generator=generator),
        "timestep": torch.tensor([500.0]) if timestep is None else timestep,
        # The I2V processor treats the last 512 context tokens as text.
        "encoder_hidden_states": torch.randn(1, 512 if i2v else 7, TEXT_DIM, generator=generator),
    }
    if i2v:
        inputs["encoder_hidden_states_image"] = torch.randn(1, 5, IMAGE_DIM, generator=generator)
    return inputs


@pytest.fixture
def single_rank(monkeypatch):
    """Install a degree-1 SP group and a runtime state that selects SDPA."""

    def install(world_size=1, ring_world_size=1):
        sp_group = SimpleNamespace(
            world_size=world_size,
            rank_in_group=0,
            ulysses_world_size=world_size // ring_world_size,
            ulysses_rank=0,
            ring_world_size=ring_world_size,
            ring_rank=0,
        )
        runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA,
            get_cross_attention_backend=lambda: AttentionBackendType.SDPA,
            fp8_comms=None,
            runtime_config=SimpleNamespace(use_sparseattn_head_balance=False),
            increment_step_counter=lambda: None,
        )
        monkeypatch.setattr("xfuser.core.distributed.parallel_state._SP", sp_group)
        monkeypatch.setattr("xfuser.core.distributed.runtime_state._RUNTIME", runtime)

    install()
    return install


def _pair(config):
    torch.manual_seed(0)
    reference = SkyReelsV2Transformer3DModel(**config).eval()
    wrapped = xFuserSkyReelsV2Transformer3DWrapper(**config).eval()
    wrapped.load_state_dict(reference.state_dict())
    return reference, wrapped


@pytest.mark.parametrize(
    "i2v, overrides, extra",
    [
        (False, {}, {}),
        (True, {}, {}),
        (False, {"inject_sample_info": True}, {"fps": [1]}),
    ],
    ids=["t2v", "i2v", "fps-embedding"],
)
def test_wrapper_matches_diffusers_at_one_rank(single_rank, i2v, overrides, extra):
    config = _config(i2v=i2v, **overrides)
    reference, wrapped = _pair(config)
    inputs = {**_inputs(config, i2v=i2v), **extra}

    with torch.no_grad():
        expected = reference(**inputs).sample
        actual = wrapped(**inputs).sample

    assert actual.shape == inputs["hidden_states"].shape[:1] + (4,) + inputs["hidden_states"].shape[2:]
    torch.testing.assert_close(actual, expected)


def test_diffusion_forcing_is_refused(single_rank):
    config = _config()
    _, wrapped = _pair(config)
    inputs = _inputs(config, timestep=torch.tensor([[500.0, 400.0, 300.0]]))

    with pytest.raises(NotImplementedError, match="diffusion forcing"):
        with torch.no_grad():
            wrapped(**inputs, enable_diffusion_forcing=True)


def test_block_causal_attention_is_refused(single_rank):
    config = _config(num_frame_per_block=3)
    _, wrapped = _pair(config)

    with pytest.raises(NotImplementedError, match="block-causal"):
        with torch.no_grad():
            wrapped(**_inputs(config))


def test_ring_attention_refuses_a_sequence_it_would_have_to_pad(single_rank):
    """18 tokens over 4 ranks needs padding, and ring attention cannot drop padded keys."""
    single_rank(world_size=4, ring_world_size=2)
    config = _config()
    _, wrapped = _pair(config)

    with pytest.raises(ValueError, match="divisible by the sequence-parallel degree"):
        with torch.no_grad():
            wrapped(**_inputs(config))
