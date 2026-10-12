"""xDiT's Chroma transformer matches stock diffusers on one rank.

The wrapper and its attention processor replace every block's attention and move the
text mask into a precomputed bias, so on a single rank the output must be identical
to diffusers' ChromaTransformer2DModel, mask included. Runs a tiny random config on
CPU against a real single-rank Gloo group; nothing is downloaded.
"""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_chroma")

from diffusers import ChromaTransformer2DModel  # noqa: E402

from xfuser.config.args import xFuserArgs  # noqa: E402
from xfuser.core.distributed import parallel_state, runtime_state  # noqa: E402
from xfuser.model_executor.models.transformers.transformer_chroma import (  # noqa: E402
    xFuserChromaAttnProcessor,
    xFuserChromaTransformer2DWrapper,
)

_TINY_CONFIG = dict(
    patch_size=1,
    in_channels=8,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=16,
    num_attention_heads=4,
    joint_attention_dim=32,
    axes_dims_rope=(4, 6, 6),
    approximator_num_channels=16,
    approximator_hidden_dim=32,
    approximator_layers=1,
)


@pytest.fixture
def single_rank(tmp_path, monkeypatch):
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    # Restore whatever runtime state the session had once this test is done.
    monkeypatch.setattr(runtime_state, "_RUNTIME", None)
    with patch.object(parallel_state, "set_device"):
        parallel_state.init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist-init'}",
        )
    try:
        parallel_state.initialize_model_parallel(backend="gloo")
        args = xFuserArgs(attention_backend="SDPA")
        engine_config, _ = args.create_config()
        runtime_state.initialize_runtime_state(engine_config=engine_config)
        yield
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def _inputs(dtype, batch=2, num_txt=7, height=4, width=5):
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(batch, height * width, 8, generator=generator).to(dtype)
    encoder_hidden_states = torch.randn(batch, num_txt, 32, generator=generator).to(dtype)
    img_ids = torch.zeros(height, width, 3)
    img_ids[..., 1] = torch.arange(height)[:, None]
    img_ids[..., 2] = torch.arange(width)[None, :]
    # As ChromaPipeline builds it: model-dtype text mask (prompt + one pad), extended
    # with ones for the image tokens. Rows differ so a per-sample mismatch shows.
    valid = torch.tensor([[2], [4]])[:batch]
    text_mask = (torch.arange(num_txt)[None] <= valid).to(dtype)
    attention_mask = torch.cat([text_mask, torch.ones(batch, height * width, dtype=torch.bool)], dim=1)
    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=torch.tensor([0.7, 0.3][:batch], dtype=dtype),
        img_ids=img_ids.reshape(-1, 3).to(dtype),
        txt_ids=torch.zeros(num_txt, 3, dtype=dtype),
        attention_mask=attention_mask,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_mask", [True, False])
def test_wrapper_matches_diffusers_on_one_rank(single_rank, dtype, with_mask):
    torch.manual_seed(0)
    reference = ChromaTransformer2DModel(**_TINY_CONFIG).to(dtype).eval()
    wrapper = xFuserChromaTransformer2DWrapper.from_config(reference.config).to(dtype).eval()
    wrapper.load_state_dict(reference.state_dict())
    processors = {type(p) for p in wrapper.attn_processors.values()}
    assert processors == {xFuserChromaAttnProcessor}

    inputs = _inputs(dtype)
    if not with_mask:
        inputs["attention_mask"] = None
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = wrapper(**inputs).sample

    assert torch.equal(actual, expected)
