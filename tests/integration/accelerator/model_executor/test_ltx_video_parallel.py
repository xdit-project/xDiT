"""LTX-Video under Ulysses and CFG parallelism must match one unsplit diffusers forward.

Each rank builds the same tiny random model and compares the xDiT wrapper's
gathered output with diffusers' LTXVideoTransformer3DModel run unsplit on the
same device. The token count is odd, so Ulysses has to pad the sequence and keep
the padded keys out of attention; prompts of different lengths and an
image-to-video per-token timestep exercise the CFG slicing of every batch-shaped
input. Ulysses needs yunchang's long-context attention, which is only enabled
with an accelerator, so these ranks run on GPUs. Nothing is downloaded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.multi_gpu

_CONFIG = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=32,
    num_layers=2,
    caption_channels=24,
)
_FRAMES, _HEIGHT, _WIDTH = 3, 3, 3  # 27 tokens
_TEXT_LEN = 6


def _ltx_video_worker(rank, world_size, init_method, cfg_degree, ulysses_degree):
    import torch
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from xfuser.model_executor.models.transformers.transformer_ltx_video import (
        xFuserLTXVideoTransformer3DWrapper,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        sequence_parallel_degree=ulysses_degree,
        ulysses_degree=ulysses_degree,
    )
    try:
        torch.manual_seed(0)
        reference = LTXVideoTransformer3DModel(**_CONFIG).eval()
        wrapper = xFuserLTXVideoTransformer3DWrapper.from_config(reference.config).eval()
        wrapper.load_state_dict(reference.state_dict())
        reference, wrapper = reference.to(device), wrapper.to(device)

        generator = torch.Generator().manual_seed(1)
        tokens = _FRAMES * _HEIGHT * _WIDTH
        mask = torch.zeros(2, _TEXT_LEN, dtype=torch.int64)
        mask[0, :2] = 1
        mask[1, :5] = 1
        timestep = torch.full((2, tokens), 800.0)
        timestep[:, : _HEIGHT * _WIDTH] = 0.0
        inputs = dict(
            hidden_states=torch.randn(2, tokens, _CONFIG["in_channels"], generator=generator).to(device),
            encoder_hidden_states=torch.randn(2, _TEXT_LEN, _CONFIG["caption_channels"], generator=generator).to(
                device
            ),
            timestep=timestep.to(device),
            encoder_attention_mask=mask.to(device),
            num_frames=_FRAMES,
            height=_HEIGHT,
            width=_WIDTH,
            rope_interpolation_scale=(8 / 25, 32, 32),
            return_dict=False,
        )

        runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA,
            runtime_config=SimpleNamespace(use_sparseattn_head_balance=False),
            fp8_comms=None,
        )
        with patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime), torch.no_grad():
            expected = reference(**inputs)[0]
            actual = wrapper(**inputs)[0]

        assert actual.shape == (2, tokens, _CONFIG["out_channels"])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    ("cfg_degree", "ulysses_degree"),
    [(1, 2), (2, 2)],
    ids=["ulysses2", "cfg2-ulysses2"],
)
def test_parallel_forward_matches_unsplit_diffusers(cfg_degree, ulysses_degree, accelerator_ranks):
    pytest.importorskip("diffusers")
    accelerator_ranks(
        _ltx_video_worker,
        world_size=cfg_degree * ulysses_degree,
        init_filename=f"ltx-video-cfg{cfg_degree}-u{ulysses_degree}",
        args=(cfg_degree, ulysses_degree),
    )
