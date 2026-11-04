"""HunyuanVideo under Ulysses, ring and both must match one unsplit diffusers forward.

Each rank builds the same tiny random model and compares the xDiT wrapper's
gathered output with diffusers' HunyuanVideoTransformer3DModel run unsplit on
the same device. When the video token count is not divisible by the
sequence-parallel degree, the wrapper has to pad the sequence and keep the
padded keys out of attention, also in the ring steps that each see only part of
the keys; a divisible case guards the unpadded path. Ulysses needs yunchang's
long-context attention, which is only enabled with an accelerator, so these
ranks run on GPUs. Nothing is downloaded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.multi_gpu

_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_attention_heads=6,
    attention_head_dim=8,
    num_layers=1,
    num_single_layers=1,
    num_refiner_layers=1,
    patch_size=1,
    patch_size_t=1,
    text_embed_dim=16,
    pooled_projection_dim=8,
    rope_axes_dim=(2, 2, 4),
)
_TEXT_LEN = 7


def _hunyuan_video_worker(rank, world_size, init_method, ulysses_degree, ring_degree, frames, height, width):
    import torch
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from xfuser.model_executor.models.transformers.transformer_hunyuan_video import (
        xFuserHunyuanVideoTransformer3DWrapper,
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
        sequence_parallel_degree=world_size, ulysses_degree=ulysses_degree, ring_degree=ring_degree
    )
    try:
        torch.manual_seed(0)
        reference = HunyuanVideoTransformer3DModel(**_CONFIG).eval()
        wrapper = xFuserHunyuanVideoTransformer3DWrapper.from_config(reference.config).eval()
        wrapper.load_state_dict(reference.state_dict())
        reference, wrapper = reference.to(device), wrapper.to(device)

        generator = torch.Generator().manual_seed(1)
        # The last text token is masked for the whole batch, so the wrapper drops it.
        mask = torch.ones(1, _TEXT_LEN, dtype=torch.int64)
        mask[:, -1] = 0
        inputs = dict(
            hidden_states=torch.randn(1, _CONFIG["in_channels"], frames, height, width, generator=generator).to(device),
            timestep=torch.tensor([700], device=device),
            encoder_hidden_states=torch.randn(1, _TEXT_LEN, _CONFIG["text_embed_dim"], generator=generator).to(device),
            encoder_attention_mask=mask.to(device),
            pooled_projections=torch.randn(1, _CONFIG["pooled_projection_dim"], generator=generator).to(device),
            guidance=torch.tensor([6000.0], device=device),
            return_dict=False,
        )

        usp_runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA_EFFICIENT,
            runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
            fp8_comms=None,
        )
        # The real runtime state needs a pipeline; the wrapper reads only these fields.
        model_runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA_EFFICIENT,
            increment_step_counter=lambda: None,
            split_text_embed_in_sp=False,
            max_condition_sequence_length=_TEXT_LEN,
        )
        with (
            patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=usp_runtime),
            patch(
                "xfuser.model_executor.models.transformers.transformer_hunyuan_video.get_runtime_state",
                return_value=model_runtime,
            ),
            torch.no_grad(),
        ):
            expected = reference(**inputs)[0]
            actual = wrapper(**inputs)[0]

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


# 27 video tokens do not divide 2 or 4 and 8 do not divide 3; 18 divide evenly by 2.
@pytest.mark.parametrize(
    ("ulysses_degree", "ring_degree", "frames", "height", "width"),
    [
        (2, 1, 3, 3, 3),
        (3, 1, 2, 2, 2),
        (1, 2, 3, 3, 3),
        (2, 2, 3, 3, 3),
        (2, 1, 2, 3, 3),
    ],
    ids=["ulysses2-padded", "ulysses3-padded", "ring2-padded", "ulysses2-ring2-padded", "ulysses2-divisible"],
)
def test_sequence_parallel_forward_matches_unsplit_diffusers(
    ulysses_degree, ring_degree, frames, height, width, accelerator_ranks
):
    pytest.importorskip("diffusers")
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    spec = registry.get(AttentionBackendType.SDPA_EFFICIENT)
    reason = spec.unavailable() or (spec.ring.unmet() if ring_degree > 1 else None)
    if reason is not None:
        pytest.skip(reason)
    accelerator_ranks(
        _hunyuan_video_worker,
        world_size=ulysses_degree * ring_degree,
        init_filename=f"hunyuan-video-u{ulysses_degree}-r{ring_degree}-{frames}x{height}x{width}",
        args=(ulysses_degree, ring_degree, frames, height, width),
    )
