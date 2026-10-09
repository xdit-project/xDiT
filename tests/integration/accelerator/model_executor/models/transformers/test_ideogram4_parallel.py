"""Ideogram 4 under Ulysses must match one unsplit diffusers forward.

Each rank builds the same tiny random transformer and compares the xDiT
wrapper's gathered output with diffusers' Ideogram4Transformer2DModel run
unsplit on the same device. Diffusers attends only within the real tokens of a
sample, so when the text + image token count does not divide the Ulysses degree
the zero-padded tokens xDiT appends must stay out of attention. Ulysses needs
yunchang's long-context attention, which is only enabled with an accelerator, so
these ranks run on GPUs. Nothing is downloaded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.multi_gpu

_CONFIG = dict(
    in_channels=8,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    intermediate_size=64,
    adaln_dim=16,
    llm_features_dim=24,
    rope_theta=10_000,
    mrope_section=(2, 1, 1),
)
_PAD_TOKENS = 2
_IMAGE_SIDE = 3  # 9 image tokens


def _ideogram4_worker(rank, world_size, init_method, num_text_tokens):
    import torch
    from diffusers.models.transformers.transformer_ideogram4 import (
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        SEQUENCE_PADDING_INDICATOR,
        Ideogram4Transformer2DModel,
    )

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from xfuser.model_executor.models.transformers.transformer_ideogram4 import (
        get_ideogram4_transformer_wrapper_class,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(sequence_parallel_degree=world_size, ulysses_degree=world_size)
    try:
        torch.manual_seed(0)
        reference = Ideogram4Transformer2DModel(**_CONFIG).eval()
        wrapper = get_ideogram4_transformer_wrapper_class().from_config(reference.config).eval()
        wrapper.load_state_dict(reference.state_dict())
        wrapper._init_sp_state()
        reference, wrapper = reference.to(device), wrapper.to(device)

        # The pipeline's layout: leading padding, then text, then image tokens.
        image_tokens = _IMAGE_SIDE * _IMAGE_SIDE
        length = _PAD_TOKENS + num_text_tokens + image_tokens
        text_start, image_start = _PAD_TOKENS, _PAD_TOKENS + num_text_tokens
        indicator = torch.full((1, length), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator[:, text_start:image_start] = LLM_TOKEN_INDICATOR
        indicator[:, image_start:] = OUTPUT_IMAGE_INDICATOR
        segment_ids = torch.ones(1, length, dtype=torch.long)
        segment_ids[:, :_PAD_TOKENS] = 0
        position_ids = torch.zeros(1, length, 3, dtype=torch.long)
        position_ids[0, text_start:image_start, 0] = torch.arange(num_text_tokens)
        rows, cols = torch.meshgrid(torch.arange(_IMAGE_SIDE), torch.arange(_IMAGE_SIDE), indexing="ij")
        position_ids[0, image_start:, 0] = num_text_tokens
        position_ids[0, image_start:, 1] = rows.flatten()
        position_ids[0, image_start:, 2] = cols.flatten()

        generator = torch.Generator().manual_seed(1)
        inputs = dict(
            hidden_states=torch.randn(1, length, _CONFIG["in_channels"], generator=generator).to(device),
            timestep=torch.tensor([0.3]).to(device),
            encoder_hidden_states=torch.randn(1, length, _CONFIG["llm_features_dim"], generator=generator).to(device),
            position_ids=position_ids.to(device),
            segment_ids=segment_ids.to(device),
            indicator=indicator.to(device),
            return_dict=False,
        )

        runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA,
            runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
            fp8_comms=None,
        )
        with patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime), torch.no_grad():
            expected = reference(**inputs)[0]
            actual = wrapper(**inputs)[0]

        assert actual.shape == expected.shape
        # Only text and image rows are model output; xDiT zero-fills the leading pad.
        torch.testing.assert_close(actual[:, text_start:], expected[:, text_start:], rtol=1e-5, atol=1e-5)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


# 4 text + 9 image tokens is odd and needs a pad token; 5 + 9 splits evenly.
@pytest.mark.parametrize("num_text_tokens", [4, 5], ids=["padded", "divisible"])
def test_ulysses_forward_matches_unsplit_diffusers(num_text_tokens, accelerator_ranks):
    pytest.importorskip("diffusers.models.transformers.transformer_ideogram4")
    accelerator_ranks(
        _ideogram4_worker,
        world_size=2,
        init_filename=f"ideogram4-u2-text{num_text_tokens}",
        args=(num_text_tokens,),
    )
