"""What each FLUX.3 attention stage hands the attention layer.

The text stage attends the prompt alone, too few KV blocks for a backend that selects blocks, so it
runs on the cross-attention backend. The joint stage marks the prompt's keys and rows exact for
such a backend. Each forward advances the per-step schedules once.
"""

from types import SimpleNamespace

import pytest
import torch

transformer_flux3 = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_flux3")

TINY = {
    "in_channels": 8,
    "cond_in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "timestep_channels": 32,
    "axes_dims_rope": (4, 4, 4, 4),
    "flow_map": False,
}
TEXT_LEN, IMAGE_GRID = 5, (2, 3)


def _ids(height, width):
    grid = torch.meshgrid(torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([0]), indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 4).float()


def _inputs():
    generator = torch.Generator().manual_seed(0)
    image_len = IMAGE_GRID[0] * IMAGE_GRID[1]
    return {
        "hidden_states": torch.randn(1, image_len, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, TEXT_LEN, 12, generator=generator),
        "timestep": torch.tensor([0.7]),
        "img_ids": _ids(*IMAGE_GRID),
        "txt_ids": torch.stack([torch.zeros(TEXT_LEN)] * 3 + [torch.arange(TEXT_LEN).float()], dim=-1),
    }


def test_each_stage_gets_its_backend_and_the_joint_stage_marks_the_prompt(monkeypatch):
    from xfuser.core.attention.spec import EXACT_KV_TOKENS_KEY, EXACT_Q_TOKENS_KEY, AttentionBackendType

    steps = []
    runtime = SimpleNamespace(
        get_cross_attention_backend=lambda: AttentionBackendType.AITER,
        increment_step_counter=lambda: steps.append(True),
    )
    calls = []

    def usp(query, key, value, *, attn_layer=None, backend=None, attention_kwargs=None, **kwargs):
        calls.append((attn_layer, backend, attention_kwargs))
        return query

    monkeypatch.setattr(transformer_flux3, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(transformer_flux3, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_flux3, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(transformer_flux3, "USP", usp)
    # One rank and no pad: gathering is the identity, and there is no process group to ask.
    monkeypatch.setattr(transformer_flux3, "gather_and_unpad", lambda x, pad, dim: x)
    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY).eval()

    with torch.no_grad():
        wrapper(**_inputs())
        wrapper(**_inputs())

    assert len(steps) == 2, "one step per forward"
    text = {id(block.attn) for block in wrapper.txt_mode_blocks}
    joint = {id(block.attn) for block in wrapper.single_transformer_blocks}
    first_forward = calls[: len(calls) // 2]
    for attn, backend, kwargs in first_forward:
        if id(attn) in text:
            assert backend is AttentionBackendType.AITER, "the text stage runs on the cross-attention backend"
        else:
            assert backend is None, "the other stages run on the selected backend"
        if id(attn) in joint:
            exact = kwargs[EXACT_KV_TOKENS_KEY]
            expected = torch.zeros(TEXT_LEN + IMAGE_GRID[0] * IMAGE_GRID[1], dtype=torch.bool)
            expected[:TEXT_LEN] = True
            assert torch.equal(exact, expected), "the prompt's keys, and only those, are exact"
            assert torch.equal(kwargs[EXACT_Q_TOKENS_KEY], expected), "and so are the prompt's own rows"
        else:
            assert kwargs is None or (
                kwargs.get(EXACT_KV_TOKENS_KEY) is None and kwargs.get(EXACT_Q_TOKENS_KEY) is None
            )
    assert {id(attn) for attn, _, _ in first_forward} >= text | joint
