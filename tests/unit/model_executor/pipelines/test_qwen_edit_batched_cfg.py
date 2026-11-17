"""Qwen-Image-Edit's batched true CFG: the transformer stand-in batches and splits both branches."""

import pytest
import torch

from xfuser.model_executor.pipelines import pipeline_qwen_image_edit as edit


class _Transformer:
    """Records calls and returns rows that name the prompt each row saw."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["encoder_hidden_states"]
        # One output row per batch row, tagged by the row's prompt sum.
        return (prompt.sum(dim=(1, 2)).view(-1, 1, 1).expand(-1, 4, 2).clone(),)


def _step_kwargs(prompt, mask=None):
    return dict(
        hidden_states=torch.ones(1, 6, 2),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=prompt,
        encoder_hidden_states_mask=mask,
        img_shapes=[[(1, 2, 3)]],
        attention_kwargs={"keep": 1},
        return_dict=False,
    )


@pytest.fixture
def sp_world(monkeypatch):
    monkeypatch.setattr(edit, "get_sequence_parallel_world_size", lambda: 2)


def test_the_two_branches_run_as_one_padded_batch(sp_world):
    positive, negative = torch.ones(1, 5, 3), torch.full((1, 2, 3), 2.0)
    transformer = _Transformer()
    standin = edit._BatchedTrueCfgTransformer(transformer, [(positive, None), (negative, None)])

    conditional = standin(**_step_kwargs(positive))
    unconditional = standin(**_step_kwargs(negative))

    assert len(transformer.calls) == 1
    batch = transformer.calls[0]
    assert batch["encoder_hidden_states"].shape == (2, 5, 3)
    assert batch["encoder_hidden_states"][1, 2:].abs().sum() == 0
    assert batch["encoder_hidden_states_mask"].tolist() == [[True] * 5, [True] * 2 + [False] * 3]
    assert batch["hidden_states"].shape == (2, 6, 2)
    assert batch["timestep"].tolist() == [0.5, 0.5]
    assert batch["img_shapes"] == [[(1, 2, 3)]] * 2
    assert batch["attention_kwargs"]["keep"] == 1
    assert batch["attention_kwargs"]["cu_seqlens_k"].tolist() == [0, 11, 19]
    assert conditional[0].shape == (1, 4, 2) and float(conditional[0][0, 0, 0]) == positive.sum()
    assert float(unconditional[0][0, 0, 0]) == negative.sum()


def test_the_stored_branch_serves_one_step_only(sp_world):
    positive, negative = torch.ones(1, 5, 3), torch.full((1, 2, 3), 2.0)
    transformer = _Transformer()
    standin = edit._BatchedTrueCfgTransformer(transformer, [(positive, None), (negative, None)])

    for _ in range(3):
        standin(**_step_kwargs(positive))
        standin(**_step_kwargs(negative))
    assert len(transformer.calls) == 3
    # Without a fresh conditional call first, the unconditional branch runs on its own.
    standin(**_step_kwargs(negative))
    assert len(transformer.calls) == 4
    assert transformer.calls[-1]["encoder_hidden_states"] is negative


def test_prompts_with_their_own_padding_keep_two_calls(sp_world):
    positive, negative = torch.ones(1, 5, 3), torch.full((1, 5, 3), 2.0)
    padded_mask = torch.tensor([[True, True, True, False, False]])
    transformer = _Transformer()
    standin = edit._BatchedTrueCfgTransformer(transformer, [(positive, None), (negative, padded_mask)])

    standin(**_step_kwargs(positive))
    standin(**_step_kwargs(negative, padded_mask))
    assert [call["encoder_hidden_states"] for call in transformer.calls] == [positive, negative]


def test_without_packing_the_branches_keep_two_calls(monkeypatch):
    # Degree 8 with 6 image tokens pads several shards, so the prompt is replicated and gets no packing.
    monkeypatch.setattr(edit, "get_sequence_parallel_world_size", lambda: 8)
    positive, negative = torch.ones(1, 5, 3), torch.full((1, 2, 3), 2.0)
    transformer = _Transformer()
    standin = edit._BatchedTrueCfgTransformer(transformer, [(positive, None), (negative, None)])

    standin(**_step_kwargs(positive))
    standin(**_step_kwargs(negative))
    assert len(transformer.calls) == 2
