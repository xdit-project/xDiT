"""How a legacy xFuserPipelineBaseWrapper decides on prompts and parallel VAE decoding."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.pipelines import base_pipeline
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper


class _Pipeline(xFuserPipelineBaseWrapper):
    @xFuserPipelineBaseWrapper.enable_data_parallel
    def __call__(self, prompt=None):
        return prompt


def _pipeline():
    return object.__new__(_Pipeline)


def _patch_state(monkeypatch, *, use_parallel_vae=False, sp_degree=1, dp_degree=1, rank=0):
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(use_parallel_vae=use_parallel_vae),
        parallel_config=SimpleNamespace(vae_parallel_size=0, dp_degree=dp_degree),
    )
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_pipeline, "get_sequence_parallel_world_size", lambda: sp_degree)
    for name in (
        "get_pipeline_parallel_world_size",
        "get_classifier_free_guidance_world_size",
        "get_tensor_model_parallel_world_size",
    ):
        monkeypatch.setattr(base_pipeline, name, lambda: 1)
    monkeypatch.setattr(base_pipeline, "get_fast_attn_enable", lambda: False)
    monkeypatch.setattr(base_pipeline, "get_world_group", lambda: SimpleNamespace(rank=rank))
    monkeypatch.setattr(base_pipeline, "get_dit_world_size", lambda: dp_degree * sp_degree)
    monkeypatch.setattr(base_pipeline, "get_data_parallel_world_size", lambda: dp_degree)


@pytest.mark.parametrize(
    ("use_parallel_vae", "sp_degree", "expected"),
    [
        (True, 2, True),
        (True, 1, False),  # a single-rank wrapper takes the naive forward and never shards its decoder
        (False, 2, False),
    ],
)
def test_wrapper_parallelizes_the_vae_only_off_the_naive_forward(monkeypatch, use_parallel_vae, sp_degree, expected):
    _patch_state(monkeypatch, use_parallel_vae=use_parallel_vae, sp_degree=sp_degree)

    assert _pipeline().parallelizes_vae() is expected


@pytest.mark.parametrize(
    ("split_by_caller", "rank", "expected"), [(False, 1, ["c", "d"]), (True, 1, ["a", "b", "c", "d"])]
)
def test_wrapper_splits_prompts_across_dp_groups_unless_the_caller_did(monkeypatch, split_by_caller, rank, expected):
    _patch_state(monkeypatch, dp_degree=2, rank=rank)
    pipeline = _pipeline()
    if split_by_caller:
        pipeline.prompts_split_by_caller = True

    assert pipeline(prompt=["a", "b", "c", "d"]) == expected
