import json
import logging
from types import SimpleNamespace

import pytest
import torch

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models import base_model
from xfuser import logger as xfuser_logger


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    """The runner logs and saves from the last rank, read from the launcher's environment."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


class _StubModel(base_model.xFuserModel):
    settings = base_model.ModelSettings(model_name="test/model")

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def test_outputs_land_in_an_output_directory_that_did_not_exist(tmp_path):
    output_directory = tmp_path / "benchmark_results" / "run"
    model = _StubModel(xFuserArgs(model="test/model", output_directory=str(output_directory)))

    model.save_timings([1.5])

    assert json.loads((output_directory / "timings.json").read_text()) == [1.5]


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_one_prompt_is_shared_by_every_data_parallel_group(monkeypatch, dp_rank):
    monkeypatch.setattr(base_model, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_model, "get_data_parallel_rank", lambda: dp_rank)
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=2)

    # --prompt "a cat" parses to a one-element list.
    split_args = model._split_prompts_for_dp({"prompt": ["a cat"], "negative_prompt": None})

    assert split_args["prompt"] == ["a cat"]


def test_too_few_prompts_for_the_data_parallel_groups_are_rejected(monkeypatch):
    monkeypatch.setattr(base_model, "get_data_parallel_world_size", lambda: 3)
    monkeypatch.setattr(base_model, "get_data_parallel_rank", lambda: 0)
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=3)

    with pytest.raises(ValueError, match="less than data_parallel_world_size"):
        model._split_prompts_for_dp({"prompt": ["a cat", "a dog"], "negative_prompt": None})


@pytest.mark.parametrize(("parallelizes_vae", "prompt_count"), [(True, 3), (False, 3), (True, 4)])
def test_a_wrapper_parallel_vae_needs_the_same_prompt_count_in_every_data_parallel_group(
    monkeypatch, parallelizes_vae, prompt_count
):
    monkeypatch.setattr(base_model, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_model, "get_data_parallel_rank", lambda: 0)
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=2)
    model._legacy_pipe = SimpleNamespace(parallelizes_vae=lambda: parallelizes_vae)
    input_args = {"prompt": [str(i) for i in range(prompt_count)], "negative_prompt": None}

    if parallelizes_vae and prompt_count % 2:
        with pytest.raises(ValueError, match="3 prompts must divide evenly across --data_parallel_degree 2"):
            model._split_prompts_for_dp(input_args)
    else:
        assert model._split_prompts_for_dp(input_args)["prompt"] == input_args["prompt"][0::2]


def _model_class(**attributes):
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel

    def _load_model(self):
        raise AssertionError("refused before loading")

    def _run_pipe(self, input_args):
        raise AssertionError("refused before running")

    return type("ConfigOnlyModel", (xFuserModel,), {"_load_model": _load_model, "_run_pipe": _run_pipe, **attributes})


def test_video_model_without_fps_is_refused_at_config_time():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import ModelSettings

    model_class = _model_class(settings=ModelSettings(model_name="video-without-fps", model_output_type="video"))

    with pytest.raises(ValueError, match="produces video output but fps is not set"):
        model_class(xFuserArgs(model="video-without-fps"))


def test_ulysses_degree_that_does_not_divide_heads_is_refused_at_config_time():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import ModelCapabilities, ModelSettings

    model_class = _model_class(
        attention_heads=6,
        capabilities=ModelCapabilities(ulysses_degree=True, ring_degree=True),
        settings=ModelSettings(model_name="six-heads"),
    )

    model_class(xFuserArgs(model="six-heads", ulysses_degree=3))
    with pytest.raises(ValueError, match=r"six-heads has 6 attention heads.*\(1, 2, 3, 6\); got 4\..*--ring_degree"):
        model_class(xFuserArgs(model="six-heads", ulysses_degree=4))


@pytest.mark.parametrize(
    ("heads", "widths", "padding", "degree", "error"),
    [
        (6, frozenset(), False, 4, "--ulysses_degree must divide"),
        (6, frozenset({128}), False, 4, "--ulysses_degree must divide"),
        (6, frozenset({64}), False, 3, "runs head dimension 64"),
        (6, frozenset({64}), True, 4, "runs head dimension 64"),
        (None, frozenset({64}), False, 1, "runs head dimension 64"),
        (6, frozenset({128}), False, 3, None),
        (6, frozenset({64, 128}), False, 3, None),
        (6, frozenset({128}), True, 4, None),
    ],
)
def test_head_width_and_ulysses_constraints_are_both_validated(heads, widths, padding, degree, error):
    # AITER_BF16 declares 128-wide heads in its spec; no kernel is loaded here.
    model_class = _model_class(
        settings=base_model.ModelSettings(model_name="head-layout"),
        attention_heads=heads,
        attention_head_dims=widths,
        supports_ulysses_head_padding=padding,
    )
    config = xFuserArgs(model="head-layout", attention_backend="AITER_BF16", ulysses_degree=degree)
    if error:
        with pytest.raises(ValueError, match=error):
            model_class(config)
    else:
        assert model_class(config).pipe is None


class _HeadDimConfig:
    use_hybrid_attn_schedule = False
    cross_attention_backend = None
    ulysses_degree = None

    def __init__(self, backend):
        self.attention_backend = backend


class _HeadDimModel:
    """The validator reads head counts after the width check, even when it refused nothing."""

    def __init__(self, dims):
        self.attention_head_dims = dims
        self.attention_heads = None
        self.attention_kv_heads = None
        self.supports_ulysses_head_padding = False
        self.capabilities = SimpleNamespace(ulysses_degree=False, ring_degree=False)
        self.settings = SimpleNamespace(model_name="model")


def test_a_model_is_refused_a_backend_that_serves_none_of_its_head_dims():
    """Ideogram 4 runs head dimension 256, which MHA v4 does not serve.
    A backend that declares no HEAD_DIM makes no claim and stays selectable."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.attention.backends.aiter_mha_v4.spec import DENSE_BACKENDS
    from xfuser.model_executor.models.runner_models.ideogram4 import xFuserIdeogram4Model
    from xfuser.model_executor.models.runner_models.ltx import _xFuserLTX25VideoModelBase

    ideogram4 = _HeadDimModel(xFuserIdeogram4Model.attention_head_dims)
    for backend in DENSE_BACKENDS:
        with pytest.raises(ValueError, match="head dimension"):
            base_model._validate_attention_head_dims(ideogram4, _HeadDimConfig(backend.name))

    assert registry.get(AttentionBackendType.AITER).accepts.head_dims() is None
    base_model._validate_attention_head_dims(ideogram4, _HeadDimConfig("AITER"))
    base_model._validate_attention_head_dims(ideogram4, _HeadDimConfig("SDPA"))

    ltx25 = _HeadDimModel(_xFuserLTX25VideoModelBase.attention_head_dims)
    assert 64 in ltx25.attention_head_dims
    base_model._validate_attention_head_dims(ltx25, _HeadDimConfig("AITER_BF16"))

    base_model._validate_attention_head_dims(_HeadDimModel(frozenset()), _HeadDimConfig("AITER_BF16"))


def test_the_head_dim_refusal_names_fallback_or_a_hard_stop():
    model = _HeadDimModel(frozenset({256}))

    with pytest.raises(ValueError, match="fall through to AITER"):
        base_model._validate_attention_head_dims(model, _HeadDimConfig("AITER_BF16"))

    with pytest.raises(ValueError, match="would be refused"):
        base_model._validate_attention_head_dims(model, _HeadDimConfig("AITER_FP8_SPARGE"))


def _gather_dp_outputs(monkeypatch, *, dp_degree, legacy_pipe, rank=0, sp_rank=0):
    """Run ``_gather_dp_outputs`` on a non-saving rank and return what it sent, or "not gathered"."""
    sent = []
    monkeypatch.setattr(base_model, "get_world_group", lambda: SimpleNamespace(world_size=2, rank=rank))
    monkeypatch.setattr(base_model, "get_sequence_parallel_rank", lambda: sp_rank)
    monkeypatch.setattr(base_model, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(base_model.torch.distributed, "gather_object", lambda obj, *args, **kwargs: sent.append(obj))
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=dp_degree)
    model._legacy_pipe = legacy_pipe

    model._gather_dp_outputs("output")
    return sent or "not gathered"


def _legacy_pipe(*, parallelizes_vae, is_dp_last_group):
    return SimpleNamespace(parallelizes_vae=lambda: parallelizes_vae, is_dp_last_group=lambda: is_dp_last_group)


def test_legacy_wrapper_without_dp_or_parallel_vae_keeps_its_output(monkeypatch):
    legacy_pipe = _legacy_pipe(parallelizes_vae=False, is_dp_last_group=True)

    assert _gather_dp_outputs(monkeypatch, dp_degree=1, legacy_pipe=legacy_pipe) == "not gathered"


@pytest.mark.parametrize(
    ("dp_degree", "parallelizes_vae"),
    [
        (2, False),
        (1, True),  # the wrapper's images are on rank 0, the saving rank is the last
    ],
)
def test_legacy_wrapper_sends_from_the_ranks_its_own_decode_selects(monkeypatch, dp_degree, parallelizes_vae):
    for is_dp_last_group, expected in ((True, ["output"]), (False, [None])):
        legacy_pipe = _legacy_pipe(parallelizes_vae=parallelizes_vae, is_dp_last_group=is_dp_last_group)

        # The SP rank would pick the opposite sender: it must not matter for a legacy wrapper.
        sent = _gather_dp_outputs(
            monkeypatch, dp_degree=dp_degree, legacy_pipe=legacy_pipe, sp_rank=int(is_dp_last_group)
        )

        assert sent == expected


@pytest.mark.parametrize(("sp_rank", "expected"), [(0, ["output"]), (1, [None])])
def test_other_runners_send_from_the_first_sequence_parallel_rank(monkeypatch, sp_rank, expected):
    assert _gather_dp_outputs(monkeypatch, dp_degree=2, legacy_pipe=None, sp_rank=sp_rank) == expected


def _pipeline(**modules):
    """Stands in for any pipeline: the eval pass walks attributes, not `components`."""
    return SimpleNamespace(scheduler=object(), **modules)


@pytest.fixture
def runner_warnings(monkeypatch):
    """Warnings the runner logged during this test, each exactly once.

    caplog.handler is not used: pytest keeps that handler on the root logger as well, so any
    unrelated warning reaching root during the test (or a propagated copy) would be counted.
    """
    records = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    # warn_once drops repeats across the whole session.
    monkeypatch.setattr(xfuser_logger, "_logged_once", set())
    base_model.logger.addHandler(handler)
    yield lambda: [record.getMessage() for record in records]
    base_model.logger.removeHandler(handler)


def test_eval_pass_reaches_modules_beside_and_inside_pipelines(runner_warnings):
    shared_vae = torch.nn.Linear(2, 2).eval()
    swapped_block = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)).eval()
    swapped_block[1].train()  # eval-mode parent, training-mode child
    model = object.__new__(_StubModel)
    model.pipe = _pipeline(vae=shared_vae, transformer=swapped_block)
    model.decode_pipe = _pipeline(vae=shared_vae, decoder=torch.nn.Dropout(0.5))
    model.upsampler = torch.nn.BatchNorm1d(2)

    model._set_eval_mode()

    for module in (shared_vae, swapped_block, model.decode_pipe.decoder, model.upsampler):
        assert not any(submodule.training for submodule in module.modules())
        warnings = runner_warnings()
        assert len(warnings) == 1, warnings
        (warning,) = warnings

    for name in ("pipe.transformer", "decode_pipe.decoder", "upsampler"):
        assert name in warning
    assert "vae" not in warning


def test_eval_pass_is_silent_when_everything_is_already_in_eval_mode(runner_warnings):
    model = object.__new__(_StubModel)
    model.pipe = _pipeline(transformer=torch.nn.Dropout(0.5).eval())

    model._set_eval_mode()

    assert runner_warnings() == []
