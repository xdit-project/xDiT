"""The single walk that replaced the format/remainder pass pair.

Drives `setup_gemm_quantization` with recording adapters, so the assertions are
about which converter saw which module, and on which side of the device move.
"""

from types import SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.loading import placement
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
    QuantizationLedger,
)
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)
from xfuser.model_executor.quant.targets import GemmTargets, Select

BLOCKS = ("transformer.transformer_blocks", "transformer.single_transformer_blocks")
TEXT_ENCODER = "text_encoder.model.language_model.layers"

TARGETS = GemmTargets(
    transformer=Select(modules=BLOCKS),
    text_encoder=Select(modules=(TEXT_ENCODER,)),
    keep_high=Select(modules=(TEXT_ENCODER,)),
)


class _Recorder:
    """An adapter that records what it was asked to convert."""

    def __init__(self, name, *, before_device_move=False):
        self.name = name
        self.converts_before_device_move = before_device_move
        self.seen = []

    def convert_module(self, module, **kwargs):
        self.seen.append((module.path, kwargs))


def _pipe(paths):
    root = SimpleNamespace()
    for path in paths:
        node, walked = root, []
        for part in path.split("."):
            walked.append(part)
            if not hasattr(node, part):
                setattr(node, part, SimpleNamespace(path=".".join(walked)))
            node = getattr(node, part)
    return root


def _loader(raw, *, text_encoder=True, fp8_before_move=True, targets=TARGETS):
    spec = GemmQuantizationSpec.parse(raw)
    settings = SimpleNamespace(
        gemm_targets=targets,
        fp8_gemm_include_suffixes=None,
        fp8_precision_overrides=None,
        fp8_precision_override_suffixes=None,
    )
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        use_fp8_text_encoder=text_encoder,
        use_hybrid_gemm_schedule=False,
    )
    model = SimpleNamespace(
        settings=settings, config=config, pipe=_pipe(BLOCKS + (TEXT_ENCODER,))
    )
    primary = _Recorder("format")
    blockwise = _Recorder("blockwise_fp8", before_device_move=fp8_before_move)
    backends = SimpleNamespace(
        format=primary,
        fp8=None,
        fp6=_Recorder("fp6"),
        blockwise_fp8=blockwise,
        format_targets_for=lambda name: (),
    )
    backends.adapter_for = lambda fmt: (
        backends.fp6
        if fmt == "fp6"
        else (backends.fp8 or backends.blockwise_fp8) if fmt == "fp8"
        else backends.format
    )
    return SimpleNamespace(
        model=model,
        backends=backends,
        quantization_plan=QuantizationPlan(model),
        quantization_ledger=QuantizationLedger(),
    )


@pytest.fixture(autouse=True)
def _no_real_descriptor(monkeypatch):
    monkeypatch.setattr(
        placement,
        "prepare_native_transformer_format_load",
        lambda *a, **k: SimpleNamespace(
            descriptor=SimpleNamespace(log_message=lambda: "")
        ),
    )
    monkeypatch.setattr(placement, "_is_cuda", lambda: False)
    # log() asks is_last_process(), which needs a distributed env
    monkeypatch.setattr(placement, "log", lambda *a, **k: None)


def _run(loader, **kwargs):
    placement.setup_gemm_quantization(
        loader, local_rank=0, offload_requested=False, **kwargs
    )


def test_each_format_goes_to_its_own_converter():
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    assert [p for p, _ in loader.backends.blockwise_fp8.seen] == [TEXT_ENCODER]
    assert sorted(p for p, _ in loader.backends.format.seen) == sorted(BLOCKS)


def test_the_adapter_decides_which_side_of_the_device_move_it_runs_on():
    """The ordering is data now, not an if/elif around the `.to(cuda)`."""
    loader = _loader("low=fp4,high=fp8", fp8_before_move=True)
    _run(loader, before_device_move=True)
    assert [p for p, _ in loader.backends.blockwise_fp8.seen] == [TEXT_ENCODER]
    assert loader.backends.format.seen == []

    loader = _loader("low=fp4,high=fp8", fp8_before_move=False)
    _run(loader, before_device_move=True)
    assert loader.backends.blockwise_fp8.seen == []
    assert loader.backends.format.seen == []
    _run(loader, before_device_move=False)
    assert [p for p, _ in loader.backends.blockwise_fp8.seen] == [TEXT_ENCODER]
    assert sorted(p for p, _ in loader.backends.format.seen) == sorted(BLOCKS)


def test_one_format_sends_everything_to_one_converter():
    """Including the text encoder, and including what keep_high names: one
    format means the model's high/low preference has nothing to act on."""
    loader = _loader("fp4", text_encoder=True)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert sorted(p for p, _ in loader.backends.format.seen) == sorted(
        BLOCKS + (TEXT_ENCODER,)
    )
    assert loader.backends.blockwise_fp8.seen == []


def test_the_text_encoder_is_skipped_unless_the_run_asks():
    loader = _loader("low=fp4,high=fp8", text_encoder=False)
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert loader.backends.blockwise_fp8.seen == []
    assert sorted(p for p, _ in loader.backends.format.seen) == sorted(BLOCKS)


def test_only_the_primary_converter_takes_the_hybrid_arguments():
    """The FP8 converter builds no per-block wrappers, so it takes none."""
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)

    _, fp8_kwargs = loader.backends.blockwise_fp8.seen[0]
    _, primary_kwargs = loader.backends.format.seen[0]
    assert "hybrid" not in fp8_kwargs and "fp8_layers" not in fp8_kwargs
    assert primary_kwargs["hybrid"] is False
    assert primary_kwargs["fp8_layers"] is None


def test_nothing_is_quantized_twice():
    """A module owned by one format is excluded from the other's walk."""
    loader = _loader("low=fp4,high=fp8")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    seen = [p for p, _ in loader.backends.format.seen] + [
        p for p, _ in loader.backends.blockwise_fp8.seen
    ]
    assert len(seen) == len(set(seen))


def test_an_unquantized_run_walks_nothing():
    loader = _loader("none")
    _run(loader, before_device_move=True)
    _run(loader, before_device_move=False)
    assert loader.backends.format.seen == []
    assert loader.backends.blockwise_fp8.seen == []
