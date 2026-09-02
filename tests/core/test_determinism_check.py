"""Determinism-check: exact comparison, snapshots, thresholds, logging, and CLI."""

import copy
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs
from xfuser.core.utils.determinism_check_results import (
    _count_determinism_failures_in_log,
    _failed_output_statistics,
)
from xfuser.core.utils.outputs_equal import _payloads_equal, outputs_equal
from xfuser.core.utils.runner_utils import log_error
from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import (
    DiffusionOutput,
    xFuserModel,
)


class _Runner(xFuserModel):
    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


class _AudioOutput(DiffusionOutput):
    def __init__(
        self,
        images=None,
        videos=None,
        pipe_args=None,
        audio=None,
        audio_sample_rate=None,
    ):
        super().__init__(images=images, videos=videos, pipe_args=pipe_args or [])
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate


class _SlotBlob:
    __slots__ = ()


class _CopyableBox:
    def __init__(self, value):
        self.value = value


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("cannot deepcopy uncopyable field")


class _CudaEvent:
    def __init__(self, enable_timing=True):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1000.0


def _eq(a, b, path=""):
    return _payloads_equal(a, b, path)


def _rgb(color=(1, 2, 3), size=(2, 2)):
    return Image.new("RGB", size, color=color)


def _paletted(fill, rgb_triple):
    palette = [0] * 768
    palette[0:3] = list(rgb_triple)
    image = Image.new("P", (2, 2), color=fill)
    image.putpalette(palette)
    return image


def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _CudaEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def _make_runner(
    monkeypatch,
    *,
    num_iterations=1,
    determinism_check=0,
    warmup_calls=0,
    batch_size=None,
    data_parallel_degree=1,
    rank=0,
    determinism_check_report_ranks=None,
    outputs=None,
    batched_outputs=None,
    stub_warmup=False,
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    _stub_cuda(monkeypatch)
    monkeypatch.setattr(
        base_model,
        "get_world_group",
        lambda: SimpleNamespace(rank=rank),
    )

    model = object.__new__(_Runner)
    if determinism_check_report_ranks is None:
        determinism_check_report_ranks = frozenset({rank})
    model.config = SimpleNamespace(
        num_iterations=num_iterations,
        determinism_check=determinism_check,
        determinism_check_report_ranks=determinism_check_report_ranks,
        warmup_calls=warmup_calls,
        batch_size=batch_size,
        data_parallel_degree=data_parallel_degree,
    )
    model.settings = SimpleNamespace(model_output_type="image")
    model._validate_args = lambda input_args: None
    model._split_prompts_for_dp = lambda input_args: input_args

    if outputs is not None:
        timed = iter(outputs)

        def _run_timed_pipe(input_args):
            return next(timed), 0.1

        model._run_timed_pipe = _run_timed_pipe

    if batched_outputs is not None:
        batched = iter(batched_outputs)

        def _run_pipe_batched(input_args):
            return next(batched), [0.1]

        model._run_pipe_batched = _run_pipe_batched

    if stub_warmup:
        model._run_warmup_calls = lambda input_args: None

    return model


def _record_determinism_checks(model):
    checks = []
    original_check = model._determinism_check

    def _check(*args):
        result = original_check(*args)
        checks.append((args, result))
        return result

    model._determinism_check = _check
    return checks


# --- NumPy -----------------------------------------------------------------


def test_numpy_identical_arrays_match():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert _eq(a, b)


def test_numpy_changed_value_diverges():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.5], dtype=np.float32)
    assert not _eq(a, b)


def test_numpy_equal_values_different_dtypes_diverge():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float64)
    assert not _eq(a, b)


def test_numpy_reshaped_equal_values_diverge():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([[1.0, 2.0]], dtype=np.float32)
    assert not _eq(a, b)


def test_numpy_nans_match_including_distinct_payloads():
    a = np.array([np.float32("nan"), 1.0], dtype=np.float32)
    b = np.frombuffer(np.array([np.uint32(0x7FC00001)], dtype=np.uint32).tobytes(), dtype=np.float32)
    b = np.concatenate([b, np.array([1.0], dtype=np.float32)])
    assert _eq(a, np.array([np.float32("nan"), 1.0], dtype=np.float32))
    assert _eq(a[:1], b[:1])


def test_numpy_positive_and_negative_zero_match():
    a = np.array([0.0], dtype=np.float64)
    b = np.array([-0.0], dtype=np.float64)
    assert _eq(a, b)


def test_numpy_noncontiguous_view_matches_logical_values():
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    view = base.T
    assert not view.flags.c_contiguous
    assert _eq(view, np.ascontiguousarray(view))


def test_numpy_infinities_match_and_finite_mismatch_diverges():
    assert _eq(np.array([np.inf], dtype=np.float32), np.array([np.inf], dtype=np.float32))
    assert _eq(np.array([-np.inf], dtype=np.float32), np.array([-np.inf], dtype=np.float32))
    assert not _eq(np.array([np.inf], dtype=np.float32), np.array([-np.inf], dtype=np.float32))
    assert not _eq(np.array([1.0], dtype=np.float32), np.array([np.nextafter(1.0, 2.0, dtype=np.float32)]))


def test_numpy_object_array_recurses_and_mismatch_diverges():
    left = np.empty(2, dtype=object)
    right = np.empty(2, dtype=object)
    left[0] = torch.tensor([1.0])
    left[1] = np.array([2.0], dtype=np.float32)
    right[0] = torch.tensor([1.0])
    right[1] = np.array([2.0], dtype=np.float32)
    assert _eq(left, right, "videos")

    right[1] = np.array([3.0], dtype=np.float32)
    assert not _eq(left, right, "videos")


def test_numpy_object_array_unsupported_element_raises_indexed_typeerror():
    left = np.empty(2, dtype=object)
    right = np.empty(2, dtype=object)
    left[0] = 1
    right[0] = 1
    left[1] = set()
    right[1] = set()
    with pytest.raises(TypeError, match=r"Unsupported payload type set at videos\[1\]"):
        _eq(left, right, "videos")


# --- PyTorch ---------------------------------------------------------------


def test_torch_identical_tensors_match():
    assert _eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 2], [3, 4]]))


def test_torch_changed_data_shape_or_dtype_diverges():
    base = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert not _eq(base, torch.tensor([1.0, 2.5], dtype=torch.float32))
    assert not _eq(base, torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert not _eq(base, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_torch_noncontiguous_view_matches_logical_values():
    base = torch.arange(6).reshape(2, 3)
    view = base.t()
    assert not view.is_contiguous()
    assert _eq(view, view.contiguous())


def test_torch_nan_equals_nan_and_signed_zeros_match():
    assert _eq(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert _eq(
        torch.tensor([float("nan")], dtype=torch.float16),
        torch.tensor([float("nan")], dtype=torch.float16),
    )
    assert _eq(
        torch.tensor([float("nan")], dtype=torch.bfloat16),
        torch.tensor([float("nan")], dtype=torch.bfloat16),
    )
    assert not _eq(torch.tensor([1.0]), torch.tensor([float("nan")]))
    assert _eq(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert _eq(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert _eq(torch.tensor([float("-inf")]), torch.tensor([float("-inf")]))


def test_torch_requires_grad_does_not_cause_divergence():
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([1.0, 2.0], requires_grad=False)
    assert _eq(a, b)


# --- PIL -------------------------------------------------------------------


def test_pil_identical_images_match():
    assert _eq(_rgb(), _rgb())


def test_pil_changed_size_mode_palette_or_pixels_diverge():
    assert not _eq(_rgb(size=(2, 2)), _rgb(size=(3, 2)))
    assert not _eq(_rgb(), Image.new("L", (2, 2), color=1))
    assert not _eq(_rgb(color=(1, 2, 3)), _rgb(color=(9, 2, 3)))
    assert not _eq(_paletted(0, (255, 0, 0)), _paletted(0, (0, 255, 0)))


def test_pil_identical_palette_images_match():
    assert _eq(_paletted(0, (255, 0, 0)), _paletted(0, (255, 0, 0)))


def test_pil_different_concrete_classes_diverge():
    FakePng = type("FakePng", (Image.Image,), {})
    a = _rgb()
    b = _rgb()
    b.__class__ = FakePng
    assert type(a) is not type(b)
    assert a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()
    assert not _eq(a, b)


# --- Structure --------------------------------------------------------------


def test_list_versus_tuple_of_same_content_diverges():
    assert not _eq([1, 2], (1, 2))


def test_sequence_length_and_order_matter():
    assert _eq([1, 2], [1, 2])
    assert not _eq([1, 2], [1, 2, 3])
    assert not _eq([1, 2], [2, 1])


def test_mapping_key_order_and_values_matter():
    assert _eq({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not _eq({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not _eq({"a": 1, "b": 2}, {"a": 1, "b": 3})


def test_nested_simplenamespace_audiovisual_payload_is_traversed():
    frames = np.array([1.0, 2.0], dtype=np.float32)
    audio = torch.tensor([0.5], dtype=torch.float32)
    left = DiffusionOutput(
        videos=[SimpleNamespace(frames=frames, audio=audio)],
        pipe_args={},
    )
    right = DiffusionOutput(
        videos=[
            SimpleNamespace(
                frames=np.array([1.0, 2.0], dtype=np.float32),
                audio=torch.tensor([0.5], dtype=torch.float32),
            )
        ],
        pipe_args={},
    )
    assert outputs_equal(left, right)

    mismatched = DiffusionOutput(
        videos=[
            SimpleNamespace(
                frames=np.array([1.0, 9.0], dtype=np.float32),
                audio=torch.tensor([0.5], dtype=torch.float32),
            )
        ]
    )
    assert not outputs_equal(left, mismatched)


# --- Scalars ----------------------------------------------------------------


def test_scalar_nan_zero_inf_and_exact_ints_bools():
    assert _eq(float("nan"), float("nan"))
    assert not _eq(1.0, float("nan"))
    assert _eq(0.0, -0.0)
    assert _eq(float("inf"), float("inf"))
    assert _eq(float("-inf"), float("-inf"))
    assert not _eq(1.0, np.nextafter(1.0, 2.0))
    assert _eq(3, 3)
    assert not _eq(3, 4)
    assert _eq(True, True)
    assert not _eq(True, False)
    assert not _eq(True, 1)


def test_numpy_float64_scalar_diverges_from_python_float():
    assert not _eq(np.float64(1.0), 1.0)


# --- Output fields -----------------------------------------------------------


def test_pipe_args_differences_are_ignored():
    img = _rgb()
    left = DiffusionOutput(images=[img], pipe_args={"prompt": "a"})
    right = DiffusionOutput(images=[_rgb()], pipe_args={"prompt": "b", "seed": 7})
    assert outputs_equal(left, right)


def test_subclass_extra_audio_fields_are_compared():
    tensor = torch.tensor([1.0, 2.0])
    left = _AudioOutput(images=[_rgb()], audio=tensor, audio_sample_rate=16000)
    right = _AudioOutput(
        images=[_rgb()],
        audio=torch.tensor([1.0, 2.0]),
        audio_sample_rate=16000,
        pipe_args={"ignored": True},
    )
    assert outputs_equal(left, right)

    assert not outputs_equal(
        left,
        _AudioOutput(images=[_rgb()], audio=torch.tensor([1.0, 9.0]), audio_sample_rate=16000),
    )
    assert not outputs_equal(
        left,
        _AudioOutput(images=[_rgb()], audio=tensor, audio_sample_rate=8000),
    )


def test_different_concrete_output_types_diverge():
    assert not outputs_equal(
        DiffusionOutput(images=[_rgb()]),
        _AudioOutput(images=[_rgb()], audio=None, audio_sample_rate=None),
    )


def test_extra_missing_or_reordered_instance_fields_diverge():
    base = DiffusionOutput(images=[_rgb()])
    extra = DiffusionOutput(images=[_rgb()])
    extra.audio = torch.tensor([1.0])
    assert not outputs_equal(base, extra)

    ordered = DiffusionOutput(images=[_rgb()])
    ordered.audio = 1
    ordered.rate = 2
    reordered = DiffusionOutput(images=[_rgb()])
    reordered.rate = 2
    reordered.audio = 1
    assert not outputs_equal(ordered, reordered)


# --- Unsupported values ------------------------------------------------------


def test_unsupported_custom_class_in_images_raises_path_typeerror():
    with pytest.raises(TypeError, match=r"Unsupported payload type _SlotBlob at images\[0\]"):
        outputs_equal(
            DiffusionOutput(images=[_SlotBlob()]),
            DiffusionOutput(images=[_SlotBlob()]),
        )


def test_unsupported_set_raises_and_does_not_use_equality():
    with pytest.raises(TypeError, match=r"Unsupported payload type set at <root>"):
        _eq(set(), set(), "")


def test_memoryview_is_unsupported_not_bytes():
    with pytest.raises(TypeError, match=r"Unsupported payload type memoryview at images"):
        _eq(memoryview(b"ab"), memoryview(b"ab"), "images")
    assert _eq(b"ab", b"ab", "images")
    assert not _eq(b"ab", bytearray(b"ab"), "images")


def test_unsupported_nested_paths_include_index_attr_and_mapping_key():
    with pytest.raises(TypeError, match=r"Unsupported payload type set at videos\[0\]\.audio"):
        _eq([SimpleNamespace(audio=set())], [SimpleNamespace(audio=set())], "videos")
    with pytest.raises(TypeError, match=r"Unsupported payload type set at audio\['key'\]"):
        _eq({"key": set()}, {"key": set()}, "audio")


# --- Snapshot / deepcopy -----------------------------------------------------


def test_deepcopy_snapshot_is_independent_for_supported_payloads():
    image = _rgb()
    frames = np.array([1.0, 2.0], dtype=np.float32)
    tensor = torch.tensor([3.0, 4.0])
    nested = SimpleNamespace(frames=frames, audio=tensor)
    original = _AudioOutput(
        images=[image],
        videos=[nested],
        pipe_args={"prompt": "keep"},
        audio=torch.tensor([5.0]),
        audio_sample_rate=16000,
    )
    snapshot = copy.deepcopy(original)

    image.putpixel((0, 0), (9, 9, 9))
    frames[0] = 99.0
    tensor.add_(1.0)
    original.audio.add_(1.0)
    original.audio_sample_rate = 8
    original.pipe_args[0]["prompt"] = "mutated"

    assert snapshot.images[0].tobytes() == _rgb().tobytes()
    assert np.array_equal(snapshot.videos[0].frames, np.array([1.0, 2.0], dtype=np.float32))
    assert torch.equal(snapshot.videos[0].audio, torch.tensor([3.0, 4.0]))
    assert torch.equal(snapshot.audio, torch.tensor([5.0]))
    assert snapshot.audio_sample_rate == 16000
    assert snapshot.pipe_args == [{"prompt": "keep"}]


def test_deepcopy_copies_custom_extra_field_independently():
    original = DiffusionOutput(images=[_rgb()])
    original.box = _CopyableBox(1)
    snapshot = copy.deepcopy(original)
    original.box.value = 99
    assert snapshot.box is not original.box
    assert snapshot.box.value == 1


def test_deepcopy_surfaces_error_from_uncopyable_field():
    original = DiffusionOutput(images=[_rgb()])
    original.bad = _Uncopyable()
    with pytest.raises(RuntimeError, match="cannot deepcopy uncopyable field"):
        copy.deepcopy(original)


# --- Run loop ---------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0, -1])
def test_nonpositive_threshold_skips_copy_error_and_determinism_check(monkeypatch, threshold):
    first = DiffusionOutput(images=[_rgb((1, 0, 0))])
    last = DiffusionOutput(images=[_rgb((9, 0, 0))])
    model = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=threshold,
        warmup_calls=0,
        outputs=[first, last],
    )
    checks = _record_determinism_checks(model)
    logged = []
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: logged.append((args, kwargs)))
    deepcopy_calls = []
    real_deepcopy = copy.deepcopy

    def _spy(obj):
        deepcopy_calls.append(obj)
        return real_deepcopy(obj)

    monkeypatch.setattr(base_model.copy, "deepcopy", _spy)

    output, _timings = model.run({"prompt": "x"})

    assert output is last
    assert checks == []
    assert logged == []
    assert deepcopy_calls == []


def test_one_enabled_iteration_establishes_baseline_without_failure(monkeypatch):
    only = DiffusionOutput(images=[_rgb()])
    model = _make_runner(
        monkeypatch,
        num_iterations=1,
        determinism_check=1,
        outputs=[only],
    )
    checks = _record_determinism_checks(model)
    logged = []
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: logged.append((args, kwargs)))

    output, _timings = model.run({"prompt": "x"})

    assert output is only
    assert len(checks) == 1
    assert checks[0][0] == (0, None, only, 0)
    failure_count, baseline = checks[0][1]
    assert failure_count == 0
    assert baseline is not only
    assert outputs_equal(baseline, only)
    assert logged == []


def test_sequence_a_b_c_a_reports_iterations_against_first_snapshot(monkeypatch):
    a1 = DiffusionOutput(videos=[np.array([1.0], dtype=np.float32)])
    b = DiffusionOutput(videos=[np.array([2.0], dtype=np.float32)])
    c = DiffusionOutput(videos=[np.array([3.0], dtype=np.float32)])
    a2 = DiffusionOutput(videos=[np.array([1.0], dtype=np.float32)])
    model = _make_runner(
        monkeypatch,
        num_iterations=4,
        determinism_check=1,
        rank=3,
        outputs=[a1, b, c, a2],
    )
    checks = _record_determinism_checks(model)
    model._save_determinism_check_failed_outputs = lambda *args: None
    logged = []
    monkeypatch.setattr(base_model, "log_error", lambda *a, **kw: logged.append((a, kw)))
    gathers = []
    original_gather = model._gather_dp_outputs

    def _gather(output):
        gathers.append(output)
        return original_gather(output)

    model._gather_dp_outputs = _gather

    output, _timings = model.run({"prompt": "x"})

    assert output is a2
    assert gathers == [a2]
    assert [call[0][0] for call in logged] == [
        "determinism_check[rank 3]: iteration 2 diverged!",
        "determinism_check[rank 3]: iteration 3 diverged!",
    ]
    assert all(call[1]["log_from_all_processes"] is True for call in logged)

    assert [args[0] for args, _result in checks] == [0, 1, 2, 3]
    assert [result[0] for _args, result in checks] == [0, 1, 2, 2]
    baseline = checks[0][1][1]
    assert baseline is not a1
    assert [args[2] for args, _result in checks] == [a1, b, c, a2]
    assert all(args[1] is baseline for args, _result in checks[1:])
    a1.videos[0][0] = 99.0
    assert np.array_equal(baseline.videos[0], np.array([1.0], dtype=np.float32))


def test_determinism_check_counts_every_failure(monkeypatch):
    outputs = [
        DiffusionOutput(videos=[np.array([value], dtype=np.float32)])
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    model = _make_runner(
        monkeypatch,
        num_iterations=4,
        determinism_check=2,
        rank=7,
        outputs=outputs,
    )
    checks = _record_determinism_checks(model)
    model._save_determinism_check_failed_outputs = lambda *args: None
    logged = []
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: logged.append(args))

    model.run({"prompt": "x"})

    assert len(logged) == 3
    assert [args[3] for args, _result in checks] == [0, 0, 1, 2]
    assert [result[0] for _args, result in checks] == [0, 1, 2, 3]
    assert [args[0] for args in logged] == [
        "determinism_check[rank 7]: iteration 2 diverged!",
        "determinism_check[rank 7]: iteration 3 diverged!",
        "determinism_check[rank 7]: iteration 4 diverged!",
    ]


def test_default_determinism_check_serializes_selected_rank_and_run_returns(monkeypatch):
    first = DiffusionOutput(images=[_rgb((1, 0, 0))])
    last = DiffusionOutput(images=[_rgb((9, 0, 0))])
    model = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=1,
        rank=1,
        outputs=[first, last],
    )
    saved = []
    model._save_determinism_check_failed_outputs = lambda *args: saved.append(args)
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: None)

    failure_count, baseline = model._determinism_check(1, first, last, 0)

    assert failure_count == 1
    assert baseline is first
    assert saved == [(first, 0, 1), (last, 1, 1)]

    saved.clear()

    output, _timings = model.run({"prompt": "x"})
    assert output is last
    assert len(saved) == 2
    assert saved[0][0] is not first
    assert outputs_equal(saved[0][0], first)
    assert saved[0][1:] == (0, 1)
    assert saved[1] == (last, 1, 1)


def test_result_parsers_match_determinism_failure_artifacts(monkeypatch, tmp_path):
    pipe_args = {"height": 2, "width": 2}
    first = DiffusionOutput(
        images=[_rgb((1, 0, 0))],
        pipe_args=[pipe_args],
    )
    last = DiffusionOutput(
        images=[_rgb((9, 0, 0))],
        pipe_args=[pipe_args],
    )
    model = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=1,
        rank=7,
        outputs=[first, last],
    )
    model.config.output_directory = str(tmp_path)
    model.config.use_torch_compile = False
    model.config.ulysses_degree = 1
    model.config.ring_degree = 1
    model.config.task = None
    model.settings.output_name = "coupling"

    log_file = tmp_path / "stdout.txt"
    logger = logging.getLogger("xfuser.core.utils.runner_utils")
    handler = logging.FileHandler(log_file, encoding="utf-8")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    try:
        model.run({"prompt": "x"})
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)

    assert _count_determinism_failures_in_log(log_file) == 1

    files = [path for path in tmp_path.iterdir() if path != log_file]
    assert len(files) == 2
    size = sum(path.stat().st_size for path in files)
    assert _failed_output_statistics(tmp_path) == (1, size)


def test_default_determinism_check_skips_unselected_rank(monkeypatch):
    first = DiffusionOutput(images=[_rgb((1, 0, 0))])
    last = DiffusionOutput(images=[_rgb((9, 0, 0))])
    model = _make_runner(
        monkeypatch,
        determinism_check=1,
        rank=2,
        determinism_check_report_ranks=frozenset({0, 1}),
    )
    saved = []
    model._save_determinism_check_failed_outputs = lambda *args: saved.append(args)
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: None)

    failure_count, baseline = model._determinism_check(1, first, last, 0)

    assert failure_count == 1
    assert baseline is first
    assert saved == []


def test_batched_and_unbatched_both_report_divergence(monkeypatch):
    unbatched = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=1,
        batch_size=None,
        rank=3,
        outputs=[
            DiffusionOutput(images=[_rgb((1, 0, 0))]),
            DiffusionOutput(images=[_rgb((2, 0, 0))]),
        ],
    )
    batched = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=1,
        batch_size=2,
        rank=3,
        batched_outputs=[
            DiffusionOutput(images=[_rgb((1, 0, 0))]),
            DiffusionOutput(images=[_rgb((2, 0, 0))]),
        ],
    )
    for model in (unbatched, batched):
        checks = _record_determinism_checks(model)
        model._save_determinism_check_failed_outputs = lambda *args: None
        monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: None)
        output, _timings = model.run({"prompt": ["a", "b"]})
        assert len(checks) == 2
        args, result = checks[1]
        assert args[0] == 1
        assert not outputs_equal(args[1], args[2])
        assert result[0] == 1
        assert output.images[0].tobytes() == _rgb((2, 0, 0)).tobytes()


def test_warmup_output_is_not_the_baseline(monkeypatch):
    warmup_seen = []
    first_timed = DiffusionOutput(images=[_rgb((1, 0, 0))])
    later = DiffusionOutput(images=[_rgb((2, 0, 0))])
    model = _make_runner(
        monkeypatch,
        num_iterations=2,
        determinism_check=1,
        warmup_calls=2,
        rank=3,
        outputs=[first_timed, later],
        stub_warmup=True,
    )
    model._run_warmup_calls = lambda input_args: warmup_seen.append(input_args)
    checks = _record_determinism_checks(model)
    model._save_determinism_check_failed_outputs = lambda *args: None
    monkeypatch.setattr(base_model, "log_error", lambda *args, **kwargs: None)

    output, _timings = model.run({"prompt": "x"})

    assert warmup_seen
    assert len(checks) == 2
    args, result = checks[1]
    assert args[2] is later
    assert args[1].images[0].tobytes() == first_timed.images[0].tobytes()
    assert result[0] == 1
    assert output is later


# --- Logging ---------------------------------------------------------------


def _attach_stderr_handler():
    logger = logging.getLogger("xfuser.core.utils.runner_utils")
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    records = []

    class _Recording(logging.Handler):
        def emit(self, record):
            records.append(record)

    recorder = _Recording()
    recorder.setLevel(logging.ERROR)
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    logger.addHandler(recorder)
    return logger, handler, recorder, records, previous_level


def test_log_error_emits_error_level_message_on_stderr(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    logger, handler, recorder, records, previous_level = _attach_stderr_handler()
    try:
        log_error("determinism boom")
        captured = capsys.readouterr()
        assert "determinism boom" in captured.err
        assert records
        assert records[0].levelno == logging.ERROR
        assert records[0].getMessage() == "determinism boom"
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)


def test_log_from_all_processes_bypasses_final_rank_filter(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    logger, handler, recorder, records, previous_level = _attach_stderr_handler()
    try:
        log_error("hidden unless last", log_from_all_processes=False)
        hidden = capsys.readouterr()
        assert "hidden unless last" not in hidden.err
        assert records == []

        log_error("visible on every rank", log_from_all_processes=True)
        shown = capsys.readouterr()
        assert "visible on every rank" in shown.err
        assert records[-1].levelno == logging.ERROR
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)


# --- Configuration ----------------------------------------------------------


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
def test_absent_options_use_determinism_defaults(add_args, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    parser = add_args(FlexibleArgumentParser(description="xDiT"))
    parsed = parser.parse_args(["--model", "test-model"])
    config = xFuserArgs.from_cli_args(parsed)
    assert config.determinism_check == 0
    assert config.determinism_check_report_ranks == frozenset({3})


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
@pytest.mark.parametrize("flag", ["--determinism-check", "--determinism_check"])
def test_determinism_check_options_accept_integer_threshold(add_args, flag, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    parser = add_args(FlexibleArgumentParser(description="xDiT"))
    parsed = parser.parse_args(["--model", "test-model", flag, "3"])
    assert xFuserArgs.from_cli_args(parsed).determinism_check == 3


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("all", frozenset({0, 1, 2, 3})),
        ("first", frozenset({0})),
        ("last", frozenset({3})),
        ("none", frozenset()),
        ("", frozenset()),
        ("3, 1,3", frozenset({1, 3})),
    ],
)
def test_report_ranks_option_normalizes_to_frozenset(
    add_args, value, expected, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    parser = add_args(FlexibleArgumentParser(description="xDiT"))
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--determinism-check-report-ranks",
            value,
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.determinism_check_report_ranks == expected
    assert isinstance(config.determinism_check_report_ranks, frozenset)


def test_report_ranks_underscore_option_is_accepted(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "3")
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(description="xDiT"))
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--determinism_check_report_ranks",
            "0,2",
        ]
    )
    assert xFuserArgs.from_cli_args(
        parsed
    ).determinism_check_report_ranks == frozenset({0, 2})


@pytest.mark.parametrize("value", ["4", "-1", "0,,1", "middle"])
def test_report_ranks_rejects_invalid_values(value, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="determinism_check_report_ranks"):
        xFuserArgs(determinism_check_report_ranks=value)


def test_report_ranks_assumes_single_rank_when_world_size_is_absent(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert xFuserArgs().determinism_check_report_ranks == frozenset({0})


def test_report_ranks_rejects_non_string_input(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(TypeError, match="must be a string"):
        xFuserArgs(determinism_check_report_ranks=[0, 1])


def test_report_ranks_help_warns_about_large_output_files():
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(description="xDiT"))
    action = next(
        action
        for action in parser._actions
        if action.dest == "determinism_check_report_ranks"
    )
    assert "Many failures may produce many large output files" in action.help
    assert "none value is useful with a positive --determinism_check" in action.help
    assert "only log messages, but no files" in action.help


def test_xfuserargs_default_and_runner_dict_propagation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert xFuserArgs().determinism_check == 0
    config = xFuserArgs.from_runner_args(
        {
            "determinism_check": 4,
            "determinism_check_report_ranks": "last",
        }
    )
    assert config.determinism_check == 4
    assert config.determinism_check_report_ranks == frozenset({3})

if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
