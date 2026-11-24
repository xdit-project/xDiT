"""CPU-only tests for H.264 encoder selection in encode_video_with_audio.

PyAV wheels bundle an FFmpeg with libx264. An image that builds PyAV against an
LGPL-only FFmpeg has libopenh264 instead. The default call must work in both.
"""

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for video utils tests")

from xfuser.core.utils import video_utils  # noqa: E402


@pytest.fixture
def video_logs():
    """(level, message) pairs that video_utils logged during this test.

    The xfuser loggers do not propagate to root, so caplog does not see them.
    """
    records = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = records.append
    video_utils.logger.addHandler(handler)
    yield lambda: [(record.levelno, record.getMessage()) for record in records]
    video_utils.logger.removeHandler(handler)


class _FakeStream:
    def encode(self, frame=None):
        return []


class _FakeContainer:
    def __init__(self):
        self.streams = []

    def add_stream(self, codec, rate):
        stream = _FakeStream()
        stream.codec = codec
        self.streams.append(stream)
        return stream

    def mux(self, packet):
        pass

    def close(self):
        pass


def _fake_av(codecs):
    container = _FakeContainer()
    return SimpleNamespace(
        codecs_available=set(codecs),
        open=lambda path, mode: container,
        VideoFrame=SimpleNamespace(from_ndarray=lambda array, format: object()),
        container=container,
    )


def _encode(monkeypatch, codecs, **kwargs):
    av = _fake_av(codecs)
    monkeypatch.setattr(video_utils, "_import_av", lambda: av)
    video = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    video_utils.encode_video_with_audio(video, fps=4, output_path="unused.mp4", **kwargs)
    (stream,) = av.container.streams
    return stream


def test_default_uses_libx264_with_crf_and_preset_when_available(monkeypatch, video_logs):
    stream = _encode(monkeypatch, {"libx264", "libopenh264", "aac"})

    assert stream.codec == "libx264"
    assert stream.options == {"crf": "12", "preset": "medium"}
    assert video_logs() == [(logging.INFO, "Encoding video with libx264")]


def test_default_falls_back_to_libopenh264_without_libx264(monkeypatch, video_logs):
    stream = _encode(monkeypatch, {"libopenh264", "aac"})

    assert stream.codec == "libopenh264"
    assert stream.options == {"profile": "high", "rc_mode": "buffer"}
    assert video_logs() == [(logging.INFO, "Encoding video with libopenh264")]


def test_libopenh264_warns_that_it_ignores_crf_and_preset(monkeypatch, video_logs):
    stream = _encode(monkeypatch, {"libopenh264", "aac"}, crf=20, preset="slow")

    assert stream.options == {"profile": "high", "rc_mode": "buffer"}
    warnings = [message for level, message in video_logs() if level == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "ignores crf and preset" in warnings[0]


def test_explicit_codec_keeps_crf_and_preset(monkeypatch, video_logs):
    stream = _encode(monkeypatch, {"libx264", "libx265"}, video_codec="libx265", crf=20, preset="slow")

    assert stream.codec == "libx265"
    assert stream.options == {"crf": "20", "preset": "slow"}
    assert video_logs() == []


def test_default_without_h264_encoder_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="No H.264 encoder"):
        _encode(monkeypatch, {"aac"})


@pytest.mark.parametrize(
    ("codecs", "expected_codec", "expected_options"),
    [
        ({"libx264", "libopenh264"}, "libx264", {"crf": "12", "preset": "medium"}),
        ({"libopenh264"}, "libopenh264", {"profile": "high", "rc_mode": "buffer"}),
    ],
)
def test_encode_video_selects_the_encoder_and_adds_no_audio_stream(
    monkeypatch, codecs, expected_codec, expected_options
):
    av = _fake_av(codecs)
    monkeypatch.setattr(video_utils, "_import_av", lambda: av)

    video_utils.encode_video(torch.zeros(2, 16, 16, 3, dtype=torch.uint8), fps=4, output_path="unused.mp4")

    (stream,) = av.container.streams
    assert stream.codec == expected_codec
    assert stream.options == expected_options


def test_encode_video_with_audio_is_encode_video():
    assert video_utils.encode_video_with_audio is video_utils.encode_video


def test_float_frames_are_clipped_and_rounded_to_uint8():
    frames = video_utils._frames_to_uint8(torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5]).numpy())

    assert str(frames.dtype) == "uint8"
    assert frames.tolist() == [0, 0, 128, 255, 255]
