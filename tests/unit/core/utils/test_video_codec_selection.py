"""CPU-only tests for H.264 encoder selection in encode_video_with_audio.

PyAV wheels bundle an FFmpeg with libx264. An image that builds PyAV against an
LGPL-only FFmpeg has libopenh264 instead. The default call must work in both.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for video utils tests")

from xfuser.core.utils import video_utils  # noqa: E402


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


def test_default_uses_libx264_with_crf_and_preset_when_available(monkeypatch):
    stream = _encode(monkeypatch, {"libx264", "libopenh264", "aac"})

    assert stream.codec == "libx264"
    assert stream.options == {"crf": "12", "preset": "medium"}


def test_default_falls_back_to_libopenh264_without_libx264(monkeypatch):
    stream = _encode(monkeypatch, {"libopenh264", "aac"}, crf=20, preset="slow")

    assert stream.codec == "libopenh264"
    assert stream.options == {"profile": "high", "rc_mode": "buffer"}


def test_explicit_codec_keeps_crf_and_preset(monkeypatch):
    stream = _encode(monkeypatch, {"libx264", "libx265"}, video_codec="libx265", crf=20, preset="slow")

    assert stream.codec == "libx265"
    assert stream.options == {"crf": "20", "preset": "slow"}


def test_default_without_h264_encoder_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="No H.264 encoder"):
        _encode(monkeypatch, {"aac"})
