import pytest
import torch


def test_encode_video_with_audio(tmp_path):
    av = pytest.importorskip("av")

    from xfuser.core.utils.video_utils import encode_video_with_audio

    video = torch.zeros(4, 16, 16, 3, dtype=torch.uint8)
    video[:, :, :, 0] = torch.arange(4, dtype=torch.uint8)[:, None, None] * 50
    audio = torch.zeros(2, 3200)
    output_path = tmp_path / "test.mp4"

    encode_video_with_audio(
        video,
        fps=4,
        output_path=str(output_path),
        audio=audio,
        audio_sample_rate=32000,
    )

    container = av.open(output_path)
    assert [stream.type for stream in container.streams] == ["video", "audio"]
    frames = list(container.decode(video=0))
    assert len(frames) == 4
    assert frames[0].width == 16
    assert frames[0].height == 16


def test_encode_video_with_audio_libopenh264_high_profile(tmp_path):
    av = pytest.importorskip("av")
    if "libopenh264" not in av.codecs_available:
        pytest.skip("PyAV's FFmpeg has no libopenh264 encoder")

    from xfuser.core.utils.video_utils import encode_video_with_audio

    video = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8)
    output_path = tmp_path / "test.mp4"

    encode_video_with_audio(
        video,
        fps=8,
        output_path=str(output_path),
        audio=torch.zeros(2, 8000),
        audio_sample_rate=8000,
        video_codec="libopenh264",
    )

    container = av.open(output_path)
    assert [stream.type for stream in container.streams] == ["video", "audio"]
    assert container.streams.video[0].codec_context.profile == "High"
    assert len(list(container.decode(video=0))) == 8
