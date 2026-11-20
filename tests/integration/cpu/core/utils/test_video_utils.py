import numpy as np
import PIL.Image
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


def _float_video(frame_count=5, height=32, width=48):
    """Frames like the ones diffusers video pipelines return with ``output_type="np"``."""
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)
    video = np.broadcast_to(gradient[None, None, :, None], (frame_count, height, width, 3)).copy()
    video[..., 1] = np.arange(frame_count, dtype=np.float32)[:, None, None] / frame_count
    return video


def _decode_rgb(av, output_path):
    container = av.open(output_path)
    assert [stream.type for stream in container.streams] == ["video"]
    return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


@pytest.mark.parametrize(
    "to_input",
    [
        pytest.param(lambda video: video, id="float-array"),
        pytest.param(list, id="list-of-float-arrays"),
        pytest.param(
            lambda video: [PIL.Image.fromarray((frame * 255).round().astype(np.uint8)) for frame in video],
            id="list-of-pil-images",
        ),
    ],
)
def test_encode_video_writes_only_a_video_stream(tmp_path, to_input):
    av = pytest.importorskip("av")

    from xfuser.core.utils.video_utils import encode_video

    video = _float_video()
    output_path = tmp_path / "test.mp4"

    encode_video(to_input(video), fps=16, output_path=str(output_path))

    decoded = _decode_rgb(av, output_path)
    assert decoded.shape == (5, 32, 48, 3)
    expected = (video * 255).round()
    assert np.abs(decoded.astype(np.float32) - expected).mean() < 8


def test_encode_video_keeps_uint8_frames_that_are_all_zero_or_one(tmp_path):
    av = pytest.importorskip("av")

    from xfuser.core.utils.video_utils import encode_video

    output_path = tmp_path / "test.mp4"

    encode_video(np.ones((3, 32, 32, 3), dtype=np.uint8), fps=8, output_path=str(output_path))

    assert _decode_rgb(av, output_path).max() < 16
