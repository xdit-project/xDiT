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
