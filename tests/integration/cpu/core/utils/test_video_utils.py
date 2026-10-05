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


@pytest.mark.parametrize("input_kind", ["numpy_uint8", "numpy_float", "tensor", "pil"])
def test_video_encoding_preserves_pixel_scale(tmp_path, input_kind):
    av = pytest.importorskip("av")
    import numpy as np
    from PIL import Image

    from xfuser.core.utils.video_utils import encode_video_with_audio

    pixels = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    pixels[:, ::2, :, :] = 1
    expected = pixels
    if input_kind == "numpy_uint8":
        video = pixels
    elif input_kind == "numpy_float":
        video = pixels.astype(np.float32)
        expected = pixels * 255
    elif input_kind == "tensor":
        video = torch.from_numpy(pixels)
    else:
        video = [Image.fromarray(frame) for frame in pixels]
    output_path = tmp_path / "pixels.mkv"

    encode_video_with_audio(video, fps=4, output_path=str(output_path), video_codec="ffv1", pixel_format="bgr0")

    with av.open(output_path) as container:
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    np.testing.assert_array_equal(decoded, expected)
