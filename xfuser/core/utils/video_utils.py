from fractions import Fraction
from itertools import chain
from typing import Iterator

import numpy as np
import PIL.Image
import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_CRF = 12
_DEFAULT_PRESET = "medium"

# H.264 encoders in order of preference. libx264 is GPL. Some FFmpeg builds
# are LGPL-only and do not include it. These builds can supply libopenh264.
_H264_ENCODERS = ("libx264", "libopenh264")

# libopenh264 does not use crf or preset. Its defaults give a 2 Mbps Constrained
# Baseline stream. With the High profile and "buffer" rate control, quality is
# near x264 CRF 12 at about 1.5x the bitrate. The bitrate cannot be set in
# "buffer" mode.
_OPENH264_OPTIONS = {"profile": "high", "rc_mode": "buffer"}


def _import_av():
    try:
        import av
    except ImportError as error:
        raise ImportError("PyAV is required to encode videos. Install it with `pip install av`.") from error
    return av


def _select_video_codec(av_module) -> str:
    for codec in _H264_ENCODERS:
        if codec in av_module.codecs_available:
            return codec
    raise RuntimeError(f"No H.264 encoder is available in the FFmpeg that PyAV uses. Expected one of {_H264_ENCODERS}.")


def _video_codec_options(video_codec: str, crf: int, preset: str) -> dict[str, str]:
    if video_codec == "libopenh264":
        return dict(_OPENH264_OPTIONS)
    return {"crf": str(crf), "preset": preset}


def _frames_to_uint8(video: np.ndarray) -> np.ndarray:
    """Return ``video`` as uint8. Scale floating-point frames from ``[0, 1]`` to ``[0, 255]``."""
    if not np.issubdtype(video.dtype, np.floating):
        return video
    # Reuse one float buffer. Each full-video float copy can be close to 1 GB at 720p.
    scaled = np.clip(video, 0.0, 1.0)
    scaled *= 255
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint8)


def _prepare_audio_stream(container, audio_sample_rate: int, audio_bitrate: int):
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.bit_rate = audio_bitrate
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _write_audio(container, audio_stream, samples, audio_sample_rate: int, av_module) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T
    if samples.shape[1] != 2:
        raise ValueError(f"Expected stereo audio, got shape {tuple(samples.shape)}.")

    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame = av_module.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame.sample_rate = audio_sample_rate

    codec_context = audio_stream.codec_context
    resampler = av_module.audio.resampler.AudioResampler(
        format=codec_context.format or "fltp",
        layout=codec_context.layout or "stereo",
        rate=codec_context.sample_rate or audio_sample_rate,
    )
    next_pts = 0
    for resampled_frame in resampler.resample(frame):
        if resampled_frame.pts is None:
            resampled_frame.pts = next_pts
        next_pts += resampled_frame.samples
        resampled_frame.sample_rate = audio_sample_rate
        container.mux(audio_stream.encode(resampled_frame))

    for packet in audio_stream.encode():
        container.mux(packet)


def encode_video(
    video: list[PIL.Image.Image] | list[np.ndarray] | np.ndarray | torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    output_path: str,
    *,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    video_chunks_number: int = 1,
    video_codec: str | None = None,
    pixel_format: str = "yuv420p",
    crf: int = _DEFAULT_CRF,
    preset: str = _DEFAULT_PRESET,
    audio_bitrate: int = 192_000,
) -> None:
    """Encode RGB video frames, with optional stereo audio, to an H.264 file.

    Frames are in ``[F, H, W, C]`` format. Tensor inputs must be uint8. Float NumPy
    arrays, and lists of float NumPy frames, must be in ``[0, 1]``. The function
    scales them to uint8. PIL images and integer NumPy arrays are used as they are.

    If ``audio`` is ``None``, the file contains only a video stream.

    If ``video_codec`` is ``None``, the function uses ``libx264`` when it is available.
    If not, it uses ``libopenh264``. ``crf`` and ``preset`` apply only to codecs that
    support them. ``libopenh264`` ignores them and uses High profile with buffer
    rate control.
    """
    av = _import_av()
    if video_codec is None:
        video_codec = _select_video_codec(av)
        logger.info(f"Encoding video with {video_codec}")
    if video_codec == "libopenh264" and (crf != _DEFAULT_CRF or preset != _DEFAULT_PRESET):
        logger.warning(f"libopenh264 ignores crf and preset. Using {_OPENH264_OPTIONS} instead.")

    if isinstance(video, list):
        video = np.stack([np.asarray(frame) for frame in video])
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(_frames_to_uint8(video))

    if isinstance(video, torch.Tensor):
        video = iter(torch.tensor_split(video, video_chunks_number, dim=0))

    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    video_stream = container.add_stream(video_codec, rate=int(fps))
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = pixel_format
    video_stream.options = _video_codec_options(video_codec, crf, preset)

    audio_stream = None
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided.")
        audio_stream = _prepare_audio_stream(
            container,
            audio_sample_rate=audio_sample_rate,
            audio_bitrate=audio_bitrate,
        )

    for video_chunk in chain([first_chunk], video):
        for frame_array in video_chunk.to("cpu").numpy():
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(
            container,
            audio_stream,
            audio,
            audio_sample_rate,
            av,
        )

    container.close()


# The LTX, MiniMax-H3 and Prism runners use this name.
encode_video_with_audio = encode_video
