from fractions import Fraction
from itertools import chain
from typing import Iterator

import numpy as np
import PIL.Image
import torch


def _import_av():
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "PyAV is required to encode videos with audio. Install it with `pip install av`."
        ) from error
    return av


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


def encode_video_with_audio(
    video: list[PIL.Image.Image] | np.ndarray | torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    output_path: str,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    video_chunks_number: int = 1,
    video_codec: str = "libx264",
    pixel_format: str = "yuv420p",
    crf: int = 12,
    preset: str = "medium",
    audio_bitrate: int = 192_000,
) -> None:
    """Encode RGB video frames with optional stereo audio and explicit quality controls.

    Tensor inputs are expected to be uint8 RGB frames in ``[F, H, W, C]`` format.
    NumPy inputs may instead contain normalized floating-point values in ``[0, 1]``.
    """
    av = _import_av()

    if isinstance(video, list):
        video = torch.from_numpy(np.stack([np.asarray(frame) for frame in video]))
    elif isinstance(video, np.ndarray):
        if np.all((video >= 0) & (video <= 1)):
            video = (video * 255).round().astype(np.uint8)
        video = torch.from_numpy(video)

    if isinstance(video, torch.Tensor):
        video = iter(torch.tensor_split(video, video_chunks_number, dim=0))

    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    video_stream = container.add_stream(video_codec, rate=int(fps))
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = pixel_format
    video_stream.options = {
        "crf": str(crf),
        "preset": preset,
    }

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
