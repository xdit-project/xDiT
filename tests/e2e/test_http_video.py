"""Opt-in real Wan generation through HTTP and a Ray GPU actor."""

import base64
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        os.environ.get("XDIT_HTTP_VIDEO_E2E") != "1",
        reason="Set XDIT_HTTP_VIDEO_E2E=1 to load Wan2.1-T2V-14B weights",
    ),
]


def test_wan_video_http_round_trip(tmp_path):
    pytest.importorskip("ray")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    av = pytest.importorskip("av")
    import numpy as np
    import torch
    from fastapi.testclient import TestClient

    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU with enough memory for Wan2.1-T2V-14B")
    from entrypoints.launch_video import create_app

    config = {
        "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "output_directory": str(tmp_path / "outputs"),
        "warmup_calls": 0,
        "num_iterations": 1,
    }
    with TestClient(create_app(config)) as client:
        for count, save_to_disk in [(5, False), (9, True)]:
            response = client.post(
                "/generate_video",
                json={
                    "prompt": "A small robot waves at the camera",
                    "height": 128,
                    "width": 128,
                    "num_frames": count,
                    "num_inference_steps": 2,
                    "guidance_scale": 1.0,
                    "seed": 0,
                    "fps": 8,
                    "save_to_disk": save_to_disk,
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["media_type"] == "video/mp4"
            assert result["save_to_disk"] is save_to_disk
            if save_to_disk:
                path = Path(result["output"])
                assert path.parent == tmp_path / "outputs"
            else:
                path = tmp_path / "base64.mp4"
                path.write_bytes(base64.b64decode(result["output"], validate=True))
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
                assert len(frames) == count
                assert container.streams.video[0].average_rate == 8
                arrays = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])
            assert arrays.shape == (count, 128, 128, 3)
            assert arrays.std() > 0
