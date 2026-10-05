import base64
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("save_to_disk", [False, True])
def test_video_worker_encodes_requests_without_mutating_defaults(tmp_path, save_to_disk):
    pytest.importorskip("ray")
    pytest.importorskip("fastapi")
    av = pytest.importorskip("av")
    pytest.importorskip("imageio_ffmpeg")
    from entrypoints.launch_video import GenerateVideoRequest, VideoGenerator
    from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput

    # Replace the accelerator/model boundary; exercise real request and MP4 handling.
    from types import SimpleNamespace

    class Runner:
        config = SimpleNamespace(seed=17, prompt="initial")
        model = SimpleNamespace(settings=SimpleNamespace(fps=16))

        def preprocess_args(self, args):
            return args

        def initialize(self, args):
            pass

        def run(self, args):
            assert args["seed"] == 0
            assert args["prompt"] == "new prompt"
            return DiffusionOutput(videos=[np.full((5, 32, 48, 3), 0.5, dtype=np.float32)]), []

    worker = object.__new__(VideoGenerator)
    worker.runner = Runner()
    worker.output_directory = tmp_path / "output"
    request = GenerateVideoRequest(prompt="new prompt", seed=0, fps=8, save_to_disk=save_to_disk)
    first = worker.generate(request)
    second = worker.generate(request)
    assert vars(worker.runner.config) == {"seed": 17, "prompt": "initial"}
    assert first["media_type"] == "video/mp4"
    assert first["fps"] == 8
    assert first["save_to_disk"] is save_to_disk
    if save_to_disk:
        assert first["output"] != second["output"]
        video_path = Path(first["output"])
    else:
        assert not worker.output_directory.exists()
        video_path = tmp_path / "decoded.mp4"
        video_path.write_bytes(base64.b64decode(first["output"]))
    with av.open(str(video_path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 5
        assert (frames[0].height, frames[0].width) == (32, 48)
        assert container.streams.video[0].average_rate == 8
        np.testing.assert_allclose(frames[0].to_ndarray(format="rgb24"), 127, atol=3)


def test_video_http_validation_and_actor_errors():
    pytest.importorskip("ray")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from entrypoints.launch_video import create_app

    class Generate:
        calls = 0

        async def remote(self, request):
            self.calls += 1
            if request.prompt == "error":
                raise RuntimeError("private worker details")
            return {"output": "encoded-video", "save_to_disk": False}

    from types import SimpleNamespace

    generate = Generate()
    app = create_app({})
    app.state.worker = SimpleNamespace(generate=generate)
    # HTTP transport is real; skip startup to replace the GPU actor boundary.
    client = TestClient(app)
    for payload in [
        {"prompt": " "},
        {"prompt": "ok", "height": 17},
        {"prompt": "ok", "width": 0},
        {"prompt": "ok", "num_frames": 4},
        {"prompt": "ok", "num_inference_steps": 0},
        {"prompt": "ok", "guidance_scale": -1},
    ]:
        assert client.post("/generate_video", json=payload).status_code == 422
    assert generate.calls == 0
    result = client.post("/generate_video", json={"prompt": "ok", "save_to_disk": False})
    assert result.status_code == 200
    assert result.json()["output"] == "encoded-video"
    error = client.post("/generate_video", json={"prompt": "error"})
    assert error.status_code == 500
    assert "private worker details" not in error.text
