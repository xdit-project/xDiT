"""Single-GPU Wan text-to-video HTTP service using the shared model runner."""

import argparse
import base64
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import ray
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


class GenerateVideoRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    height: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    num_frames: int | None = Field(default=None, gt=0)
    num_inference_steps: int | None = Field(default=None, gt=0)
    guidance_scale: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    fps: int | None = Field(default=None, gt=0)
    save_to_disk: bool = True

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value

    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, value: int | None) -> int | None:
        if value is not None and value % 16:
            raise ValueError("Wan height and width must be divisible by 16")
        return value

    @field_validator("num_frames")
    @classmethod
    def validate_frames(cls, value: int | None) -> int | None:
        if value is not None and value % 4 != 1:
            raise ValueError("Wan num_frames must be 4k + 1")
        return value


class VideoGenerator:
    def __init__(self, config: dict):
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        from xfuser.runner import xFuserModelRunner

        self.runner = xFuserModelRunner(config)
        self.output_directory = Path(config["output_directory"])

    def generate(self, request: GenerateVideoRequest) -> dict:
        input_args = vars(self.runner.config).copy()
        input_args.update(request.model_dump(exclude={"fps", "save_to_disk"}))
        input_args["input_images"] = []
        input_args = self.runner.preprocess_args(input_args)
        self.runner.initialize(input_args)
        start = time.perf_counter()
        output, _ = self.runner.run(input_args)
        if not output.videos or len(output.videos) != 1:
            raise RuntimeError("Expected exactly one generated video")
        fps = request.fps or self.runner.model.settings.fps
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = self.output_directory if request.save_to_disk else Path(temp_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"generated_video_{uuid.uuid4().hex}.mp4"
            export_to_video(output.videos[0], str(path), fps=fps)
            result = str(path.resolve()) if request.save_to_disk else base64.b64encode(path.read_bytes()).decode()
        return {
            "output": result,
            "save_to_disk": request.save_to_disk,
            "media_type": "video/mp4",
            "fps": fps,
            "elapsed_time": time.perf_counter() - start,
        }


def create_app(config: dict) -> FastAPI:
    # One actor serializes model access, including after a disconnected HTTP client.
    @asynccontextmanager
    async def lifespan(app):
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray.init()
        worker = ray.remote(num_gpus=1)(VideoGenerator).remote(config)
        app.state.worker = worker
        try:
            yield
        finally:
            ray.kill(worker)
            if owns_ray:
                ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.post("/generate_video")
    async def generate_video(request: GenerateVideoRequest):
        try:
            return await app.state.worker.generate.remote(request)
        except Exception as error:
            logging.getLogger(__name__).exception("Video generation failed")
            raise HTTPException(status_code=500, detail="Video generation failed; see server logs") from error

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--output_directory", default="output")
    args = parser.parse_args()
    config = {
        "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "output_directory": args.output_directory,
        "warmup_calls": 0,
        "num_iterations": 1,
    }
    uvicorn.run(create_app(config), host=args.host, port=args.port)
