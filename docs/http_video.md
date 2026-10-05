# Text-to-video HTTP service

The video service runs `Wan-AI/Wan2.1-T2V-14B-Diffusers` on one GPU through
`xFuserModelRunner`. It uses the runner's model loading, preprocessing, and inference
paths. Install Ray, FastAPI, Uvicorn, and the video encoding dependencies used by
Diffusers. Model weights are downloaded on the first valid generation request.

```bash
python entrypoints/launch_video.py --output_directory output/videos
curl http://127.0.0.1:6000/generate_video \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A small robot waves at the camera", "seed":42}'
```

The response contains `output`, `save_to_disk`, `media_type`, `fps`, and elapsed
seconds. By default, `output` is an absolute MP4 path on the server. Set
`save_to_disk` to `false` to receive base64-encoded MP4 bytes instead; the temporary
file is removed after encoding. Persistent files have unique names and are retained
in the server's configured output directory.

Optional generation fields are `negative_prompt`, `height`, `width`, `num_frames`,
`num_inference_steps`, and `guidance_scale`. Omitted values use the Wan runner's
defaults. Dimensions must be positive multiples of 16; frames must be `4k + 1`.
`seed` defaults to 42. Optional `fps` controls playback rate, not inference.

Requests share one Ray GPU actor and execute serially. Awaiting generation does
not block the HTTP event loop. A disconnected client does not interrupt an
already running model call. This entrypoint supports one GPU and one model;
distributed video serving and image-conditioned generation are outside its scope.
The existing image service remains available through `entrypoints/launch.py`.

Run the opt-in real-model HTTP test on a GPU with enough memory:

```bash
XDIT_HTTP_VIDEO_E2E=1 python -m pytest tests/e2e/test_http_video.py -v
```

It generates short, low-resolution videos with two denoising steps to check
request handling and MP4 decoding. It does not assess video quality.

Use `--host` and `--port` to configure the listening address. The default is local
access only. Configure access control before exposing the service publicly.
