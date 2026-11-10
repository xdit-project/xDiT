# HTTP service example

`entrypoints/launch.py` provides a single-node FastAPI/Ray image service.
It uses dedicated pipeline wrappers and a separate model map.
Install xDiT from this checkout with the service dependencies:

```bash
python3 -m pip install -e ".[ray]" fastapi uvicorn
```

Checkpoint directory names select PixArt-alpha, PixArt-Sigma, SD3 medium, HunyuanDiT,
or FLUX.1 dev/schnell wrappers. The service exposes a subset of runner features.
See [the launcher](https://github.com/xdit-project/xDiT/blob/main/entrypoints/launch.py) for the model map and request schema.

## Launch FLUX.1-schnell

Use a local checkpoint directory ending in `FLUX.1-schnell`. This command creates two GPU workers with Ulysses degree 2.
The service listens on port 6000:

```bash
python3 entrypoints/launch.py \
    --model_path /your_model_path/FLUX.1-schnell \
    --world_size 2 \
    --ulysses_parallel_degree 2
```

| Launcher option | Behavior |
| --- | --- |
| `--model_path` | Required checkpoint path; the last path component selects the pipeline wrapper. |
| `--world_size` | Number of Ray GPU workers; must match the requested parallel layout. |
| `--ulysses_parallel_degree` | Sequence-parallel degree; defaults to 1. |
| `--pipefusion_parallel_degree` | Pipeline-stage count; defaults to 1. |
| `--use_cfg_parallel` | Split active CFG branches; supported only by the relevant model wrappers, not FLUX.1. |

The launcher accepts `--ring_degree` and `--save_disk_path` but does not forward them
to the engine configuration. Set the save path in the request.
For Ring parallelism or the full model and option set, use the [unified runner](../runner/runner.md).

## Generate an image

The request overrides the generic 50-step service default with four steps for FLUX.1-schnell:

```bash
curl -X POST "http://localhost:6000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"A small rabbit in a garden","num_inference_steps":4,"seed":42,"cfg":0.0}'
```

Requests accept `prompt`, `num_inference_steps`, `seed`, `cfg`, `height`, `width`,
and optional `save_disk_path`. Height and width default to 1024.
The endpoint returns JSON with `message`, `elapsed_time`, `output`, and `save_to_disk`.

Without a save path, `output` contains a base64-encoded PNG and `save_to_disk` is false.
With `save_disk_path`, the service writes a PNG on the server. It returns the path
and sets `save_to_disk` to true. Both responses use JSON.
