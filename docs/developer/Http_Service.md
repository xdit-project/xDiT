## Launch a Text-to-Image Http Service

Launch an HTTP-based text-to-image service that generates images from textual descriptions (prompts) using the DiT model. 
The generated images can either be returned directly to users or saved to a specified disk location.
For example, the following command launches a HTTP service with 4 GPUs, 2 Ulysses parallel degree, 2 PipeFusion parallel degree, and the model path is `./models/FLUX.1-schnell`.

```bash
python ./entrypoints/launch.py --world_size 4 --ulysses_parallel_degree 2 --pipefusion_parallel_degree 2 --model_path /your_model_path/FLUX.1-schnell
```

`--world_size` must equal the product of the parallel degrees (`--ulysses_parallel_degree`, `--ring_degree`, `--pipefusion_parallel_degree`, and 2 with `--use_cfg_parallel`). For example, `--world_size 4 --ulysses_parallel_degree 2 --ring_degree 2` runs 4-way sequence parallelism.

Other options:

| Option | Default | Meaning |
|---|---|---|
| `--dtype {bf16,fp16,fp32}` | `fp16` | Precision the model is loaded and run in. FLUX.1 is released in bf16, so `--dtype bf16` can suit it better on GPUs that support bf16. |
| `--host`, `--port` | `0.0.0.0`, `6000` | Address and port the HTTP server listens on. |
| `--master_port` | `29500` | Port the workers use to set up `torch.distributed`. The address comes from the `MASTER_ADDR` environment variable (default `127.0.0.1`). Change the port to run several servers on one machine. |
| `--save_disk_path` | unset | Directory to save images to when a request does not set `save_disk_path`. |


To an example HTTP request is shown below. The `save_disk_path` parameter is optional - if set, the generated image will be saved to the specified directory on disk; if not set, the server's `--save_disk_path` is used, and if that is unset too, the image is returned base64-encoded in the response.

```bash
curl -X POST "http://localhost:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a cute rabbit",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5, 
           "save_disk_path": "/tmp"
         }'
```
