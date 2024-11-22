## Launch a Text-to-Image Service

Launch an HTTP-based text-to-image service that generates images from textual descriptions (prompts) using the DiT model. The generated images can either be returned directly to users or saved to a specified disk location. To enhance processing efficiency, we've implemented a concurrent processing mechanism: requests containing prompts are stored in a request queue, and DiT processes these requests in parallel across multiple GPUs.

```bash
python ./http-service/launch_host.py --config ./http-service/config.json --max_queue_size 4
```

The default content in `./config.json` is shown below, which includes settings for the number of GPU cards, hybrid parallelism strategy, output image dimensions, and image storage location:

```json
{
    "nproc_per_node": 2,
    "model": "/cfs/dit/HunyuanDiT-v1.2-Diffusers",
    "pipefusion_parallel_degree": 1,
    "ulysses_degree": 2,
    "ring_degree": 1,
    "use_cfg_parallel": false,
    "height": 512,
    "width": 512,
    "save_disk_path": "/cfs/dit/output"
}
```

To interact with the service, send HTTP requests as shown below. The `save_disk_path` parameter is optional - if not set, the image will be returned directly; if set, the generated image will be saved to the specified directory on disk.

```bash
curl -X POST http://127.0.0.1:6000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "A lovely rabbit",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5,
           "save_disk_path": "/tmp"
         }'
```
