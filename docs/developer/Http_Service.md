## Launch a Text-to-Image Http Service

Launch an HTTP-based text-to-image service that generates images from textual descriptions (prompts) using the DiT model. The generated images can either be returned directly to users or saved to a specified disk location. To enhance processing efficiency, we've implemented a concurrent processing mechanism: requests containing prompts are stored in a request queue, and DiT processes these requests in parallel across multiple GPUs.

```bash
python ./entrypoints/launch.py --world_size 4 --ulysses_parallel_degree 2 --pipefusion_parallel_degree 2 --model_path /your_model_path
```


To interact with the service, send HTTP requests as shown below. The `save_disk_path` parameter is optional - if not set, the image will be returned directly; if set, the generated image will be saved to the specified directory on disk.

```bash
curl -X POST http://127.0.0.1:6000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "A lovely rabbit",
           "num_inference_steps": 20,
           "seed": 42,
           "cfg": 7.5,
           "save_disk_path": "/tmp"
         }'
```
