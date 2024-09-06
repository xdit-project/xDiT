## Launching a Text-to-Image Http Service

### Creating the Service Image

```
docker build -t xdit-server:0.3.1 -f ./docker/Dockerfile .
```

or (version number may need to be updated)

```
docker pull thufeifeibear/xdit-service:0.3.1
```

Start the service using the following command. The service-related parameters are written in the configuration script `config.json`. We have mapped disk files to the Docker container because we need to pass the downloaded model files. Note the mapping of port 6000; if there is a conflict, please modify it.

```
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 6000:6000 -v /cfs:/cfs xdit-server:0.3.1 --config ./config.json
```

The content of `./config.json` includes the number of GPUs to use, the mixed parallelism strategy, the size of the output images, the storage location for generated images, and other information.

```
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

Access the service using an HTTP request. The `save_disk_path` is an optional parameter. If not set, an image is returned; if set, the generated image is saved in the corresponding directory on the disk.

```
curl -X POST http://127.0.0.1:6001/generate \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "A lovely rabbit",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5,
           "save_disk_path": "/tmp"
         }'
```