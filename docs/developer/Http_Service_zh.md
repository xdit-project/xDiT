## 启动一个文生图服务


启动一个基于HTTP的文本生成图像服务。该服务接收用户的文本描述（prompt），利用DiT模型生成相应的图像。生成的图像可以直接返回给用户，或保存到指定的磁盘位置。为了提高处理效率，我们实现了一个并发处理机制：使用请求队列来存储incoming requests，并通过xdit在多个GPU上并行处理队列中的请求。

```
python ./http-service/launch_host.py --config ./http-service/config.json --max_queue_size 4
```

./config.json中默认内容如下，包括启动GPU卡数，混合并行策略，输出图片的大小，生成图片存储位置等信息。

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

使用http请求访问服务，"save_disk_path"是一个可选项，如果不设置则返回一个图片，如果设置则将生成图片存在磁盘上对应位置的目录中。

```
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
