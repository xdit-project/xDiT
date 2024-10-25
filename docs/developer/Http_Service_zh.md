## 启动一个文生图服务

### 制作服务镜像

```
docker build -t xdit-service -f ./docker/Dockerfile .
```

或者直接从dockerhub拉取(版本号可能需要更新)
```
docker pull thufeifeibear/xdit-service
```

用下面方式启动一个服务，服务相关参数写在配置脚本config.json里。我们映射了磁盘文件到docker container中，因为需要传递下载的模型文件。注意映射端口6000，如果冲突请修改。

```
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 6000:6000 -v /cfs:/cfs xdit-service --config ./config.json
```

./config.json中内容如下，包括启动GPU卡数，混合并行策略，输出图片的大小，生成图片存储位置等信息。

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

