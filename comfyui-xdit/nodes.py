import time
import os
import torch
import torch.distributed
import subprocess
import requests
import base64
import pickle

from .utils import convert_images_to_tensors
from multiprocessing import Process

current_dir = os.path.dirname(__file__)
custom_nodes_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(custom_nodes_dir)

models_dir = os.path.join(base_dir, "models")
checkpoints_dir = os.path.join(models_dir, "checkpoints")
xfuser_models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]

class XfuserClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"multiline": True}),
            "negative": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRINGC", "STRINGC",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "concat_embeds"

    CATEGORY = "Xfuser"

    def concat_embeds(self, positive, negative):
        return (positive, negative,)

class XfuserPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (xfuser_models,), 
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "device_num": (list([1, 2, 4, 8]),),
            }
        }

    RETURN_TYPES = ("XFUSER_PIPELINE",)

    FUNCTION = "launch_host"

    CATEGORY = "Xfuser"        

    def launch_host(self, model_name, width, height, device_num):
        assert model_name in ["HunyuanDiT-v1.2-Diffusers", "PixArt-XL-2-1024-MS", "PixArt-Sigma-XL-2-2K-MS", "stable-diffusion-3-medium-diffusers", "FLUX.1-schnell"], \
            "model_name must be one of the following: HunyuanDiT-v1.2-Diffusers, PixArt-XL-2-1024-MS, PixArt-Sigma-XL-2-2K-MS, stable-diffusion-3-medium-diffusers, FLUX.1-schnell"
        model_path = os.path.join(checkpoints_dir, model_name)
        nproc_per_node = min(device_num, torch.cuda.device_count())
        ulysses_degree = 1
        ring_degree = 1
        CFG_ARGS=""

        if nproc_per_node == 8:
            pipefusion_parallel_degree = 4
            CFG_ARGS="--use_cfg_parallel"   
        elif nproc_per_node == 4:
            pipefusion_parallel_degree = 4
        elif nproc_per_node == 2:
            pipefusion_parallel_degree = 2
        elif nproc_per_node == 1:
            pipefusion_parallel_degree = 1
        else:
            pass

        if model_name == "FLUX.1-schnell":
            pipefusion_parallel_degree = 1
            ulysses_degree = nproc_per_node
            ring_degree = 1
            CFG_ARGS=""

        host_script_path = os.path.join(current_dir, 'host.py')

        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc_per_node}',
            host_script_path,
            f'--model={model_path}',
            f'--pipefusion_parallel_degree={pipefusion_parallel_degree}',
            f'--ulysses_degree={ulysses_degree}',
            f'--ring_degree={ring_degree}',
            f'--height={height}',
            f'--width={width}',
            f'--prompt="setup"',
            CFG_ARGS,
        ]

        # 过滤掉空字符串
        cmd = [arg for arg in cmd if arg]
         # 打印命令行参数
        print("Running command:", " ".join(cmd))

        # 使用subprocess启动子进程
        process = subprocess.Popen(cmd)

        host = 'http://localhost:6000'

        # 轮询 host 直到初始化完成
        initialize_url = f"{host}/initialize"
        while True:
            try:
                response = requests.get(initialize_url)
                if response.status_code == 200 and response.json().get("status") == "initialized":
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        return (f"{host}/generate", )


class XfuserSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("XFUSER_PIPELINE",),
                "positive": ("STRINGC",),
                "negative": ("STRINGC",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "predict"

    CATEGORY = "Xfuser"

    def predict(self, pipeline, positive, negative, steps, seed, cfg):
        url = pipeline
        data = {
            "prompt": positive,            
            "num_inference_steps": steps,
            "seed": seed,
        }
        response = requests.post(url, json=data)
        response_data = response.json()

        # 反序列化 output 对象
        output_base64 = response_data.get("output", "")
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            output = pickle.loads(output_bytes)
            print("Output object deserialized successfully")
        else:
            output = None
            print("No output object received")
        images = output.images
        return (convert_images_to_tensors(images), )

NODE_CLASS_MAPPINGS = {
    "XfuserClipTextEncode": XfuserClipTextEncode,
    "XfuserPipelineLoader": XfuserPipelineLoader,
    "XfuserSampler": XfuserSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XfuserClipTextEncode": "XfuserClipTextEncode",
    "XfuserPipelineLoader": "XfuserPipelineLoader",
    "XfuserSampler": "XfuserSampler"
}
