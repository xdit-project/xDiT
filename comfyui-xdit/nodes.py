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

class XfuserPipelineHostLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (list([
                    "/cfs/dit/HunyuanDiT-v1.2-Diffusers",
                    "/cfs/dit/models/PixArt-XL-2-1024-MS",
                    "/cfs/dit/PixArt-Sigma-XL-2-2K-MS",
                    "/cfs/dit/models/stable-diffusion-3-medium-diffusers",
                    "/cfs/dit/FLUX.1-schnell",
                ]), ),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "nproc_per_node": ("INT", {"default": 8, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("XFUSER_PIPELINE_HOST",)

    FUNCTION = "launch_host"

    CATEGORY = "Xfuser"        

    def launch_host(self, model_path, width, height, warmup_steps, nproc_per_node):
        ulysses_degree = 1
        ring_degree = 1
        CFG_ARGS=""

        if nproc_per_node == 8:
            pipefusion_parallel_degree = 4
            ulysses_degree = 1
            CFG_ARGS="--use_cfg_parallel"   
        elif nproc_per_node == 4:
            pipefusion_parallel_degree = 4
            ulysses_degree = 1
        elif nproc_per_node == 2:
            pipefusion_parallel_degree = 2
        elif nproc_per_node == 1:
            pipefusion_parallel_degree = 1
        else:
            pass

        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc_per_node}',
            './custom_nodes/comfyui-xdit-server/host.py',
            f'--model={model_path}',
            f'--pipefusion_parallel_degree={pipefusion_parallel_degree}',
            f'--ulysses_degree={ulysses_degree}',
            f'--ring_degree={ring_degree}',
            f'--height={height}',
            f'--width={width}',
            f'--warmup_steps={warmup_steps}',
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
            time.sleep(1)  # 等待一秒后重试

        return (f"{host}/generate", )


class XfuserPipelineHost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRINGC",),
                "negative": ("STRINGC",),
                "host": ("XFUSER_PIPELINE_HOST",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "predict"

    CATEGORY = "Xfuser"

    def predict(self, host, positive, negative, steps, seed):
        url = host
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
            image_path = response_data.get("image_path", "")
            output.images[0].save(image_path)

            print("Output object deserialized successfully")
            print(f"Image saved to {image_path}")
        else:
            output = None
            print("No output object received")
        images = output.images
        return (convert_images_to_tensors(images), )

NODE_CLASS_MAPPINGS = {
    "XfuserClipTextEncode": XfuserClipTextEncode,
    "XfuserPipelineHostLoader": XfuserPipelineHostLoader,
    "XfuserPipelineHost": XfuserPipelineHost
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XfuserClipTextEncode": "XfuserClipTextEncode",
    "XfuserPipelineHostLoader": "XfuserPipelineHostLoader",
    "XfuserPipelineHost": "XfuserPipelineHost"
}
