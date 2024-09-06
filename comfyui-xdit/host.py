import os
import time
import torch
import torch.distributed as dist
import pickle
import io
import logging
import base64
import torch.multiprocessing as mp

from PIL import Image
from flask import Flask, request, jsonify
from xfuser import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserFluxPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserArgs,
)
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
)

app = Flask(__name__)

# 设置 NCCL 超时和错误处理
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# 全局变量
pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False


def setup_logger():
    global logger
    rank = dist.get_rank()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    setup_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    model_name = engine_config.model_config.model.split("/")[-1]

    pipeline_map = {
        "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
        "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
        "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
        "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
        "FLUX.1-schnell": xFuserFluxPipeline,
    }

    PipelineClass = pipeline_map.get(model_name)
    if PipelineClass is None:
        raise NotImplementedError(f"{model_name} is currently not supported!")

    pipe = PipelineClass.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")

    pipe.prepare_run(input_config)
    logger.info("Model initialization completed")
    initialized = True  # 设置初始化完成标志


def generate_image_parallel(
    prompt, num_inference_steps, seed, cfg, save_disk_path=None
):
    global pipe, local_rank, input_config
    logger.info(f"Starting image generation with prompt: {prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(seed),
        guidance_scale=cfg,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if save_disk_path is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        file_path = os.path.join(save_disk_path, filename)
        if is_dp_last_group():
            # Create the directory if it doesn't exist
            os.makedirs(save_disk_path, exist_ok=True)
            # Save the image to the specified directory
            output.images[0].save(file_path)
            logger.info(f"Image saved to: {file_path}")

        output = file_path
    else:
        if is_dp_last_group():
            # serialize output object
            output_bytes = pickle.dumps(output)

            # send output to rank 0
            dist.send(
                torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0
            )
            dist.send(
                torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0
            )

            logger.info(f"Output sent to rank 0")

        if dist.get_rank() == 0:
            # recv from rank world_size - 1
            size = torch.tensor(0, device=f"cuda:{local_rank}")
            dist.recv(size, src=dist.get_world_size() - 1)
            output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
            dist.recv(output_bytes, src=dist.get_world_size() - 1)

            # deserialize output object
            output = pickle.loads(output_bytes.cpu().numpy().tobytes())

    return output, elapsed_time


@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    prompt = data.get("prompt", input_config.prompt)
    num_inference_steps = data.get(
        "num_inference_steps", input_config.num_inference_steps
    )
    seed = data.get("seed", input_config.seed)
    cfg = data.get("cfg", 8.0)
    save_disk_path = data.get("save_disk_path")

    # Check if save_disk_path is valid, if not, set it to a default directory
    if save_disk_path:
        if not os.path.isdir(save_disk_path):
            default_path = os.path.join(os.path.expanduser("~"), "tacodit_output")
            os.makedirs(default_path, exist_ok=True)
            logger.warning(
                f"Invalid save_disk_path. Using default path: {default_path}"
            )
            save_disk_path = default_path
    else:
        save_disk_path = None

    logger.info(
        f"Request parameters: prompt='{prompt}', steps={num_inference_steps}, seed={seed}, save_disk_path={save_disk_path}"
    )
    # Broadcast request parameters to all processes
    params = [prompt, num_inference_steps, seed, cfg, save_disk_path]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    if save_disk_path:
        # output is a disk path
        output_base64 = ""
        image_path = save_disk_path
    else:
        # Ensure output is not None before accessing its attributes
        if output and hasattr(output, "images") and output.images:
            output_base64 = base64.b64encode(output.images[0].tobytes()).decode("utf-8")
        else:
            output_base64 = ""
        image_path = ""

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_base64 if not save_disk_path else output,
        "save_to_disk": save_disk_path is not None,
    }

    # logger.info(f"Sending response: {response}")
    return jsonify(response)


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            # 非主进程等待广播的参数
            params = [None] * 5
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


# curl -X POST http://127.0.0.1:6000/generate \
#      -H "Content-Type: application/json" \
#      -d '{
#            "prompt": "A lovely rabbit",
#            "num_inference_steps": 50,
#            "seed": 42,
#            "cfg": 7.5,
#            "save_disk_path": true
#          }'
if __name__ == "__main__":
    initialize()

    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()
