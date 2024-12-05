import os
import time
import torch
import torch.distributed as dist
import pickle
import io
import logging
import base64
import torch.multiprocessing as mp
from queue import Queue
import threading
import asyncio
from collections import deque

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

# Set NCCL timeout and error handling
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Global variables
pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False
args = None

# a global queue to store request prompts
request_queue = deque()
queue_lock = threading.Lock()
queue_event = threading.Event()
results_store = {}  # store request results


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
    global pipe, engine_config, input_config, local_rank, initialized, args
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--max_queue_size", type=int, default=4,
                       help="Maximum size of the request queue")
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
    initialized = True  # Set initialization completion flag


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
    # single gpu didn't need to distribute
    elif dist.get_world_size() > 1:
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
def queue_image_request():
    logger.info("Received POST request for image generation")
    data = request.json
    request_id = str(time.time())
    
    with queue_lock:
        # Check queue size
        if len(request_queue) >= args.max_queue_size:
            return jsonify({
                "error": "Queue is full, please try again later",
                "queue_size": len(request_queue)
            }), 503
        
        request_params = {
            "id": request_id,
            "prompt": data.get("prompt", input_config.prompt),
            "num_inference_steps": data.get("num_inference_steps", input_config.num_inference_steps),
            "seed": data.get("seed", input_config.seed),
            "cfg": data.get("cfg", 8.0),
            "save_disk_path": data.get("save_disk_path")
        }
        
        request_queue.append(request_params)
        queue_event.set()
    
    return jsonify({
        "message": "Request accepted",
        "request_id": request_id,
        "status_url": f"/status/{request_id}"
    }), 202

@app.route("/status/<request_id>", methods=["GET"])
def check_status(request_id):
    if request_id in results_store:
        result = results_store.pop(request_id) 
        return jsonify(result), 200
    
    position = None
    with queue_lock:
        for i, req in enumerate(request_queue):
            if req["id"] == request_id:
                position = i
                break
    
    if position is not None:
        return jsonify({
            "status": "pending",
            "queue_position": position
        }), 202
    
    return jsonify({"status": "not_found"}), 404

def process_queue():
    while True:
        queue_event.wait()
        
        with queue_lock:
            if not request_queue:
                queue_event.clear()
                continue
            
            params = request_queue.popleft()
            if not request_queue:
                queue_event.clear()
        
        try:
            # Extract parameters
            request_id = params["id"]
            prompt = params["prompt"]
            num_inference_steps = params["num_inference_steps"]
            seed = params["seed"]
            cfg = params["cfg"]
            save_disk_path = params["save_disk_path"]
            
            # Broadcast parameters to all processes
            broadcast_params = [prompt, num_inference_steps, seed, cfg, save_disk_path]
            dist.broadcast_object_list(broadcast_params, src=0)
            
            # Generate image and get results
            output, elapsed_time = generate_image_parallel(*broadcast_params)
            
            # Process output results
            if save_disk_path:
                # output is disk path
                result = {
                    "message": "Image generated successfully",
                    "elapsed_time": f"{elapsed_time:.2f} sec",
                    "output": output,  # This is the file path
                    "save_to_disk": True
                }
            else:
                # Process base64 output
                if output and hasattr(output, "images") and output.images:
                    output_base64 = base64.b64encode(output.images[0].tobytes()).decode("utf-8")
                else:
                    output_base64 = ""
                
                result = {
                    "message": "Image generated successfully",
                    "elapsed_time": f"{elapsed_time:.2f} sec",
                    "output": output_base64,
                    "save_to_disk": False
                }
            
            # Store results
            results_store[request_id] = result
            
        except Exception as e:
            logger.error(f"Error processing request {params['id']}: {str(e)}")
            results_store[request_id] = {
                "error": str(e),
                "status": "failed"
            }


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        # process 0 will process the queue in a separate thread
        queue_thread = threading.Thread(target=process_queue, daemon=True)
        queue_thread.start()
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            # Non-master processes wait for broadcasted parameters
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