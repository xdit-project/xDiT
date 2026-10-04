import os
import time
import torch
import ray
import io
import logging
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from xfuser import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserFluxPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserArgs,
)

if __package__:
    from .server_args import DTYPES, parse_args, to_xfuser_args
else:  # run as a script: python entrypoints/launch.py
    from server_args import DTYPES, parse_args, to_xfuser_args


# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = 42
    cfg: Optional[float] = 7.5
    save_disk_path: Optional[str] = None
    height: Optional[int] = 1024
    width: Optional[int] = 1024

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "num_inference_steps": 50,
                "seed": 42,
                "cfg": 7.5,
                "height": 1024,
                "width": 1024,
            }
        }


app = FastAPI()


@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(
        self,
        xfuser_args: xFuserArgs,
        rank: int,
        world_size: int,
        dtype: torch.dtype,
        master_addr: str,
        master_port: int,
    ):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args, dtype)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args: xFuserArgs, dtype: torch.dtype):
        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        # PipeFusion sizes its receive buffers from the runtime dtype
        self.engine_config.runtime_config.dtype = dtype

        model_name = self.engine_config.model_config.model.split("/")[-1]
        pipeline_map = {
            "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
            "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
            "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
            "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
            "FLUX.1-schnell": xFuserFluxPipeline,
            "FLUX.1-dev": xFuserFluxPipeline,
        }

        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")

        self.logger.info(f"Initializing model {model_name} from {xfuser_args.model} in {dtype}")

        self.pipe = PipelineClass.from_pretrained(
            pretrained_model_name_or_path=xfuser_args.model,
            engine_config=self.engine_config,
            torch_dtype=dtype,
        ).to("cuda")

        self.pipe.prepare_run(self.input_config)
        self.logger.info("Model initialization completed")

    def generate(self, request: GenerateRequest):
        try:
            start_time = time.time()
            output = self.pipe(
                height=request.height,
                width=request.width,
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                output_type="pil",
                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                guidance_scale=request.cfg,
                max_sequence_length=self.input_config.max_sequence_length,
            )
            elapsed_time = time.time() - start_time

            if self.pipe.is_dp_last_group():
                if request.save_disk_path:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"generated_image_{timestamp}.png"
                    file_path = os.path.join(request.save_disk_path, filename)
                    os.makedirs(request.save_disk_path, exist_ok=True)
                    output.images[0].save(file_path)
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True,
                    }
                else:
                    # Convert to base64
                    buffered = io.BytesIO()
                    output.images[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": img_str,
                        "save_to_disk": False,
                    }
            return None

        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


class Engine:
    def __init__(
        self,
        world_size: int,
        xfuser_args: xFuserArgs,
        dtype: torch.dtype,
        master_addr: str,
        master_port: int,
        save_disk_path: Optional[str] = None,
    ):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()

        self.save_disk_path = save_disk_path
        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(
                xfuser_args,
                rank=rank,
                world_size=world_size,
                dtype=dtype,
                master_addr=master_addr,
                master_port=master_port,
            )
            for rank in range(num_workers)
        ]

    async def generate(self, request: GenerateRequest):
        if request.save_disk_path is None and self.save_disk_path is not None:
            request = request.model_copy(update={"save_disk_path": self.save_disk_path})
        results = ray.get([worker.generate.remote(request) for worker in self.workers])

        return next(path for path in results if path is not None)


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(status_code=400, detail="Height and width must be positive")
        if request.num_inference_steps <= 0:
            raise HTTPException(status_code=400, detail="num_inference_steps must be positive")

        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    args = parse_args()

    engine = Engine(
        world_size=args.world_size,
        xfuser_args=to_xfuser_args(args),
        dtype=DTYPES[args.dtype],
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=args.master_port,
        save_disk_path=args.save_disk_path,
    )

    # Start the server
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
