import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.executor.gpu_executor import RayDiffusionPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size
from xfuser.executor.gpu_executor import RayDiffusionPipeline
from xfuser.worker.worker import xFuserPixArtAlphaPipeline, xFuserPixArtSigmaPipeline, xFuserStableDiffusion3Pipeline, xFuserHunyuanDiTPipeline, xFuserFluxPipeline

def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    pipeline_map = {
            "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
            "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
            "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
            "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
            "FLUX.1-schnell": xFuserFluxPipeline,
        }
    model_name = engine_config.model_config.model.split("/")[-1]
    PipelineClass = pipeline_map.get(model_name)    
    if PipelineClass is None:
        raise NotImplementedError(f"{model_name} is currently not supported!")
    text_encoder_3 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        quantize(text_encoder_3, weights=qfloat8)
        freeze(text_encoder_3)

    pipe = RayDiffusionPipeline.from_pretrained(
        PipelineClass=PipelineClass,
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        text_encoder_3=text_encoder_3,
    )
    pipe.prepare_run(input_config)

    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"elapsed time:{elapsed_time}")


if __name__ == "__main__":
    main()