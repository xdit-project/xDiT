import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserArgs
from xfuser.ray.pipeline.pipeline_utils import RayDiffusionPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.model_executor.pipelines import xFuserStableDiffusion3Pipeline

def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    model_name = engine_config.model_config.model.split("/")[-1]
    PipelineClass = xFuserStableDiffusion3Pipeline
    
    # equal to
    # text_encoder_3 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
    # but load encoder in worker
    encoder_kwargs = {
        'text_encoder_3': {
            'model_class': T5EncoderModel,
            'pretrained_model_name_or_path': engine_config.model_config.model,
            'subfolder': 'text_encoder_3',
            'torch_dtype': torch.float16
        },
    }
    
    # if args.use_fp8_t5_encoder:
    #     from optimum.quanto import freeze, qfloat8, quantize
    #     print(f"rank {local_rank} quantizing text encoder 2")
    #     quantize(text_encoder_3, weights=qfloat8)
    #     freeze(text_encoder_3)

    pipe = RayDiffusionPipeline.from_pretrained(
        PipelineClass=PipelineClass,
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        **encoder_kwargs
    )
    pipe.prepare_run(input_config)

    torch.cuda.reset_peak_memory_stats()
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
    if not os.path.exists("results"):
        os.mkdir("results")
        
    for _, images in enumerate(output):
        if images is not None:
            image = images[0]
            path = f"./results/{model_name}_ray_result.png"
            image.save(path)
            print(
                f"image saved to {path}"
            )
            break


if __name__ == "__main__":
    main()