import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserArgs
from xfuser.ray.pipeline.pipeline_utils import RayDiffusionPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.model_executor.pipelines import xFuserFluxPipeline

def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    model_name = engine_config.model_config.model.split("/")[-1]
    PipelineClass = xFuserFluxPipeline
    text_encoder_2 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    pipe = RayDiffusionPipeline.from_pretrained(
        PipelineClass=PipelineClass,
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )
    pipe.prepare_run(input_config)

    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"elapsed time:{elapsed_time}")
    if not os.path.exists("results"):
        os.mkdir("results")
        
    for i, result in enumerate(output):
        if result is not None:
            image = result.images[0]
            image.save(
                f"./results/{model_name}_result_{i}.png"
            )
            print(
                f"image {i} saved to ./results/{model_name}_result_{i}.png"
            )
            break


if __name__ == "__main__":
    main()
