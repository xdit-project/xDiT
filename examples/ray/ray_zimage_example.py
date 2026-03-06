import os
import time
import torch
import torch.distributed
from transformers import AutoModel

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.ray.pipeline.pipeline_utils import RayDiffusionPipeline
from xfuser.model_executor.pipelines.pipeline_zimage import xFuserZImagePipeline


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    model_name = engine_config.model_config.model.split("/")[-1]

    # Z-Image uses a Qwen3 model as text encoder.
    # Pass as encoder_kwargs so each Ray worker loads it independently
    # into its own GPU process (same mechanism as T5 in Flux/HunyuanDiT).
    encoder_kwargs = {
        "text_encoder": {
            "model_class": AutoModel,
            "pretrained_model_name_or_path": engine_config.model_config.model,
            "subfolder": "text_encoder",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    }
    # NOTE: fp8 quantization of the Qwen text encoder is not yet supported
    # in the Ray worker path (no post-load hook in RayDiffusionPipeline).
    # Use the non-Ray zimage_example.py with --use_fp8_t5_encoder instead.

    pipe = RayDiffusionPipeline.from_pretrained(
        PipelineClass=xFuserZImagePipeline,
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        **encoder_kwargs,
    )
    pipe.prepare_run(input_config)

    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=input_config.guidance_scale,
        max_sequence_length=input_config.max_sequence_length,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time:.2f} sec")

    if not os.path.exists("results"):
        os.mkdir("results")

    for _, images in enumerate(output):
        if images is not None:
            image = images[0]
            path = f"./results/{model_name}_ray_result.png"
            image.save(path)
            print(f"image saved to {path}")
            break


if __name__ == "__main__":
    main()
