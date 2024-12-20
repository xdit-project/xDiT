import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserArgs
from xfuser.ray.pipeline.pipeline_utils import RayDiffusionPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.model_executor.pipelines import xFuserPixArtAlphaPipeline, xFuserPixArtSigmaPipeline, xFuserStableDiffusion3Pipeline, xFuserHunyuanDiTPipeline, xFuserFluxPipeline
import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    model_name = engine_config.model_config.model.split("/")[-1]
    PipelineClass = xFuserStableDiffusion3Pipeline
    text_encoder_3 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        print(f"rank {local_rank} quantizing text encoder 2")
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
    # output is a list of results from each worker, we take the last one
    for i, image in enumerate(output[-1].images):
        image.save(
            f"/data/results/{model_name}_result_{i}.png"
        )
        print(
            f"image {i} saved to /data/results/{model_name}_result_{i}.png"
        )


if __name__ == "__main__":
    main()