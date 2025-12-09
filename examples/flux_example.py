import logging
import time
import torch
import diffusers
import torch.distributed
from xfuser.config.diffusers import has_valid_diffusers_version, get_minimum_diffusers_version

if not has_valid_diffusers_version("flux"):
    minimum_diffusers_version = get_minimum_diffusers_version("flux")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Flux.")

from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    text_encoder_2 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    cache_args = {
            "use_teacache": engine_args.use_teacache,
            "use_fbcache": engine_args.use_fbcache,
            "rel_l1_thresh": 0.12,
            "return_hidden_states_first": False,
            "num_steps": input_config.num_inference_steps,
        }

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )

    if args.use_fp8_gemms:
        import itertools
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, _is_linear, quantize_
        from torchao.quantization.quantize_.common import KernelPreference
        logging.info(f"rank {local_rank} quantizing transformer linear layers")
        for module in itertools.chain(pipe.transformer.transformer_blocks, pipe.transformer.single_transformer_blocks):
            quantize_(
                module, 
                config=Float8DynamicActivationFloat8WeightConfig(
                            granularity=PerTensor(),
                            set_inductor_config=False,
                            kernel_preference=KernelPreference.AUTO
                        ),
                filter_fn=_is_linear,
                device=f"cuda:{local_rank}",
            )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
