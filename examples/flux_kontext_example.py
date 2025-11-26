# Flux inference with USP
# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_flux.py

import functools

import logging
import time
import torch
from xfuser.config.diffusers import has_valid_diffusers_version, get_minimum_diffusers_version
from typing import List, Optional

if not has_valid_diffusers_version("flux_kontext"):
    minimum_diffusers_version = get_minimum_diffusers_version("flux_kontext")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Flux-Kontext.")

from diffusers import DiffusionPipeline, FluxKontextPipeline
from diffusers.utils import load_image

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxAttnProcessor

def pad_to_sp_divisable(tensor: torch.Tensor, padding_length: int, dim: int) -> torch.Tensor:

    padding =  torch.zeros(
        *tensor.shape[:dim], padding_length, *tensor.shape[dim + 1 :], dtype=tensor.dtype, device=tensor.device
    )
    tensor = torch.cat([tensor, padding], dim=dim)
    return tensor

def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):

        sp_world_size = get_sequence_parallel_world_size()
        sequence_length = hidden_states.shape[1]
        padding_length = (sp_world_size - (sequence_length % sp_world_size)) % sp_world_size
        if padding_length > 0:
            hidden_states = pad_to_sp_divisable(hidden_states, padding_length, dim=1)
            img_ids = pad_to_sp_divisable(img_ids, padding_length, dim=0)
        assert hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0, \
            f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]


        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if padding_length > 0:
            sample = sample[:, :-padding_length, :]
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserFluxAttnProcessor()


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    if not args.img_file_path:
        raise ValueError("Please provide an input image path via --img_file_path. This may be a local path or a URL.")
    image = load_image(args.img_file_path)

    pipe = FluxKontextPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_input_parameters(
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        max_condition_sequence_length=512,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )

    parallelize_transformer(pipe)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")

        # one step to warmup the torch compiler
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            max_area=input_config.height * input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=1,
            output_type=input_config.output_type,
            guidance_scale=2.5,
            image=image,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).images

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        max_area=input_config.height * input_config.width,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=2.5,
        image=image,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_kontext_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}_{input_config.height}x{input_config.width}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
