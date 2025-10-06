import time
import torch
import functools
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from typing import Any, Dict, List, Tuple, Callable, Optional, Union
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    get_sp_group,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
)
from xfuser.model_executor.pipelines.pipeline_wan_i2v import xFuserWanImageToVideoPipeline

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        #torch.distributed.breakpoint(0)
        # Step 1. Split tensors
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        # Step 2. Perform the original forward
        output =  original_forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, return_dict, attention_kwargs)

        return_dict = not isinstance(output, tuple)
        sample = output[0]

        # Step 3. Merge the output from all GPUs
        sample = get_sp_group().all_gather(sample, dim=-2)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    print(engine_args)
    engine_config, input_config = engine_args.create_config()
    print(input_config)
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    dtype = torch.bfloat16
    device = torch.device("cuda")
    pipe = xFuserWanImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    parallelize_transformer(pipe)
    pipe = pipe.to(f"cuda:{local_rank}")

    image = load_image(
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    #pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        image=image,
        prompt=input_config.prompt,
        #negative_prompt=X,
        num_inference_steps=40,
        num_frames=81,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]
    if pipe.is_dp_last_group():
        export_to_video(output, "i2v_output.mp4")

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
