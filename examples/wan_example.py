import os
import math
import time
import torch
import functools
import numpy as np
from typing import Any, Dict, Optional, Union

from xfuser.config.diffusers import get_minimum_diffusers_version, has_valid_diffusers_version
if not has_valid_diffusers_version("wan"):
    minimum_diffusers_version = get_minimum_diffusers_version("wan")
    raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Wan.")

from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import Fp32LayerNorm

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_sp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    initialize_runtime_state,
    is_dp_last_group,
    shard_dit,
    shard_t5_encoder,
)
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanAttnProcessor

TASK_FPS = {
    "i2v": 16,
    "t2v": 16,
    "ti2v": 24,
}

TASK_FLOW_SHIFT = {
    "i2v": 5,
    "t2v": 12,
    "ti2v": 5,
}

# Wrapper to only wrap the transformer in case it exists, i.e. Wan2.2
def maybe_transformer_2(transformer_2):
    if transformer_2 is not None:
        return functools.wraps(transformer_2.__class__.forward)
    else:
        return (lambda f:f)

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    transformer_2 = pipe.transformer_2



    @functools.wraps(transformer.__class__.forward)
    @maybe_transformer_2(transformer_2)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        else:
            # Wan2.1 fails if we chunk encoder_hidden_states when cross attention is used. Should cross attention really be sharded?
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        max_chunked_sequence_length = int(math.ceil(hidden_states.shape[1] / get_sequence_parallel_world_size())) * get_sequence_parallel_world_size()
        sequence_pad_amount = max_chunked_sequence_length - hidden_states.shape[1]
        hidden_states = torch.cat([
            hidden_states,
            torch.zeros(batch_size, sequence_pad_amount, hidden_states.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
        ], dim=1)
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        if ts_seq_len is not None: # (wan2.2 ti2v)
            temb = torch.cat([
                temb,
                torch.zeros(batch_size, sequence_pad_amount, temb.shape[2], device=temb.device, dtype=temb.dtype)
            ], dim=1)
            timestep_proj = torch.cat([
                timestep_proj,
                torch.zeros(batch_size, sequence_pad_amount, timestep_proj.shape[2], timestep_proj.shape[3], device=timestep_proj.device, dtype=timestep_proj.dtype)
            ], dim=1)
            temb = torch.chunk(temb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            timestep_proj = torch.chunk(timestep_proj, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, sequence_pad_amount):
            freqs = torch.cat([
                freqs,
                torch.zeros(1, sequence_pad_amount, freqs.shape[2], freqs.shape[3], device=freqs.device, dtype=freqs.dtype)
            ], dim=1)
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, sequence_pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, sequence_pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)


        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)


        # Removing excess padding to get back to original sequence length
        hidden_states = hidden_states[:, :math.prod([post_patch_num_frames, post_patch_height, post_patch_width]), :]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    new_forward_1 = new_forward.__get__(transformer)
    transformer.forward = new_forward_1
    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor()
        block.attn2.processor = xFuserWanAttnProcessor()

    if transformer_2 is not None:
        new_forward_2 = new_forward.__get__(transformer_2)
        transformer_2.forward = new_forward_2
        for block in transformer_2.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor()


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["i2v", "t2v", "ti2v"],
        help="The task to run."
    )
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    is_i2v_task = args.task == "i2v" or (args.task == "ti2v" and args.img_file_path != None)
    task_pipeline = WanImageToVideoPipeline if is_i2v_task else WanPipeline
    pipe = task_pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler.config.flow_shift = TASK_FLOW_SHIFT[args.task]
    initialize_runtime_state(pipe, engine_config)
    parallelize_transformer(pipe)

    # Shard model
    pipe.transformer = shard_dit(
        pipe.transformer, local_rank, "blocks"
    ) if args.shard_dit else pipe.transformer.to(f"cuda:{local_rank}")
    if pipe.transformer_2 is not None:
        pipe.transformer_2 = shard_dit(
            pipe.transformer_2, local_rank, "blocks"
        ) if args.shard_dit else pipe.transformer_2.to(f"cuda:{local_rank}")
    pipe.text_encoder = shard_t5_encoder(
        pipe.text_encoder, local_rank, "block"
    ) if args.shard_t5_encoder else pipe.text_encoder.to(f"cuda:{local_rank}")
    pipe.vae = pipe.vae.to(f"cuda:{local_rank}")
    pipe.scheduler.config.flow_shift = TASK_FLOW_SHIFT[args.task]

    if not args.img_file_path and args.task == "i2v":
        raise ValueError("Please provide an input image path via --img_file_path. This may be a local path or a URL.")

    if is_i2v_task:
        image = load_image(args.img_file_path)
        max_area = input_config.height * input_config.width
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        if is_dp_last_group():
            print("Max area is calculated from input height and width values, but the aspect ratio for the output video is retained from the input image.")
            print(f"Input image resolution: {image.height}x{image.width}")
            print(f"Generating a video with resolution: {height}x{width}")
    else: # T2V or TI2V with no image
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = input_config.height // mod_value * mod_value
        width = input_config.width // mod_value * mod_value
        if height != input_config.height or width != input_config.width:
            if is_dp_last_group():
                print(f"Adjusting height and width to be multiples of {mod_value}. New dimensions: {height}x{width}")
        image = None

    def run_pipe(input_config, image):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optional_kwargs = {}
        if image:
            optional_kwargs["image"] = image
        output = pipe(
            height=height,
            width=width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            num_frames=input_config.num_frames,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            **optional_kwargs,
        ).frames[0]
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
        torch.cuda.synchronize()
        if is_dp_last_group():
            print(f"Iteration took {end - start}s, Peak memory: {peak_memory / 1024 ** 2:.2f} MB")
        return output

    if args.use_fp8_gemms:
        import itertools
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, _is_linear, quantize_
        from torchao.quantization.quantize_.common import KernelPreference
        for module in itertools.chain(pipe.transformer.blocks, pipe.transformer_2.blocks) if pipe.transformer_2 is not None else pipe.transformer.blocks:
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

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")
        if pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="default")

        # one step to warmup the torch compiler
        _ = run_pipe(input_config, image)

    output = run_pipe(input_config, image)
    if is_dp_last_group():
        file_name = f"{args.task}_output.mp4"
        export_to_video(output, file_name, fps=TASK_FPS[args.task])
        print(f"Output video saved to {file_name}")

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
