import os
import time
import torch
import functools
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.models.modeling_outputs import Transformer2DModelOutput


from typing import Any, Dict, Optional, Union
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
)
from xfuser.model_executor.layers.attention_processor import xFuserWanAttnProcessor

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
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
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
    if transformer_2 is not None:
        new_forward_2 = new_forward.__get__(transformer_2)
        transformer_2.forward = new_forward_2

    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor()
        block.attn2.processor = xFuserWanAttnProcessor()
    if transformer_2 is not None:
        for block in transformer_2.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor()



def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    rank = int(os.getenv("RANK", 0))
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    dtype = torch.bfloat16
    device = torch.device("cuda")
    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    pipe = WanImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16
    )
    initialize_runtime_state(pipe, engine_config)
    parallelize_transformer(pipe)
    pipe = pipe.to(f"cuda:{local_rank}")

    image = load_image(
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )

    def run_pipe(input_config, image):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            image=image,
            prompt=input_config.prompt,
            #negative_prompt=X,
            num_inference_steps=input_config.num_inference_steps,
            num_frames=input_config.num_frames,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).frames[0]
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
        print(f"Iteration took {end - start}s, Peak memory: {peak_memory / 1024 ** 2:.2f} MB")
        return output

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
        if pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        _ = run_pipe(input_config, image)

    output = run_pipe(input_config, image)
    if is_dp_last_group():
        export_to_video(output, "i2v_output.mp4", fps=16)

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
