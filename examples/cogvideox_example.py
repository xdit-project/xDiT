import logging
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video

# for teacache
import argparse
import numpy as np
from typing import Any, Dict, Optional, Tuple,  Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers, export_to_video, load_image
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline

coefficients_dict = {
    "CogVideoX-2b":[-3.10658903e+01,  2.54732368e+01, -5.92380459e+00,  1.75769064e+00, -3.61568434e-03],
    "CogVideoX-5b":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX-5b-I2V":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX1.5-5B":[ 2.50210439e+02, -1.65061612e+02,  3.57804877e+01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V":[ 1.22842302e+02, -1.04088754e+02,  2.62981677e+01, -3.06009921e-01, 3.71213220e-02],
}

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.enable_teacache:
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = emb
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0            
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                # 4. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            emb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=emb,
                            image_rotary_emb=image_rotary_emb,
                        )

                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
        else:
            # 4. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B and CogvideoX1.5-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    
    # TeaCache
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.rel_l1_thresh = 0.2
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None
    pipe.transformer.__class__.num_steps = input_config.num_inference_steps
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.coefficients = coefficients_dict["CogVideoX1.5-5B"]
    pipe.transformer.__class__.forward = teacache_forward
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    torch._inductor.config.reorder_for_compute_comm_overlap = True
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    
    # warmup
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=1,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
