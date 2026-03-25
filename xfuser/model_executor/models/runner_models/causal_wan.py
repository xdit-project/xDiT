import copy
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.model_executor.models.transformers.transformer_causal_wan import xFuserCausalWanTransformer3DWrapper
from xfuser.model_executor.pipelines.pipeline_causal_wan_test import xFuserCausalWanPipeline
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser.core.utils.runner_utils import (
    log,
)



def _setup_parallel_vae(vae) -> None:
    """Parallelizes the VAE decoder using distvae."""
    try:
        from distvae.modules.adapters.vae.decoder_adapters import WanDecoderAdapter
        patched_decoder = WanDecoderAdapter(
            vae.decoder, vae_group=get_vae_parallel_group().device_group
        ).to(vae.device)
        vae.decoder = patched_decoder
        log("Parallel VAE decoder enabled successfully.")
    except ImportError:
        raise ValueError(
            "DistVAE library is missing or does not support WanDecoderAdapter. "
            "Try installing latest DistVAE from https://github.com/xdit-project/DistVAE."
        )
    except Exception as e:
        raise ValueError(f"Failed to patch VAE decoder. {e}")


@register_model("CausalWan")
class xFuserCausalWanModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=False,   # SP incompatible with KV cache initially
        ring_degree=False,
        fully_shard_degree=False,
        use_fp8_gemms=False,
        use_parallel_vae=False,
    )
    default_input_values = DefaultInputValues(
        height=512,
        width=512,
        num_inference_steps=8, # DMD used, doesn't matter
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        mod_value=8,
        fps=16,
        model_output_type="video",
        model_name="FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers",
        output_name="causal_wan_i2v",
        fp8_gemm_module_list=["transformer.blocks"],
        flow_shift=12,
    )


    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserCausalWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        transformer_2 = xFuserCausalWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
        )
        pipe = xFuserCausalWanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            transformer_2=transformer_2,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            num_frames_per_block=3, # Processes X frames at a time
            sliding_window_num_frames=21, # Sliding window size
            context_noise=0, # Noise to add to the context as a regularization
            local_attn_size=-1, # Local attention size, -1 means no local attention
            sink_size=0, # Sink size, 0 means no sink, i.e, no context in kept
            max_attention_size=32760, # Max KV size for attention
            dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],
            flow_shift=self.settings.flow_shift,
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)