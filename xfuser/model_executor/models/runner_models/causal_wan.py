import copy
import os
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.model_executor.models.transformers.transformer_causal_wan import xFuserCausalWanTransformer3DWrapper
from xfuser.model_executor.pipelines.pipeline_causal_wan import xFuserCausalWanPipeline
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)



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
        flow_shift=3,
        valid_tasks=["t2v", "i2v"],
    )

    _NUM_FRAMES_PER_BLOCK = 3
    _SLIDING_WINDOW_NUM_FRAMES = 21
    _CONTEXT_NOISE = 0
    _LOCAL_ATTN_SIZE = -1
    _SINK_SIZE = 0
    _MAX_ATTENTION_SIZE = 32760
    _DMD_DENOISING_STEPS = [1000, 850, 700, 550, 350, 275, 200, 125]
    _FLOW_SHIFT = 3


    def _load_transformer(self, subfolder: str) -> xFuserCausalWanTransformer3DWrapper:
        """Load transformer, falling back to manual loading if weight index is mismatched."""
        try:
            return xFuserCausalWanTransformer3DWrapper.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                subfolder=subfolder,
            )
        except (OSError, ValueError, RuntimeError):
            from safetensors.torch import load_file
            config = xFuserCausalWanTransformer3DWrapper.load_config(
                self.settings.model_name, subfolder=subfolder
            )
            model = xFuserCausalWanTransformer3DWrapper.from_config(config)
            if os.path.isdir(self.settings.model_name):
                weight_path = os.path.join(
                    self.settings.model_name, subfolder, "diffusion_pytorch_model.safetensors"
                )
            else:
                from huggingface_hub import hf_hub_download
                weight_path = hf_hub_download(
                    self.settings.model_name,
                    filename=f"{subfolder}/diffusion_pytorch_model.safetensors",
                )
            state_dict = load_file(weight_path)
            model.load_state_dict(state_dict, strict=True)
            return model.to(torch.bfloat16)

    def _load_model(self) -> DiffusionPipeline:
        transformer = self._load_transformer("transformer")
        transformer_2 = self._load_transformer("transformer_2")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.settings.model_name, subfolder="scheduler",
        )
        # Register aliases for non-standard class names in model_index.json
        # so diffusers' from_pretrained doesn't fail during class resolution.
        import diffusers
        if not hasattr(diffusers, "SelfForcingFlowMatchScheduler"):
            diffusers.SelfForcingFlowMatchScheduler = FlowMatchEulerDiscreteScheduler
        if not hasattr(diffusers, "CausalWanTransformer3DModel"):
            from diffusers import WanTransformer3DModel
            diffusers.CausalWanTransformer3DModel = WanTransformer3DModel
        pipe = xFuserCausalWanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            image=input_args.get("image"),
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            num_frames_per_block=self._NUM_FRAMES_PER_BLOCK, # Processes X frames at a time
            sliding_window_num_frames=self._SLIDING_WINDOW_NUM_FRAMES, # Sliding window size
            context_noise=self._CONTEXT_NOISE, # Noise to add to the context as a regularization
            local_attn_size=self._LOCAL_ATTN_SIZE, # Local attention size, -1 means no local attention
            sink_size=self._SINK_SIZE, # Sink size, 0 means no sink, i.e, no context in kept
            max_attention_size=self._MAX_ATTENTION_SIZE, # Max KV size for attention
            dmd_denoising_steps=self._DMD_DENOISING_STEPS,
            flow_shift=self._FLOW_SHIFT,
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        images = input_args.get("input_images", [])
        if images:
            image = images[0]
            input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        task = input_args.get("task")
        images = input_args.get("input_images", [])
        if task == "i2v" and len(images) != 1:
            raise ValueError("Exactly one input image is required for CausalWan I2V mode.")

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.transformer_2 = torch.compile(self.pipe.transformer_2, mode="default")
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)
