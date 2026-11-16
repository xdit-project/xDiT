"""Runner for OpenMOSS-Team/MOSS-Image-Base.

The serving recipe matches the model card: 1024², 50 steps, flow shift 5,
CFG 3, and unconditional block 22 skipped from the first step
(``slg_start=0.0``). Weights are BF16 for the DiT, the Qwen3.5 encoder, and the
FLUX.2 VAE convolutions; the VAE's latent BatchNorm stays FP32. Image editing
is not implemented.

Sequence parallelism shards the joint DiT sequence. Ulysses degree must divide
the 24 heads. The text refiner stays replicated. ``--use_cfg_parallel`` runs
the positive and negative passes on two ranks.
"""

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.contracts import LoadRoute, LoadSupport


@register_model("OpenMOSS-Team/MOSS-Image-Base")
@register_model("MOSS-Image-Base")
@register_model("MOSS-Image")
class xFuserMOSSImageModel(xFuserModel):
    attention_heads = 24
    attention_head_dims = frozenset({128})
    min_diffusers_version = "0.38.0"

    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        pipefusion_parallel_degree=False,
        tensor_parallel_degree=False,
        use_cfg_parallel=True,
        use_parallel_vae=True,
        enable_tiling=True,
        enable_slicing=True,
        fully_shard_degree=False,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        flow_shift=5.0,
        max_sequence_length=4096,
    )
    settings = ModelSettings(
        model_name="OpenMOSS-Team/MOSS-Image-Base",
        output_name="moss_image",
        model_output_type="image",
        resolution_divisor=16,
    )

    def _get_compile_mode(self) -> str:
        # CUDA graphs are incompatible with cross-step caching and slow on RDNA4.
        if self.config.cache_method or PACKAGES_CHECKER._on_rdna4():
            return "default"
        return "reduce-overhead"

    def _load_model(self) -> DiffusionPipeline:
        from xfuser.model_executor.pipelines.pipeline_moss_image import xFuserMOSSImagePipeline

        return xFuserMOSSImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            # BF16 decode is ~3x faster than FP32 on MI355 (MIOpen FP32 convs are slow) at ~56 dB PSNR.
            vae_dtype=torch.bfloat16,
            # With CFG parallel each rank runs one sequence and the separate v GEMM overlaps the
            # q/k all-to-all; the serial CFG pair shares one exchange, so fusing all three wins.
            fuse_projections=("q", "k") if self.config.use_cfg_parallel else ("q", "k", "v"),
            cache_dir=self.config.download_dir,
        )

    def _get_compiled_pipe_components(self):
        # The eager Qwen3.5 encoder is launch bound (~7.8k kernels for prompt + negative);
        # compiled it encodes ~3x faster. Prompts are re-encoded on every call, inside the timed region.
        return ["transformer", "text_encoder"]

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        negative = input_args.get("negative_prompt")
        output = self.pipe(
            prompt=input_args["prompt"],
            negative_prompt="" if negative is None else negative,
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            shift=input_args.get("flow_shift"),
            generator=self._make_generator(input_args["seed"]),
            # Model-card recipe: skip block 22 on the negative branch from step 0.
            uncond_skip_layers=(22,),
            slg_start=0.0,
            slg_stop=1.0,
        )
        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)
