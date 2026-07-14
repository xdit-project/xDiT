import copy
import re
import torch
from typing import List, Optional
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.utils import load_image
from safetensors.torch import load_file

from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_wan_vace import xFuserWanVACETransformer3DWrapper
from xfuser.model_executor.pipelines.pipeline_wan_i2v import (
    xFuserWanImageToVideoPipeline,
)
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser.core.utils.runner_utils import (
    log,
    resize_and_crop_image,
    resize_image_to_max_area,
)
from xfuser.envs import PACKAGES_CHECKER


COMMON_FSDP_STRATEGY = {
    "transformer": {
        "wrap_attrs": ["blocks"],
        "dtype": torch.bfloat16,
    },
    "text_encoder": {
        # CPU offload keeps shards on CPU between uses so the text encoder
        # all_gather buffer doesn't compete with sharded transformer params
        # during encode_prompt.
        "wrap_attrs": ["encoder.block"],
        "offload_policy": "cpu",
    }
}


def _build_attention_kwargs(config: "xFuserArgs") -> dict:
    """Build the per-model attention_kwargs dict used by the AITER Sparge backends. """
    return {
        "thw": None,
        "spargeattn_simthreshold": config.spargeattn_simthreshold,
        "spargeattn_cdfthreshold": config.spargeattn_cdfthreshold,
        "spargeattn_reorder_sequence": config.spargeattn_reorder_sequence,
        "use_spargeattn_static_block_mask": config.use_spargeattn_static_block_mask,
    }


def _setup_parallel_vae(vae, enable_parallel_encoder: bool = True) -> None:
    """ Parallelizes VAE en-/decoder using distvae """
    # Handle encoder
    if enable_parallel_encoder:
        try:
            from distvae.modules.adapters.vae.encoder_adapters import WanEncoderAdapter
            vae_scale_factor = getattr(vae.config, 'scaling_factor', 8)
            if hasattr(vae.config, 'vae_scale_factor_spatial'):
                vae_scale_factor = vae.config.vae_scale_factor_spatial
            patched_encoder = WanEncoderAdapter(
                vae.encoder,
                vae_group=get_vae_parallel_group().device_group,
                vae_scale_factor=vae_scale_factor,
            ).to(vae.device)
            vae.encoder = patched_encoder
            log(f"Parallel VAE encoder enabled successfully.")
        except ImportError:
            log(
                "DistVAE library is missing or does not support WanEncoderAdapter. "
                "Try installing latest DistVAE from https://github.com/xdit-project/DistVAE. "
                "Defaulting to single-rank encoder."
            )
        except Exception as e:
            raise ValueError(f"Failed to patch VAE encoder. {e}")
    # Handle decoder
    try:
        from distvae.modules.adapters.vae.decoder_adapters import WanDecoderAdapter
        patched_decoder = WanDecoderAdapter(
            vae.decoder, vae_group=get_vae_parallel_group().device_group
        ).to(vae.device)
        vae.decoder = patched_decoder
        log(f"Parallel VAE decoder enabled successfully.")
    except ImportError:
        log(
            "DistVAE library is missing or does not support WanDecoderAdapter. "
            "Try installing latest DistVAE from https://github.com/xdit-project/DistVAE. "
            "Defaulting to single-rank decoder."
        )
    except Exception as e:
        raise ValueError(f"Failed to patch VAE decoder. {e}")


def _remap_lightx2v_to_diffusers(k: str) -> str:
    """Remap a LightX2V-format state dict key to the diffusers WanTransformer3DModel naming."""
    k = re.sub(r'\.self_attn\.q\.', '.attn1.to_q.', k)
    k = re.sub(r'\.self_attn\.k\.', '.attn1.to_k.', k)
    k = re.sub(r'\.self_attn\.v\.', '.attn1.to_v.', k)
    k = re.sub(r'\.self_attn\.o\.', '.attn1.to_out.0.', k)
    k = re.sub(r'\.self_attn\.norm_q\.', '.attn1.norm_q.', k)
    k = re.sub(r'\.self_attn\.norm_k\.', '.attn1.norm_k.', k)
    k = re.sub(r'\.cross_attn\.q\.', '.attn2.to_q.', k)
    k = re.sub(r'\.cross_attn\.k\.', '.attn2.to_k.', k)
    k = re.sub(r'\.cross_attn\.v\.', '.attn2.to_v.', k)
    k = re.sub(r'\.cross_attn\.o\.', '.attn2.to_out.0.', k)
    k = re.sub(r'\.cross_attn\.norm_q\.', '.attn2.norm_q.', k)
    k = re.sub(r'\.cross_attn\.norm_k\.', '.attn2.norm_k.', k)
    k = re.sub(r'\.ffn\.0\.', '.ffn.net.0.proj.', k)
    k = re.sub(r'\.ffn\.2\.', '.ffn.net.2.', k)
    k = re.sub(r'\.norm3\.', '.norm2.', k)
    k = re.sub(r'(blocks\.\d+)\.modulation$', r'\1.scale_shift_table', k)
    k = re.sub(r'^head\.head\.', 'proj_out.', k)
    k = re.sub(r'^head\.modulation$', 'scale_shift_table', k)
    k = re.sub(r'^text_embedding\.0\.', 'condition_embedder.text_embedder.linear_1.', k)
    k = re.sub(r'^text_embedding\.2\.', 'condition_embedder.text_embedder.linear_2.', k)
    k = re.sub(r'^time_embedding\.0\.', 'condition_embedder.time_embedder.linear_1.', k)
    k = re.sub(r'^time_embedding\.2\.', 'condition_embedder.time_embedder.linear_2.', k)
    k = re.sub(r'^time_projection\.1\.', 'condition_embedder.time_proj.', k)
    return k


def _load_distilled_weights(model: torch.nn.Module, path: str) -> None:
    """Load a LightX2V distilled safetensors checkpoint into a diffusers WanTransformer3DModel."""
    raw = load_file(path, device="cpu")
    remapped = {_remap_lightx2v_to_diffusers(k): v for k, v in raw.items()}
    model.load_state_dict(remapped, strict=True)


def _distilled_scheduler_sigmas() -> torch.Tensor:
    """Sigmas (including terminal 0) for Wan2.2 I2V 4-step distilled schedule.

    LightX2V distilled Wan2.2 uses sample_shift=5 and denoising_step_list=[1000, 750, 500, 250]:
      σ_linear indices [0, 250, 500, 750] → shifted → [1.0, 0.9375, 0.8333, 0.625]
      timesteps = σ * 1000               →           [1000, 937.5, 833.3, 625.0]
    """
    sample_shift = 5.0
    sigma_linear = torch.linspace(1.0, 0.0, 1001)[:-1]
    sigma_shifted = sample_shift * sigma_linear / (1.0 + (sample_shift - 1.0) * sigma_linear)
    indices = [0, 250, 500, 750]  # 1000 - [1000, 750, 500, 250]
    return torch.cat([sigma_shifted[indices], torch.zeros(1)])


class _DistilledWanScheduler(FlowMatchEulerDiscreteScheduler):
    """FlowMatchEulerDiscreteScheduler with fixed 4-step sigmas for LightX2V Wan2.2 distilled inference."""

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        dev = torch.device(device) if device else torch.device("cpu")
        sigmas = _distilled_scheduler_sigmas().to(dev)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1] * self.config.num_train_timesteps
        self._step_index = None
        self._begin_index = None


@register_model("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
@register_model("Wan2.1-I2V")
class xFuserWan21I2VModel(xFuserModel):

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        do_cfg = input_args["guidance_scale"] > 1.0
        if do_cfg:
            return 2
        return 1

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_cfg_parallel=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        cross_attention_backend=True,
        supports_sparge_attention_backends=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
        guidance_scale_2=None,
        flow_shift=5,
        num_hybrid_attn_high_precision_steps = 5,
    )
    settings = ModelSettings(
        model_name = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        output_name = "wan2.1_i2v",
        model_output_type = "video",
        mod_value = 16, # vae_scale_factor_spatial * patch_size[1] = 8
        fps = 16,
        fp8_gemm_module_list=["transformer.blocks", "text_encoder.encoder.block"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "2.", "3.", "4.",
                                 "5.", "6.", "7.", "8.", "9.",
                                 "30.", "31.", "32.", "33.", "34.",
                                 "35.", "36.", "37.", "38.", "39."),
        fsdp_strategy=COMMON_FSDP_STRATEGY,
    )

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        if self.config.use_parallel_vae:
            _setup_parallel_vae(self.pipe.vae, self.capabilities.use_parallel_vae_encoder)
        self.pipe.scheduler.config.flow_shift = input_args["flow_shift"]

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                quantization_config=te_quant,
                **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            image=input_args["image"],
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            guidance_scale_2=input_args["guidance_scale_2"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, width, height, self.settings.mod_value)
        else:
            image = resize_image_to_max_area(image, height, width, self.settings.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Wan I2V model.")

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        # Full warmup when a per-step attention or GEMM schedule is active,
        # to trigger all backend paths; reduce to 2 steps otherwise.
        if not get_runtime_state().has_attention_schedule() and not get_runtime_state().has_gemm_schedule():
            return 2
        return None


@register_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
@register_model("Wan2.2-I2V")
class xFuserWan22I2VModel(xFuserWan21I2VModel):

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        self.settings.model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_i2v"
        self.settings.fsdp_strategy["transformer_2"] = {
                "wrap_attrs": ["blocks"],
                "dtype": torch.bfloat16,
        }
        self.settings.fp8_gemm_module_list = ["transformer.blocks", "transformer_2.blocks", "text_encoder.encoder.block"]
        self.settings.fp8_precision_overrides = None


    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
                quantization_config=te_quant,
                **te_kwargs,
        )
        return pipe

    def _get_compiled_pipe_components(self):
        return ["transformer", "transformer_2"]

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        return None  # full warmup cycle


@register_model("Wan2.2-Distilled-I2V")
class xFuserWan22DistilledI2VModel(xFuserWan22I2VModel):
    """Wan2.2 I2V with LightX2V 4-step distilled weights.

    Loads the base diffusers architecture from Wan-AI/Wan2.2-I2V-A14B-Diffusers and
    replaces the transformer weights from LightX2V BF16 distilled checkpoints.

    Required extra args (passed via xFuserArgs or runner config):
        distilled_transformer_path:   path to high-noise .safetensors (transformer)
        distilled_transformer_2_path: path to low-noise .safetensors (transformer_2)
    """

    # LightX2V boundary_step_index=2 (step-index comparison) maps to a timestep threshold
    # between shifted t[1]≈937 and t[2]≈833. 0.9 → threshold 900 correctly splits 2+2.
    _BOUNDARY_RATIO = 0.9
    _BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_cfg_parallel=False,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        cross_attention_backend=True,
        supports_sparge_attention_backends=True,
        supports_distilled_weights=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=81,
        num_inference_steps=4,
        guidance_scale=1.0,
        guidance_scale_2=None,
        flow_shift=5,
        num_hybrid_attn_high_precision_steps=1,
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        self.settings.model_name = self._BASE_MODEL
        self.settings.output_name = "wan2.2_distilled_i2v"

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self._BASE_MODEL,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
            low_cpu_mem_usage=True,
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self._BASE_MODEL,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
            attention_kwargs=_build_attention_kwargs(self.config),
            low_cpu_mem_usage=True,
        )
        _load_distilled_weights(transformer,   self.config.distilled_transformer_path)
        _load_distilled_weights(transformer_2, self.config.distilled_transformer_2_path)
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self._BASE_MODEL,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=self._BOUNDARY_RATIO,
        )
        pipe.scheduler = _DistilledWanScheduler.from_config(pipe.scheduler.config)
        return pipe

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if not self.config.distilled_transformer_path:
            raise ValueError(
                "--distilled_transformer_path is required for Wan2.2-Distilled-I2V "
                "(path to high-noise safetensors file)."
            )
        if not self.config.distilled_transformer_2_path:
            raise ValueError(
                "--distilled_transformer_2_path is required for Wan2.2-Distilled-I2V "
                "(path to low-noise safetensors file)."
            )
        steps = input_args.get("num_inference_steps")
        if steps != 4:
            raise ValueError(
                f"Wan2.2-Distilled-I2V uses a fixed 4-step schedule; "
                f"num_inference_steps must be 4, got {steps}."
            )
        guidance_scale = input_args.get("guidance_scale")
        if guidance_scale != 1.0:
            log(f"Using guidance_scale=1.0. Other guindance scale values are not supported with this model.")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        # Guidance is baked into the distilled weights. guidance_scale=1.0 keeps
        # do_classifier_free_guidance=False, so negative_prompt has no effect.
        output = self.pipe(
            image=input_args["image"],
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=1.0,
            guidance_scale_2=None,
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)


@register_model("Wan-AI/Wan2.1-T2V-14B-Diffusers")
@register_model("Wan2.1-T2V")
class xFuserWan21T2VModel(xFuserModel):

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        do_cfg = input_args["guidance_scale"] > 1.0
        if do_cfg:
            return 2
        return 1

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        use_parallel_vae=True,
        cross_attention_backend=True,
        supports_sparge_attention_backends=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
        guidance_scale_2=None,
        flow_shift=12,
        num_hybrid_attn_high_precision_steps = 5,
    )
    settings = ModelSettings(
        mod_value=8,
        fps=16,
        model_output_type="video",
        model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        output_name="wan2.1_t2v",
        fp8_gemm_module_list=["transformer.blocks", "text_encoder.encoder.block"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "2.", "3.", "4.",
                                 "5.", "6.", "7.", "8.", "9.",
                                 "30.", "31.", "32.", "33.", "34.",
                                 "35.", "36.", "37.", "38.", "39."),
        fsdp_strategy=COMMON_FSDP_STRATEGY,
    )

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        if self.config.use_parallel_vae:
            _setup_parallel_vae(self.pipe.vae, self.capabilities.use_parallel_vae_encoder)
        self.pipe.scheduler.config.flow_shift = input_args["flow_shift"]

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=te_quant,
            **te_kwargs,
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
            guidance_scale_2=input_args["guidance_scale_2"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        if not get_runtime_state().has_attention_schedule() and not get_runtime_state().has_gemm_schedule():
            return 2
        return None


@register_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
@register_model("Wan2.2-T2V")
class xFuserWan22T2VModel(xFuserWan21T2VModel):

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        self.settings.model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_t2v"
        self.settings.fsdp_strategy["transformer_2"] = {
                "wrap_attrs": ["blocks"],
                "dtype": torch.bfloat16,
        }
        self.settings.fp8_gemm_module_list=["transformer.blocks", "transformer_2.blocks", "text_encoder.encoder.block"]
        self.settings.fp8_precision_overrides=None

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            transformer_2=transformer_2,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _get_compiled_pipe_components(self):
        return ["transformer", "transformer_2"]

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        return None  # full warmup cycle


@register_model("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
@register_model("Wan2.2-TI2V")
class xFuserWan22TI2VModel(xFuserWan21T2VModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        use_hybrid_gemm_schedule=True,
        use_parallel_vae=True,
        cross_attention_backend=True,
        supports_sparge_attention_backends=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=736,
        width=1280,
        num_inference_steps=50,
        num_frames=121,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=5.0,
        guidance_scale_2=None,
        flow_shift=5,
        num_hybrid_attn_high_precision_steps=5,
        num_hybrid_gemm_high_precision_steps=5,
    )
    settings = ModelSettings(
        mod_value=32,
        fps=24,
        model_output_type="video",
        model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        output_name="wan2.2_ti2v",
        fp8_gemm_module_list=["transformer.blocks", "text_encoder.encoder.block"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "28.", "29."),
        fp8_precision_override_suffixes=(".net.0.proj", ".net.2"),
        fsdp_strategy=COMMON_FSDP_STRATEGY,
        valid_tasks=["i2v", "t2v"],
    )

    def _load_model(self) -> DiffusionPipeline:
        torch.set_float32_matmul_precision('high')
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            attention_kwargs=_build_attention_kwargs(self.config),
        )
        pipe_class = xFuserWanImageToVideoPipeline if self.config.task == "i2v" else WanPipeline
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = pipe_class.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                quantization_config=te_quant,
                **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "height": input_args["height"],
            "width": input_args["width"],
            "prompt": input_args["prompt"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "num_frames": input_args["num_frames"],
            "guidance_scale": input_args["guidance_scale"],
            "guidance_scale_2": input_args["guidance_scale_2"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        return None  # full warmup cycle

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task != "i2v":
            return input_args

        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, width, height, self.settings.mod_value)
        else:
            image = resize_image_to_max_area(image, height, width, self.settings.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if self.config.task == "i2v":
            if len(images) != 1:
                raise ValueError("Exactly one input image is required for Wan TI2V model when using i2v task.")
        else:
            if len(images) != 0:
                raise ValueError("No input images should be provided for Wan TI2V model when using t2v task.")



@register_model("Wan-AI/Wan2.1-VACE-14B-diffusers")
@register_model("Wan-AI/Wan2.1-VACE-1.3B-diffusers")
@register_model("Wan2.1-VACE-14B")
@register_model("Wan2.1-VACE-1.3B")
class xFuserWan21VACEModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
        cross_attention_backend=True,
        enable_tiling=True,
        enable_slicing=True,
        fully_shard_degree=True,
    )

    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=30,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=5.0,
    )

    settings = ModelSettings(
        fps=16,
        model_output_type="video",
        fp8_gemm_module_list=["transformer.blocks", "transformer.vace_blocks", "text_encoder.encoder.block"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["blocks", "vace_blocks"],
                "dtype": torch.bfloat16,
            },
            "text_encoder": {
                "wrap_attrs": ["encoder.block"],
            },
        },
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        if "14B" in config.model:
            self.settings.model_name = "Wan-AI/Wan2.1-VACE-14B-diffusers"
            self.settings.output_name = "wan.2.1_vace_14b"
        else:
            self.settings.model_name = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
            self.settings.output_name = "wan.2.1_vace_1.3b"

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanVACETransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        te_kwargs, te_quant = self._meta_te_kwargs()
        pipe = WanVACEPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=te_quant,
            **te_kwargs,
        )
        pipe.scheduler.flow_shift = 5.0 # 5.0 for 720p, 3.0 for 480p
        return pipe

    def _prepare_video_and_mask(self, first_img: Image, last_img: Image, height: int, width: int, num_frames: int) -> tuple[List[Image.Image], List[Image.Image]]:
        """ Prepare video and mask for Wan VACE model """
        first_img = first_img.resize((width, height))
        last_img = last_img.resize((width, height))
        frames = []
        frames.append(first_img)
        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        # match the original code.
        frames.extend([Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
        frames.append(last_img)
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)
        mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
        return frames, mask

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        self._validate_args(input_args)
        images = [load_image(path) for path in input_args.get("input_images", [])]
        video, mask = self._prepare_video_and_mask(images[0], images[1], input_args["height"], input_args["width"], input_args["num_frames"])
        input_args["video"] = video
        input_args["mask"] = mask
        return input_args

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
            video=input_args["video"],
            mask=input_args["mask"],
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 2:
            raise ValueError("Exactly two input images are required for Wan VACE model (first frame and last frame).")
