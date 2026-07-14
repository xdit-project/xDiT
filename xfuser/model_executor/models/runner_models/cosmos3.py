import json
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_cosmos3 import (
    get_cosmos3_transformer_wrapper_class,
)
from xfuser.model_executor.pipelines.pipeline_cosmos3_omni import (
    get_cosmos3_pipeline_class,
)
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    _parse_attention_backend,
)
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.utils.runner_utils import log, resize_and_crop_image

# Only full-precision attention backends produce correct results on Cosmos3.
# Quantized backends (FP8, MXFP4, MLA) cause >50% relative error per layer
# due to extreme K/V dynamic range mismatch in the MoT attention.
_COSMOS3_SUPPORTED_ATTN_BACKENDS = frozenset({
    AttentionBackendType.AITER,
    AttentionBackendType.SDPA,
    AttentionBackendType.SDPA_MATH,
    AttentionBackendType.SDPA_EFFICIENT,
    AttentionBackendType.SDPA_FLASH,
    AttentionBackendType.FLASH,
    AttentionBackendType.FLASH_3,
    AttentionBackendType.FLASH_4,
})


COSMOS3_FSDP_STRATEGY = {
    "transformer": {
        "wrap_attrs": ["layers"],
        "dtype": torch.bfloat16,
    },
}


def _setup_parallel_vae(vae, enable_parallel_encoder=True):
    if enable_parallel_encoder:
        try:
            from distvae.modules.adapters.vae.encoder_adapters import WanEncoderAdapter
            # scale_factor_spatial includes patch_size (e.g. 16 = 8x from encoder + 2x
            # from patching). The encoder adapter needs just the encoder's own
            # downsampling ratio, excluding the patching step.
            vae_scale_factor = getattr(vae.config, 'scale_factor_spatial', 8)
            patch_size = getattr(vae.config, 'patch_size', None)
            if patch_size and patch_size > 1:
                vae_scale_factor = vae_scale_factor // patch_size
            patched_encoder = WanEncoderAdapter(
                vae.encoder,
                vae_group=get_vae_parallel_group().device_group,
                vae_scale_factor=vae_scale_factor,
            ).to(vae.device)
            vae.encoder = patched_encoder
            log("Parallel VAE encoder enabled.")
        except ImportError:
            log("DistVAE not available for encoder. Defaulting to single-rank.")
        except Exception as e:
            raise ValueError(f"Failed to patch VAE encoder: {e}")
    try:
        from distvae.modules.adapters.vae.decoder_adapters import WanDecoderAdapter
        patched_decoder = WanDecoderAdapter(
            vae.decoder, vae_group=get_vae_parallel_group().device_group
        ).to(vae.device)
        # Cosmos3 VAE has patch_size=2 (extra 2x spatial upsampling from
        # unpatching). The decoder adapter's scale_factor defaults to 1 in
        # its Patchify, which is correct for Wan (patch_size=None). For
        # Cosmos3 we need to account for the extra factor so that the
        # narrow in _forward crops to the right size.
        patch_size = getattr(vae.config, 'patch_size', None)
        if patch_size and patch_size > 1:
            patched_decoder.patchify.scale_factor = patch_size
        vae.decoder = patched_decoder
        log("Parallel VAE decoder enabled.")
    except ImportError:
        log("DistVAE not available for decoder. Defaulting to single-rank.")
    except Exception as e:
        raise ValueError(f"Failed to patch VAE decoder: {e}")


@register_model("nvidia/Cosmos3-Super")
@register_model("Cosmos3-Super")
class xFuserCosmos3SuperModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_cfg_parallel=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        enable_slicing=True,
        enable_tiling=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=189,
        num_inference_steps=35,
        guidance_scale=6.0,
        flow_shift=10.0,
    )
    settings = ModelSettings(
        model_name="nvidia/Cosmos3-Super",
        output_name="cosmos3_super",
        model_output_type="video",
        fps=24,
        mod_value=16,
        fp8_gemm_module_list=["transformer.layers"],
        fp4_gemm_module_list=["transformer.layers"],
        fp8_precision_overrides=tuple(
            f"{i}." for i in list(range(10)) + list(range(54, 64))
        ),
        fsdp_strategy=COSMOS3_FSDP_STRATEGY,
    )

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        backend = _parse_attention_backend(config.attention_backend, "attention backend")
        if backend is not None and backend not in _COSMOS3_SUPPORTED_ATTN_BACKENDS:
            supported = ", ".join(sorted(b.name for b in _COSMOS3_SUPPORTED_ATTN_BACKENDS))
            raise ValueError(
                f"Cosmos3 does not support --attention_backend {backend.name}. "
                f"Quantized attention backends produce >50% relative error on this "
                f"model due to extreme K/V dynamic range mismatch in the MoT attention. "
                f"Supported backends: {supported}"
            )

    def _load_model(self) -> DiffusionPipeline:
        xFuserCosmos3OmniTransformerWrapper = get_cosmos3_transformer_wrapper_class()

        transformer = self._build_transformer(
            xFuserCosmos3OmniTransformerWrapper, stream_quant=False
        )

        xFuserCosmos3OmniPipeline = get_cosmos3_pipeline_class()

        pipe = xFuserCosmos3OmniPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            enable_safety_checker=False,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = input_args.get("prompt", "")
        negative_prompt = input_args.get("negative_prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0] if negative_prompt else ""

        # Cosmos3 expects JSON-formatted prompts for best quality.
        # If prompt is a path to a .json file, load and serialize it.
        if isinstance(prompt, str) and prompt.endswith(".json"):
            with open(prompt, "r", encoding="utf-8") as f:
                prompt = json.dumps(json.load(f))
        if isinstance(negative_prompt, str) and negative_prompt.endswith(".json"):
            with open(negative_prompt, "r", encoding="utf-8") as f:
                negative_prompt = json.dumps(json.load(f))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            fps=float(self.settings.fps),
            enable_sound=False,
            generator=self._make_generator(input_args["seed"]),
            output_type="np",
            add_resolution_template=False,
            add_duration_template=False,
        )
        if "image" in input_args and input_args["image"] is not None:
            kwargs["image"] = input_args["image"]
        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.video, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        images = input_args.get("input_images", [])
        if images:
            image = images[0]
            width, height = input_args["width"], input_args["height"]
            if input_args.get("resize_input_images", False):
                image = resize_and_crop_image(image, width, height, self.settings.mod_value)
            input_args["image"] = image
        return input_args

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        flow_shift = input_args.get("flow_shift", 10.0)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=flow_shift
        )
        log(f"Scheduler set to UniPCMultistepScheduler with flow_shift={flow_shift}")
        if self.config.fully_shard_degree > 1:
            if hasattr(self.pipe.transformer, '_patch_time_embedder_for_fsdp'):
                self.pipe.transformer._patch_time_embedder_for_fsdp()
        if self.config.use_parallel_vae:
            _setup_parallel_vae(self.pipe.vae, self.capabilities.use_parallel_vae_encoder)


@register_model("nvidia/Cosmos3-Nano")
@register_model("Cosmos3-Nano")
class xFuserCosmos3NanoModel(xFuserCosmos3SuperModel):

    settings = ModelSettings(
        model_name="nvidia/Cosmos3-Nano",
        output_name="cosmos3_nano",
        model_output_type="video",
        fps=24,
        mod_value=16,
        fp8_gemm_module_list=["transformer.layers"],
        fp4_gemm_module_list=["transformer.layers"],
        fsdp_strategy=COSMOS3_FSDP_STRATEGY,
    )
