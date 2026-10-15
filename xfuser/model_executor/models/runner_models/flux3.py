import functools
from types import SimpleNamespace

import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.utils.runner_utils import log, resize_and_crop_image
from xfuser.model_executor.models.runner_models.base_model import (
    DIFFUSERS_FROM_SOURCE,
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)

# Generation size of text-to-image runs; with reference images the pipeline keeps this
# area at the first reference's aspect ratio.
_DEFAULT_SIZE = 1024
_TRANSFORMER_BLOCKS = [
    "transformer.txt_mode_blocks",
    "transformer.mode_blocks",
    "transformer.single_transformer_blocks",
]


@register_model("black-forest-labs/flux-3-image-klein")
@register_model("flux-3-image-klein")
@register_model("FLUX.3-image-klein")
class xFuserFlux3KleinModel(xFuserModel):
    """FLUX.3 image klein: step-distilled (flow-map) text-to-image and image editing.

    16 steps without classifier-free guidance. ``--input_images`` are reference images
    (up to 10); without ``--height``/``--width`` an edit keeps the first reference's
    aspect ratio at about 1024x1024 pixels. One prompt per pipeline call: a prompt
    list runs one call per prompt.
    """

    # From the registered checkpoint's transformer config.
    attention_heads = 24

    # Flux3Transformer2DModel and the FLUX.3 modular pipeline are in diffusers main
    # (huggingface/diffusers#15046) but no release yet.
    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        # The modular pipeline loads the text encoder itself.
        meta_text_encoders=(),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=False,
        use_fp8_gemms=True,
        use_fp8_comms=True,
        fully_shard_degree=True,
        enable_tiling=True,
        enable_slicing=True,
        use_parallel_vae=True,
    )
    default_input_values = DefaultInputValues(num_inference_steps=16)
    settings = ModelSettings(
        model_name="black-forest-labs/flux-3-image-klein",
        output_name="flux_3_image_klein",
        model_output_type="image",
        mod_value=16,
        fp8_gemm_module_list=_TRANSFORMER_BLOCKS,
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": [
                    "txt_mode_blocks",
                    "mode_blocks.image",
                    "mode_blocks.image_cond",
                    "single_transformer_blocks",
                ],
                # Setting the dtype keeps the fp32 QK norms out of the FSDP units.
                "dtype": torch.bfloat16,
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
    )

    def _validate_config(self, config: xFuserArgs) -> None:
        super()._validate_config(config)
        if config.batch_size is not None and config.batch_size > 1:
            raise ValueError(
                "FLUX.3 encodes one prompt per pipeline call, so --batch_size must be 1; "
                "prompt lists run one call per prompt."
            )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if input_args.get("negative_prompt"):
            raise ValueError("FLUX.3 klein is guidance-distilled and does not accept a negative prompt.")
        height, width = input_args.get("height"), input_args.get("width")
        for name, value in (("height", height), ("width", width)):
            if value is not None and value % self.settings.mod_value != 0:
                raise ValueError(f"FLUX.3 needs --{name} divisible by {self.settings.mod_value}, got {value}.")
        if input_args.get("guidance_scale") is not None:
            log("FLUX.3 klein is guidance-distilled; ignoring --guidance_scale.")

    def _load_model(self):
        from diffusers import ModularPipeline
        from xfuser.model_executor.models.transformers.transformer_flux3 import (
            xFuserFlux3Transformer2DWrapper,
        )

        log(f"Loading {self.settings.model_name}")
        transformer = self.loader.load_transformer(xFuserFlux3Transformer2DWrapper)
        pipe = ModularPipeline.from_pretrained(self.settings.model_name)
        pipe.update_components(transformer=transformer)
        pipe.load_components(dtype=torch.bfloat16)
        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self._expose_transformer_signature()

    def _expose_transformer_signature(self) -> None:
        """Show the denoise step the transformer's real forward parameters.

        The step passes the transformer only the fields its forward signature names.
        FSDP1 wraps the transformer in a module whose forward takes (*args, **kwargs),
        which would drop img_ids and txt_ids.
        """
        transformer = self.pipe.transformer
        inner = getattr(transformer, "module", None)
        if not isinstance(inner, torch.nn.Module):
            return
        outer_forward = transformer.forward

        @functools.wraps(inner.forward)
        def forward(*args, **kwargs):
            return outer_forward(*args, **kwargs)

        transformer.forward = forward

    def _prefer_blockwise_compile(self) -> bool:
        # Compiling the blocks keeps the transformer object, and with it the forward
        # signature the denoise step reads, and leaves the SP shard and gather glue eager.
        return True

    def _get_runtime_state_pipeline(self):
        # The transformer takes tokens the pipeline already packed 2x2, so it has no
        # patch_size; runtime state reads one, and the 16-pixel token grid.
        config = self.pipe.transformer.config
        return SimpleNamespace(
            transformer=SimpleNamespace(
                config=SimpleNamespace(
                    patch_size=1,
                    in_channels=config.in_channels,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                )
            ),
            vae_scale_factor=self.pipe.vae_scale_factor * 2,
        )

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        images = input_args["input_images"]
        if not images:
            input_args["height"] = input_args.get("height") or _DEFAULT_SIZE
            input_args["width"] = input_args.get("width") or _DEFAULT_SIZE
        elif input_args.get("resize_input_images", False):
            if input_args.get("height") is None or input_args.get("width") is None:
                raise ValueError("--resize_input_images needs --height and --width.")
            input_args["input_images"] = [
                resize_and_crop_image(
                    image,
                    target_height=input_args["height"],
                    target_width=input_args["width"],
                    mod_value=self.settings.mod_value,
                )
                for image in images
            ]
        return input_args

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompts = input_args["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        pipe_kwargs = {
            "height": input_args.get("height"),
            "width": input_args.get("width"),
            "num_inference_steps": input_args["num_inference_steps"],
            "output": "images",
        }
        if input_args.get("input_images"):
            pipe_kwargs["image"] = input_args["input_images"]
        if input_args.get("max_sequence_length") is not None:
            pipe_kwargs["max_sequence_length"] = input_args["max_sequence_length"]

        images, args_per_image = [], []
        for prompt in prompts:
            # The pipeline draws the noise in fp32 from a CPU generator, as in the model
            # card, so a seed gives the same image as the documented diffusers usage.
            image = self.pipe(
                prompt=prompt,
                generator=torch.Generator().manual_seed(input_args["seed"]),
                **pipe_kwargs,
            )[0]
            images.append(image)
            # Edits size the output from the reference unless told otherwise.
            args_per_image.append({**input_args, "prompt": prompt, "height": image.height, "width": image.width})
        return DiffusionOutput(images=images, pipe_args=args_per_image)
