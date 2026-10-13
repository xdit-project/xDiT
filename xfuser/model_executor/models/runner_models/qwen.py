import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.model_executor.cache import (
    DBCachePreset,
    CacheDitAdapterConfig,
    DBCacheSettings,
)
from xfuser.model_executor.models.runner_models.base_model import (
    register_model,
    xFuserModel,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser import xFuserArgs
from xfuser.core.distributed import get_runtime_state
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)


@register_model("Qwen/Qwen-Image-Edit-2511")
@register_model("Qwen/Qwen-Image-Edit-2509")
@register_model("Qwen/Qwen-Image-Edit")
@register_model("Qwen-Image-Edit-2511")
@register_model("Qwen-Image-Edit-2509")
@register_model("Qwen-Image-Edit")
class xFuserQwenImageEditModel(xFuserModel):
    # From the registered checkpoint's transformer config.
    attention_heads = 24

    min_diffusers_version = "0.37.0"

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        meta_text_encoders=("text_encoder",),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp8_text_encoder=True,
        use_fp8_comms=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        enable_tiling=True,
        enable_slicing=True,
        supports_step_caching=True,
    )
    default_input_values = DefaultInputValues(
        num_inference_steps=50,
        guidance_scale=4.0,
        negative_prompt=" ",
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image-Edit",
        output_name="qwen_image_edit",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        step_cache_config={
            "dbcache": DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("transformer_blocks", "Pattern_1"),),
                    enable_separate_cfg=True,
                ),
                preset=DBCachePreset(Fn_compute_blocks=6, residual_diff_threshold=0.12, scm_policy="ultra"),
            ),
        },
        fp8_text_encoder_module_list=["text_encoder.model.language_model.layers"],
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        # 2509 and 2511 are Edit Plus checkpoints: diffusers loads them with
        # QwenImageEditPlusPipeline, which conditions on a list of images.
        self._is_edit_plus = False
        if "2511" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2511"
            self.settings.output_name = "qwen_image_edit_2511"
            self._is_edit_plus = True
        elif "2509" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2509"
            self.settings.output_name = "qwen_image_edit_2509"
            self._is_edit_plus = True

    def _load_model(self) -> DiffusionPipeline:
        from xfuser.model_executor.pipelines.pipeline_qwen_image_edit import (
            xFuserQwenImageEditPipeline,
            xFuserQwenImageEditPlusPipeline,
        )
        from xfuser.model_executor.models.transformers.transformer_qwen import (
            xFuserQwenImageTransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImageTransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipeline_cls = xFuserQwenImageEditPlusPipeline if self._is_edit_plus else xFuserQwenImageEditPipeline
        if pipeline_cls is None:
            raise ImportError(f"{self.settings.model_name} needs a diffusers release with QwenImageEditPlusPipeline.")
        pipe = pipeline_cls.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        images = input_args["input_images"]
        kwargs = {
            "image": images if self._is_edit_plus else images[0],
            "prompt": input_args["prompt"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": self._make_generator(input_args["seed"]),
        }
        if "height" in input_args:
            kwargs["height"] = input_args["height"]
        if "width" in input_args:
            kwargs["width"] = input_args["width"]

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)

    def _validate_args(self, input_args: dict) -> None:
        """Validate input arguments"""
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if self._is_edit_plus:
            if not images:
                raise ValueError(f"At least one input image is required for {self.settings.model_name}.")
        elif len(images) != 1:
            raise ValueError("Exactly one input image is required for Qwen Image Edit model.")


@register_model("Qwen/Qwen-Image-2512")
@register_model("Qwen/Qwen-Image")
@register_model("Qwen-Image-2512")
@register_model("Qwen-Image")
class xFuserQwenImageModel(xFuserModel):
    # From the registered checkpoint's transformer config.
    attention_heads = 24

    min_diffusers_version = "0.37.0"

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        meta_text_encoders=("text_encoder",),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        supports_step_caching=True,
        use_fp8_text_encoder=True,
        use_fp8_comms=True,
        use_parallel_vae=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=928,
        width=1664,
        num_inference_steps=50,
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image",
        output_name="qwen_image",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp8_text_encoder_module_list=["text_encoder.model.language_model.layers"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
        step_cache_config={
            "dbcache": DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("transformer_blocks", "Pattern_1"),),
                    enable_separate_cfg=False,
                ),
                preset=DBCachePreset(Fn_compute_blocks=6, residual_diff_threshold=0.12, scm_policy="ultra"),
            )
        },
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        if "2512" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-2512"
            self.settings.output_name = "qwen_image_2512"

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import QwenImagePipeline
        from xfuser.model_executor.models.transformers.transformer_qwen import (
            xFuserQwenImageTransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImageTransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "prompt": input_args["prompt"],
            "height": input_args["height"],
            "width": input_args["width"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": self._make_generator(input_args["seed"]),
        }

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)


@register_model("Qwen/Qwen-Image-2.1")
@register_model("Qwen-Image-2.1")
class xFuserQwenImage21Model(xFuserModel):
    """Qwen-Image 2.1 T2I / image-conditioned runner.

    Single-stream block-causal transformer with a prefix KV cache; needs diffusers >= 0.41.0.
    ``--input_images`` are condition images, and without ``--height``/``--width`` the output
    follows the last one's aspect ratio. No Ring: it would split the block-causal mask.
    """

    # From the registered checkpoint's transformer config.
    attention_heads = 32

    min_diffusers_version = "0.41.0"
    attention_head_dims = frozenset({128})

    load_support = LoadSupport(
        meta_transformers=("transformer",),
        meta_text_encoders=("text_encoder",),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    # No height/width: _preprocess_args_images resolves them, from the condition image when given.
    default_input_values = DefaultInputValues(
        num_inference_steps=40,
        # true_cfg_scale; 2.1 is meant to be sampled without guidance (cfg off at 1.0).
        # No default negative_prompt: with CFG off it would be ignored, and passing it
        # only makes diffusers warn that it has no effect.
        guidance_scale=1.0,
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image-2.1",
        output_name="qwen_image_21",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
    )

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import QwenImage21Pipeline
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            xFuserQwenImage21TransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImage21TransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = QwenImage21Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    @staticmethod
    def _qwen_image21_output_size(images, height, width, resolution=1024):
        """Output (height, width): explicit when given, else the last condition image's aspect
        ratio at ``resolution**2`` area -- what QwenImage21Pipeline picks itself -- else square.

        The ``calculate_dimensions`` import stays inside this class. A module-level helper
        would make the floor audit treat that unreleased symbol as a requirement of every
        model in this file, including the released Qwen-Image pipelines.
        """
        if height is not None and width is not None:
            return height, width
        if height is not None or width is not None:
            raise ValueError("Qwen-Image-2.1 needs both --height and --width, or neither.")
        if images:
            from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

            image_width, image_height = images[-1].size
            width, height, _ = calculate_dimensions(resolution * resolution, image_width / image_height)
            return height, width
        return resolution, resolution

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        explicit = input_args.get("height") is not None and input_args.get("width") is not None
        height, width = self._qwen_image21_output_size(
            input_args["input_images"], input_args.get("height"), input_args.get("width")
        )
        input_args["height"], input_args["width"] = height, width
        if not explicit:
            source = "last input image's aspect ratio" if input_args["input_images"] else "model default"
            log(f"Qwen-Image-2.1: output size {width}x{height} (from {source}).")
        return input_args

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            xFuserQwenImage21AttnProcessor,
        )

        blocks = self.pipe.transformer.transformer_blocks
        for block in blocks:
            block.attn.set_processor(xFuserQwenImage21AttnProcessor())
        log(f"Qwen-Image-2.1: decode attention uses {get_runtime_state().attention_backend.name}.")
        if not self.config.use_torch_compile:
            log("Qwen-Image-2.1: block-causal prefill uses per-segment SDPA.")

    def _prefer_blockwise_compile(self) -> bool:
        return True

    def _compile_model(self, input_args: dict) -> None:
        # flex_attention is only efficient compiled, so the one-call prefill is set up here, not at load.
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            flex_attention_available,
            xFuserQwenImage21FlexAttnProcessor,
        )

        if flex_attention_available():
            blocks = self.pipe.transformer.transformer_blocks
            for block in blocks:
                block.attn.set_processor(xFuserQwenImage21FlexAttnProcessor())
            log(f"Qwen-Image-2.1: block-causal prefill uses flex_attention on {len(blocks)} blocks.")
        else:
            log("Qwen-Image-2.1: flex_attention unavailable; block-causal prefill uses per-segment SDPA.")
        super()._compile_model(input_args)

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        # Passed through on purpose. The pipeline enables classifier-free guidance
        # only when this is greater than 1 and a negative prompt is set; the default
        # of 1.0 is the unguided sampling 2.1 is meant to use.
        true_cfg_scale = input_args["guidance_scale"]
        kwargs = {
            "prompt": input_args["prompt"],
            "image": input_args["input_images"] or None,
            "height": input_args["height"],
            "width": input_args["width"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": true_cfg_scale,
            "generator": self._make_generator(input_args["seed"]),
        }
        negative_prompt = input_args.get("negative_prompt")
        if true_cfg_scale > 1 and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)
