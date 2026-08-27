import functools

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.model_executor.cache import (
    DBCachePreset,
    CacheDitAdapterConfig,
    DBCacheSettings,
)
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)


def _normalize_prompt(prompt_input):
    if isinstance(prompt_input, str):
        return [prompt_input]
    if isinstance(prompt_input, list):
        return list(prompt_input) # Recreates the list to avoid issues with in-place editing
    raise TypeError(f"prompt must be str or list[str], got {type(prompt_input)}")


def _keep_timesteps_host_resident(pipe) -> None:
    """Elide the per-denoise-step device->host sync in ``ZImagePipeline.__call__``.

    ``pipeline_z_image.py`` computes ``t_norm = ((1000 - t.expand(B)) / 1000)[0].item()``
    at the top of *every* denoise step.  ``retrieve_timesteps`` hands the scheduler the
    execution device, so ``scheduler.timesteps`` lives on the GPU and that ``.item()``
    is a genuine blocking D2H sync which drains the queue of the previous step's
    transformer work, converting the whole loop from pipelined to lock-step.

    The timestep schedule is a tiny 1-D vector of *host-known* constants; nothing in
    ``FlowMatchEulerDiscreteScheduler.step`` needs it on the device (``self.sigmas``
    stays device-resident and ``_init_step_index`` explicitly moves the incoming
    timestep onto ``self.timesteps.device``).  So we keep ``scheduler.timesteps`` on
    the host: the ``.item()`` becomes free, and the transformer wrapper performs a
    single cheap non-blocking H2D copy of the (B,) timestep vector instead.

    Values are bit-identical -- ``(1000 - t) / 1000`` is one IEEE-754 fp32 op either
    way -- and the whole thing is selected from tensor properties with a hard
    fallback: if the scheduler does not expose a tensor ``timesteps`` we leave it
    exactly as it was.
    """
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None or getattr(scheduler, "_xfuser_host_timesteps", False):
        return
    original_set_timesteps = getattr(scheduler, "set_timesteps", None)
    if not callable(original_set_timesteps):
        return

    # ``functools.wraps`` is load-bearing, not cosmetic: ``retrieve_timesteps``
    # feature-detects scheduler support with
    # ``"sigmas" in inspect.signature(scheduler.set_timesteps).parameters``.
    # A bare ``(*args, **kwargs)`` wrapper reports no such parameter and the
    # pipeline raises "does not support custom sigmas schedules".  ``wraps``
    # sets ``__wrapped__``, which ``inspect.signature`` follows back to the
    # scheduler's real signature, so the probe keeps seeing the truth.
    @functools.wraps(original_set_timesteps)
    def set_timesteps(*args, **kwargs):
        out = original_set_timesteps(*args, **kwargs)
        ts = getattr(scheduler, "timesteps", None)
        # Fallback path: anything we do not recognise is left untouched.
        if isinstance(ts, torch.Tensor) and ts.device.type != "cpu" and ts.dim() == 1:
            scheduler.timesteps = ts.to("cpu")
        return out

    scheduler.set_timesteps = set_timesteps
    scheduler._xfuser_host_timesteps = True


def _set_effective_heads_for_ulysses(transformer, ulysses_degree: int) -> None:
    """Expose a Ulysses-divisible head count for runtime validation.

    Keep the real model head layout untouched (e.g., n_heads=30) and only set
    config.num_attention_heads used by runtime pre-checks.
    """
    ulysses_degree = int(ulysses_degree or 1)
    if ulysses_degree <= 1:
        return

    real_heads = getattr(transformer.config, "n_heads", None)
    if not isinstance(real_heads, int):
        real_heads = getattr(transformer.config, "num_attention_heads", None)
    if not isinstance(real_heads, int):
        return

    effective_heads = ((real_heads + ulysses_degree - 1) // ulysses_degree) * ulysses_degree
    if effective_heads == real_heads:
        return

    transformer.config.num_attention_heads = effective_heads

@register_model("Tongyi-MAI/Z-Image")
@register_model("Z-Image")
class xFuserZImageModel(xFuserModel):
    min_diffusers_version = "0.36.0"

    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
    )
    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp8_text_encoder=True,
        use_int8_gemms=True,
        supports_step_caching=True,
        use_parallel_vae=True,
    )
    settings = ModelSettings(
        model_name="Tongyi-MAI/Z-Image",
        output_name="z_image",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["noise_refiner", "context_refiner", "layers"],
            },
            "text_encoder": {
                "wrap_attrs": ["layers"],
            },
        },
        fp8_gemm_module_list=["transformer.layers", "transformer.noise_refiner", "transformer.context_refiner"],
        fp8_text_encoder_module_list=["text_encoder.layers"],
        int8_gemm_module_list=[
            "transformer.layers",
            "transformer.noise_refiner",
            "transformer.context_refiner"
        ],
        step_cache_config={
            "dbcache":DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("layers", "Pattern_3"),),
                ),
                preset=DBCachePreset(Fn_compute_blocks=3, residual_diff_threshold=0.12, scm_policy="ultra"),
            ),
        },
    )

    def _customize_settings(self, config) -> None:
        """Exclude context_refiner from INT8 quant when sequence parallelism is active.

        Both Ulysses and Ring attention split the sequence across GPUs.  The
        caption features processed by ``context_refiner`` may be very short; 
        after SP chunking each GPU may see M <= 16, which is below the minimum 
        M required by the ``torch._int_mm`` kernel used by torch.compile.
        """
        sp_world_size = (config.ulysses_degree or 1) * (config.ring_degree or 1)
        if sp_world_size > 1 and config.use_int8_gemms:
            self.settings.int8_gemm_module_list = [
                m for m in self.settings.int8_gemm_module_list
                if m != "transformer.context_refiner"
            ]

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import ZImagePipeline
        from xfuser.model_executor.models.transformers.transformer_z_image import (
            xFuserZImageTransformer2DWrapper,
        )

        transformer = self.loader.load_transformer(xFuserZImageTransformer2DWrapper)
        _set_effective_heads_for_ulysses(transformer, self.config.ulysses_degree)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        _keep_timesteps_host_resident(pipe)
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = _normalize_prompt(input_args["prompt"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=self._make_generator(input_args["seed"]),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)


@register_model("Tongyi-MAI/Z-Image-Turbo")
@register_model("Z-Image-Turbo")
class xFuserZImageTurboModel(xFuserModel):
    min_diffusers_version = "0.36.0"

    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        enable_tiling=True,
        enable_slicing=True,
        use_fp8_gemms=True,
        use_fp8_text_encoder=True,
        use_int8_gemms=True,
        fully_shard_degree=True,
        use_parallel_vae=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        model_name="Tongyi-MAI/Z-Image-Turbo",
        output_name="z_image_turbo",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["noise_refiner", "context_refiner", "layers"],
            },
            "text_encoder": {
                "wrap_attrs": ["layers"],
            },
        },
        fp8_gemm_module_list=["transformer.layers", "transformer.noise_refiner", "transformer.context_refiner"],
        fp8_text_encoder_module_list=["text_encoder.layers"],
        int8_gemm_module_list=["transformer.layers", "transformer.noise_refiner", "transformer.context_refiner"],
    )

    def _customize_settings(self, config) -> None:
        """Exclude context_refiner from INT8 quant when sequence parallelism is active.

        Both Ulysses and Ring attention split the sequence across GPUs.  The
        caption features processed by ``context_refiner`` may be very short; 
        after SP chunking each GPU may see M <= 16, which is below the minimum 
        M required by the ``torch._int_mm`` kernel used by torch.compile.
        """
        sp_world_size = (config.ulysses_degree or 1) * (config.ring_degree or 1)
        if sp_world_size > 1 and config.use_int8_gemms:
            self.settings.int8_gemm_module_list = [
                m for m in self.settings.int8_gemm_module_list
                if m != "transformer.context_refiner"
            ]

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import ZImagePipeline
        from xfuser.model_executor.models.transformers.transformer_z_image import (
            xFuserZImageTransformer2DWrapper,
        )

        transformer = self.loader.load_transformer(xFuserZImageTransformer2DWrapper)
        _set_effective_heads_for_ulysses(transformer, self.config.ulysses_degree)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        _keep_timesteps_host_resident(pipe)
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = _normalize_prompt(input_args["prompt"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=self._make_generator(input_args["seed"]),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)
