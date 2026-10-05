import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadRoute,
    LoadSupport,
)


@register_model("Alpha-VLLM/Lumina-Image-2.0")
@register_model("Lumina-Image-2.0")
@register_model("Lumina2")
class xFuserLumina2Model(xFuserModel):
    # The composition-style pipeline wrapper loads the transformer eagerly.
    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        pipefusion_parallel_degree=False,
        tensor_parallel_degree=False,
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
        max_sequence_length=256,
    )
    settings = ModelSettings(
        model_name="Alpha-VLLM/Lumina-Image-2.0",
        output_name="lumina_image_2_0",
        model_output_type="image",
    )

    def _load_model(self) -> DiffusionPipeline:
        # Keep this import lazy so the runner still registers on diffusers builds
        # that predate Lumina2 and can report the required version at load time.
        from xfuser import xFuserLumina2Pipeline

        return xFuserLumina2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            engine_config=self.engine_config,
            torch_dtype=torch.bfloat16,
            cache_dir=self.config.download_dir,
        )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            prompt=input_args["prompt"],
            negative_prompt=input_args.get("negative_prompt"),
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            max_sequence_length=input_args["max_sequence_length"],
            cfg_trunc_ratio=0.25,
            cfg_normalization=True,
            generator=self._make_generator(input_args["seed"]),
        )
        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)
