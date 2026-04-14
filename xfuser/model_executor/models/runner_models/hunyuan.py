import torch
import copy
import json
import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from collections import OrderedDict
from diffusers import HunyuanVideoPipeline, HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import xFuserHunyuanVideoTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import xFuserHunyuanVideo15Transformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
)
from xfuser.envs import PACKAGES_CHECKER

@register_model("tencent/HunyuanVideo")
@register_model("HunyuanVideo")
class xFuserHunyuanvideoModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
        use_hybrid_attn_schedule=True
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=50,
        guidance_scale=6.0,
        num_hybrid_attn_high_precision_steps = 5,
    )
    settings = ModelSettings(
        model_name="tencent/HunyuanVideo",
        output_name="hunyuan_video",
        model_output_type="video",
        fps=24,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserHunyuanVideoTransformer3DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            revision="refs/pr/18",
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            prompt=input_args["prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer.compile()

        compile_args = copy.deepcopy(input_args)
        # If a per-step attention schedule is active, do a full warmup to trigger all backend paths.
        if not get_runtime_state().has_attention_schedule():
            compile_args["num_inference_steps"] = 2 # Reduce steps for warmup
        self._run_timed_pipe(compile_args)


@register_model("tencent/HunyuanVideo-1.5")
@register_model("Hunyuanvideo-1.5")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v")
class xFuserHunyuanvideo15Model(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=121,
        num_inference_steps=50,
    )
    settings = ModelSettings(
        output_name="hunyuan_video_1_5",
        model_output_type="video",
        fps=24,
        mod_value=16,
        valid_tasks=["i2v", "t2v"],
    )


    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        if self.config.task == "i2v": # TODO: different model for 480p
            self.settings.model_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v"
        else:
            self.settings.model_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"


    def _load_model(self) -> DiffusionPipeline:
        task = self.config.task
        pipeline = HunyuanVideo15Pipeline if task == "t2v" else HunyuanVideo15ImageToVideoPipeline
        transformer = xFuserHunyuanVideo15Transformer3DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "num_inference_steps": input_args["num_inference_steps"],
            "num_frames": input_args["num_frames"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            "prompt": input_args["prompt"],
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        else: #t2v task
            kwargs["height"] = input_args["height"]
            kwargs["width"] = input_args["width"]

        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task == "i2v":
            image = input_args["input_images"][0]
            if input_args.get("resize_input_images", False):
                image = resize_and_crop_image(image, input_args["width"], input_args["height"], self.settings.mod_value)
            input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        if self.config.task == "i2v":
            images = input_args.get("input_images", [])
            if len(images) != 1:
                raise ValueError("Exactly one input image is required for HunyuanVideo-1.5 model when task is 'i2v'.")

    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)


@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled")
@register_model("tencent/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled")
@register_model("Hunyuanvideo-1.5-Distilled")
class xFuserHunyuanvideo15DistilledModel(xFuserHunyuanvideo15Model):
    
    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        self.settings.model_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"
        self.settings.output_name = "hunyuan_video_1_5_distilled"
        self.settings.valid_tasks = ["i2v"]


HUNYUANVIDEO_15_SPARSE_BLOCK_KEY_MAP = {
    # Image-side attention
    "img_attn_q.": "attn.to_q.",
    "img_attn_k.": "attn.to_k.",
    "img_attn_v.": "attn.to_v.",
    "img_attn_proj.": "attn.to_out.0.",
    "img_attn_q_norm.": "attn.norm_q.",
    "img_attn_k_norm.": "attn.norm_k.",
    # Text-side attention
    "txt_attn_q.": "attn.add_q_proj.",
    "txt_attn_k.": "attn.add_k_proj.",
    "txt_attn_v.": "attn.add_v_proj.",
    "txt_attn_proj.": "attn.to_add_out.",
    "txt_attn_q_norm.": "attn.norm_added_q.",
    "txt_attn_k_norm.": "attn.norm_added_k.",
    # Modulation
    "img_mod.": "norm1.",
    "txt_mod.": "norm1_context.",
    # MLP (img)
    "img_mlp.fc1.": "ff.net.0.proj.",
    "img_mlp.fc2.": "ff.net.2.",
    # MLP (txt)
    "txt_mlp.fc1.": "ff_context.net.0.proj.",
    "txt_mlp.fc2.": "ff_context.net.2.",
}
HUNYUANVIDEO_15_SPARSE_SINGLE_BLOCK_KEY_MAP = {
    "linear1_q.": "attn.to_q.",
    "linear1_k.": "attn.to_k.",
    "linear1_v.": "attn.to_v.",
    "linear1_mlp.": "proj_mlp.",
    "linear2.fc.": "proj_out.",
    "q_norm.": "attn.norm_q.",
    "k_norm.": "attn.norm_k.",
    "modulation.": "norm.",
}

@register_model("tencent/HunyuanVideo-1.5-Sparse")
@register_model("Hunyuanvideo-1.5-Sparse")
@register_model("tencent/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled_sparse")
class xFuserHunyuanvideo15SparseModel(xFuserHunyuanvideo15Model):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
        supports_sparse_attention_backends=True,
    )

    def _validate_ssta_attention_kwargs(self, attn_param: dict) -> None:
        assert attn_param["tile_size"] is not None, "tile_size is not set"
        assert len(attn_param["tile_size"]) == 3, "tile_size must be a tuple of 3 integers"
        assert np.prod(attn_param["tile_size"]) == 128 or np.prod(attn_param["tile_size"]) == 384, "product of ssta_tile_thw must be 128 or 384"
        TritonSparseAttentionBackendTypes = [AttentionBackendType.AITER_SPARSE_SAGE,
                                             AttentionBackendType.AITER_SPARSE_SAGE_V2]
        if AttentionBackendType[self.config.attention_backend.upper()] in TritonSparseAttentionBackendTypes:
            assert np.prod(attn_param["tile_size"]) == 128, "product of ssta_tile_thw must be 128 for AITER_SPARSE_SAGE and AITER_SPARSE_SAGE_V2"

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        self.settings.model_name = "tencent/HunyuanVideo-1.5"
        self.settings.output_name = "hunyuan_video_1_5_sparse"
        self.settings.valid_tasks = ["i2v"]
        self.pipe_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"


    def _load_model(self) -> DiffusionPipeline:
        pipeline = HunyuanVideo15ImageToVideoPipeline
        # Load the distilled transformer (diffusers format) to get non-block weights
        distilled_transformer = xFuserHunyuanVideo15Transformer3DWrapper.from_pretrained(
            self.pipe_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        distilled_state = distilled_transformer.state_dict()

        config_path = hf_hub_download(
            self.settings.model_name,
            "config.json",
            subfolder="transformer/720p_i2v_distilled_sparse",
        )
        with open(config_path) as f:
            sparse_config = json.load(f)

        if self.config.ssta_tile_thw is not None:
            sparse_config["attn_param"]["tile_size"] = self.config.ssta_tile_thw
        self._validate_ssta_attention_kwargs(sparse_config["attn_param"])
        
        transformer = xFuserHunyuanVideo15Transformer3DWrapper(
            in_channels=65,  # diffusers i2v: 32 latent * 2 + 1 mask
            out_channels=sparse_config.get("out_channels", 32),
            num_attention_heads=sparse_config.get("heads_num", 16),
            attention_head_dim=sparse_config.get("hidden_size", 2048) // sparse_config.get("heads_num", 16),
            num_layers=sparse_config.get("mm_double_blocks_depth", 54),
            num_refiner_layers=sparse_config.get("num_refiner_layers", 2),
            mlp_ratio=sparse_config.get("mlp_width_ratio", 4.0),
            patch_size=sparse_config.get("patch_size",  1),
            patch_size_t=sparse_config.get("patch_size_t", 1),
            qk_norm=sparse_config.get("qk_norm_type", "rms_norm"),
            text_embed_dim=sparse_config.get("text_states_dim", 3584),
            text_embed_2_dim=sparse_config.get("text_states_dim_2") or 1472,
            image_embed_dim=sparse_config.get("vision_states_dim", 1152),
            rope_theta=sparse_config.get("rope_theta", 256),
            rope_axes_dim=sparse_config.get("rope_dim_list", (16, 56, 56)),
            target_size=sparse_config.get("target_size", 960),
            task_type="i2v",
            use_meanflow=sparse_config.get("use_meanflow", False),
            attention_kwargs=sparse_config.get("attn_param", None),
        )

        # Load non-block weights from distilled model (already in diffusers naming)
        non_block_state = OrderedDict()
        for key, value in distilled_state.items():
            if not key.startswith("transformer_blocks.") and not key.startswith("single_transformer_blocks."):
                non_block_state[key] = value
        transformer.load_state_dict(non_block_state, strict=False)

        # Load sparse block weights (Tencent format), remap, and load
        weight_file = hf_hub_download(
            self.settings.model_name,
            "diffusion_pytorch_model.safetensors",
            subfolder="transformer/720p_i2v_distilled_sparse",
        )
        state_dict = load_file(weight_file)
        
        # Remap double_blocks -> transformer_blocks
        BLOCK_REMAP = {
            "double_blocks.": ("transformer_blocks.", HUNYUANVIDEO_15_SPARSE_BLOCK_KEY_MAP),
            "single_blocks.": ("single_transformer_blocks.", HUNYUANVIDEO_15_SPARSE_SINGLE_BLOCK_KEY_MAP),
        }
        remapped = OrderedDict()
        for key, value in state_dict.items():
            for src_prefix, (dst_prefix, key_map) in BLOCK_REMAP.items():
                if key.startswith(src_prefix):
                    parts = key.split(".", 2)
                    block_idx, rest = parts[1], parts[2]
                    for old, new in key_map.items():
                        if rest.startswith(old):
                            rest = new + rest[len(old):]
                            break
                    remapped[f"{dst_prefix}{block_idx}.{rest}"] = value
                    break

        # Only load block keys to avoid overwriting correct non-block weights with unremapped Tencent keys
        block_state = OrderedDict()
        for key, value in remapped.items():
            if key.startswith("transformer_blocks.") or key.startswith("single_transformer_blocks."):
                block_state[key] = value
        transformer.load_state_dict(block_state, strict=False)
        transformer = transformer.to(torch.bfloat16)

        del distilled_transformer, distilled_state

        pipe = pipeline.from_pretrained(
            pretrained_model_name_or_path=self.pipe_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe