from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed import get_world_group
from xfuser.core.utils.runner_utils import log
from xfuser.core.utils.video_utils import encode_video_with_audio
from xfuser.model_executor.models.runner_models.base_model import (
    DIFFUSERS_FROM_SOURCE,
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    _parse_attention_backend,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    LoadRoute,
)


_SUPPORTED_ATTN_BACKENDS = frozenset({
    AttentionBackendType.AITER,
    AttentionBackendType.AITER_FP8,
})
_SUPPORTED_ULYSSES_DEGREES = frozenset({1, 2, 4, 8})
_SUPPORTED_TASKS = frozenset({"t2va", "i2va", "l2va", "fl2va", "ref2va"})


def _patch_minimax_h3_text_encoder_broadcast() -> None:
    from diffusers.modular_pipelines.minimax_h3 import encoders

    if getattr(encoders, "_xfuser_broadcast_patched", False):
        return

    original_get_prompt_embeds = encoders.get_qwen3vl_prompt_embeds

    def distributed_get_prompt_embeds(
        text_encoder,
        processor,
        token_ids,
        vision_inputs=None,
        text_encoder_layer=50,
        device=None,
        dtype=None,
    ):
        world_group = get_world_group()
        if world_group.world_size == 1:
            return original_get_prompt_embeds(
                text_encoder,
                processor,
                token_ids,
                vision_inputs,
                text_encoder_layer=text_encoder_layer,
                device=device,
                dtype=dtype,
            )

        device = device or text_encoder.device
        dtype = dtype or text_encoder.dtype
        source_rank = world_group.first_rank
        metadata = [None]

        if world_group.rank == source_rank:
            prompt_embeds = original_get_prompt_embeds(
                text_encoder,
                processor,
                token_ids,
                vision_inputs,
                text_encoder_layer=text_encoder_layer,
                device=device,
                dtype=dtype,
            )
            metadata[0] = tuple(prompt_embeds.shape)

        torch.distributed.broadcast_object_list(
            metadata,
            src=source_rank,
            group=world_group.cpu_group,
        )
        if world_group.rank != source_rank:
            prompt_embeds = torch.empty(
                metadata[0],
                dtype=dtype,
                device=device,
            )

        torch.distributed.broadcast(
            prompt_embeds,
            src=source_rank,
            group=world_group.device_group,
        )
        return prompt_embeds

    encoders.get_qwen3vl_prompt_embeds = distributed_get_prompt_embeds
    encoders._xfuser_broadcast_patched = True


class MiniMaxH3DiffusionOutput(DiffusionOutput):
    def __init__(
        self,
        videos,
        audio,
        audio_sample_rate: int,
        pipe_args,
    ) -> None:
        super().__init__(videos=videos, pipe_args=pipe_args)
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate


@register_model("MiniMaxAI/MiniMax-H3")
@register_model("MiniMax-H3")
class xFuserMiniMaxH3Model(xFuserModel):
    # Native MiniMax-H3 is on Diffusers main from f53d552, but not in a release yet.
    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    default_input_values = DefaultInputValues(
        height=768,
        width=1344,
        num_frames=124,
        num_inference_steps=50,
    )

    settings = ModelSettings(
        model_name="MiniMaxAI/MiniMax-H3",
        output_name="minimax_h3",
        model_output_type="video",
        fps=24,
        resolution_divisor=32,
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp8_gemm_include_suffixes=("attn.to_qkv", "ff.net.0.proj"),
        fp4_gemm_module_list=["transformer.transformer_blocks"],
        fp8_precision_override_suffixes=(
            "attn.to_out.0",
            "ff.net.2",
            "adaln_proj.linear",
        ),
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
        },
        valid_tasks=["t2va", "i2va", "l2va", "fl2va"],
    )

    # Modular loading and QKV fusion leave no collective-safe checkpoint mapping.
    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        data_parallel_degree=False,
        text_encoder_tp_degree=True,
        use_cfg_parallel=False,
        use_parallel_vae=False,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        enable_slicing=False,
        enable_tiling=False,
    )
    _transformer_component_name = "transformer"

    def _get_runtime_state_pipeline(self):
        if self._transformer_component_name == "transformer":
            return self.pipe
        return SimpleNamespace(
            transformer=getattr(self.pipe, self._transformer_component_name)
        )

    def _validate_config(self, config) -> None:
        if config.task is None:
            if self.settings.valid_tasks == ["ref2va"]:
                config.task = "ref2va"
            elif len(config.input_images) == 0:
                config.task = "t2va"
            elif len(config.input_images) == 1:
                config.task = "i2va"
            else:
                config.task = "fl2va"
        super()._validate_config(config)
        backend = _parse_attention_backend(
            config.attention_backend,
            "attention backend",
        )
        if backend is not None and backend not in _SUPPORTED_ATTN_BACKENDS:
            supported = ", ".join(sorted(item.name for item in _SUPPORTED_ATTN_BACKENDS))
            raise ValueError(
                f"MiniMax-H3 currently supports only packed varlen AITER attention. "
                f"Supported backends: {supported}."
            )
        if backend == AttentionBackendType.AITER_FP8:
            try:
                from aiter import flash_attn_varlen_fp8_pertensor_func  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "MiniMax-H3 FP8 attention requires AITER varlen FP8 flash attention."
                ) from None

        ulysses_degree = config.ulysses_degree or 1
        if ulysses_degree not in _SUPPORTED_ULYSSES_DEGREES:
            raise ValueError(
                "MiniMax-H3 Ulysses degree must divide both 56 attention heads "
                "and the 64-row packed-sequence alignment. Supported values: "
                f"{sorted(_SUPPORTED_ULYSSES_DEGREES)}."
            )
        text_encoder_tp_degree = config.text_encoder_tp_degree or 1
        if text_encoder_tp_degree not in (1, ulysses_degree):
            raise ValueError(
                "MiniMax-H3 text encoder TP reuses the Ulysses ranks, so "
                "--text_encoder_tp_degree must be 1 or match --ulysses_degree."
            )
        if text_encoder_tp_degree > 1 and (
            config.enable_model_cpu_offload
            or config.enable_sequential_cpu_offload
            or config.enable_group_cpu_offload
        ):
            raise ValueError(
                "MiniMax-H3 text encoder TP is incompatible with CPU offloading."
            )
        if config.batch_size is not None or config.dataset_path is not None:
            raise ValueError(
                "MiniMax-H3 packs one request into one audiovisual sequence and "
                "does not support runner batching or dataset batching."
            )
        if config.task is not None and config.task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported MiniMax-H3 task {config.task!r}. "
                f"Supported tasks: {sorted(_SUPPORTED_TASKS)}."
            )

    def preprocess_args(self, input_args: dict) -> dict:
        args = super().preprocess_args(input_args)
        if isinstance(args.get("prompt"), list) and len(args["prompt"]) == 1:
            args["prompt"] = args["prompt"][0]
        return args

    def _load_model(self):
        from diffusers import ModularPipeline
        from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
            xFuserMiniMaxH3Transformer3DWrapper,
        )

        workflow = "t2va" if self.config.task == "t2va" else "fl2va"
        log(f"Loading {self.settings.model_name} {workflow.upper()} components")
        if self.config.text_encoder_tp_degree <= 1:
            _patch_minimax_h3_text_encoder_broadcast()
        pipe = ModularPipeline.from_pretrained(
            self.settings.model_name,
            workflow=workflow,
        )
        transformer = xFuserMiniMaxH3Transformer3DWrapper.from_pretrained(
            self.settings.model_name,
            subfolder="transformer",
            dtype=torch.bfloat16,
        )
        pipe.update_components(transformer=transformer)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.transformer.fuse_qkv_projections()
        pipe.text_encoder.lm_head = None
        self._parallelize_text_encoder(pipe.text_encoder)
        return pipe

    @staticmethod
    def _build_text_encoder_tp_plan(num_layers: int):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
        )

        plan = {}
        for layer_index in range(num_layers):
            prefix = f"layers.{layer_index}"
            plan.update(
                {
                    f"{prefix}.self_attn.q_proj": ColwiseParallel(),
                    f"{prefix}.self_attn.k_proj": ColwiseParallel(),
                    f"{prefix}.self_attn.v_proj": ColwiseParallel(),
                    f"{prefix}.self_attn.o_proj": RowwiseParallel(),
                    f"{prefix}.mlp.gate_proj": ColwiseParallel(),
                    f"{prefix}.mlp.up_proj": ColwiseParallel(),
                    f"{prefix}.mlp.down_proj": RowwiseParallel(),
                }
            )
        return plan

    def _parallelize_text_encoder(self, text_encoder) -> None:
        degree = self.config.text_encoder_tp_degree or 1
        if degree <= 1:
            return

        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor.parallel import parallelize_module

        world_group = get_world_group()
        if degree != world_group.world_size:
            raise ValueError(
                f"MiniMax-H3 text encoder TP degree {degree} must match world size "
                f"{world_group.world_size}."
            )

        device = torch.device(f"cuda:{world_group.local_rank}")
        language_model = text_encoder.model.language_model
        text_encoder.to(device)
        mesh = DeviceMesh.from_group(world_group.device_group, "cuda")
        parallelize_module(
            language_model,
            device_mesh=mesh,
            parallelize_plan=self._build_text_encoder_tp_plan(
                len(language_model.layers)
            ),
            src_data_rank=None,
        )
        text_encoder._xfuser_tp_mesh = mesh
        log(f"Sharded MiniMax-H3 Qwen3-VL text encoder with TP{degree}.")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = input_args["prompt"]
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniMax-H3 currently supports one prompt per request.")
            prompt = prompt[0]

        pipe_args = {
            "prompt": prompt,
            "height": input_args["height"],
            "width": input_args["width"],
            "num_frames": input_args["num_frames"],
            "num_inference_steps": input_args["num_inference_steps"],
            "generator": torch.Generator(device="cuda").manual_seed(
                input_args["seed"]
            ),
            "output_type": "np",
        }
        images = input_args.get("input_images") or []
        task = input_args.get("task")
        if task is None:
            task = "t2va" if not images else ("i2va" if len(images) == 1 else "fl2va")
        if task == "t2va":
            if images:
                raise ValueError("T2VA does not accept input images.")
        elif task == "i2va":
            if len(images) != 1:
                raise ValueError("I2VA requires exactly one first-frame image.")
            pipe_args["image"] = images[0]
        elif task == "l2va":
            if len(images) != 1:
                raise ValueError("L2VA requires exactly one last-frame image.")
            pipe_args["last_image"] = images[0]
        else:
            if len(images) != 2:
                raise ValueError(
                    "FL2VA requires two input images: first frame followed by last frame."
                )
            pipe_args["image"] = images[0]
            pipe_args["last_image"] = images[1]

        state = self.pipe(
            **pipe_args,
        )
        videos = state.get("videos")
        audio = state.get("audio")
        sampling_rate = int(state.get("sampling_rate"))
        return MiniMaxH3DiffusionOutput(
            videos=videos,
            audio=audio,
            audio_sample_rate=sampling_rate,
            pipe_args=input_args,
        )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        prompt = input_args["prompt"]
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("MiniMax-H3 currently supports one prompt per request.")
        if input_args.get("negative_prompt") is not None:
            raise ValueError(
                "MiniMax-H3 is guidance-distilled and does not accept a negative prompt."
            )

    def _compile_model(self, input_args: dict) -> None:
        if self.config.fully_shard_degree > 1:
            super()._compile_model(input_args)
            return
        self._enable_compute_comm_overlap()
        transformer = getattr(self.pipe, self._transformer_component_name)
        transformer.forward = torch.compile(
            transformer.forward,
            mode=self._get_compile_mode(),
            fullgraph=True,
        )
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 3
        self._run_timed_pipe(compile_args)

    def _run_warmup_calls(self, input_args: dict) -> None:
        if not self.config.warmup_calls:
            return
        log(
            f"Warming up MiniMax-H3 with {self.config.warmup_calls} "
            "three-point calls..."
        )
        warmup_args = copy.deepcopy(input_args)
        warmup_args["num_inference_steps"] = 3
        for iteration in range(self.config.warmup_calls):
            log(f"Warmup iteration {iteration + 1}/{self.config.warmup_calls}")
            self._run_timed_pipe(warmup_args)
        log("Warmup complete.")

    @staticmethod
    def _video_to_uint8(video) -> torch.Tensor:
        if isinstance(video, torch.Tensor):
            if video.dtype == torch.uint8:
                return video.cpu()
            video = video.detach().float().cpu().numpy()
        if isinstance(video, list):
            video = np.stack([np.asarray(frame) for frame in video])
        if video.dtype != np.uint8:
            video = (video * 255.0).round().clip(0, 255).astype("uint8")
        return torch.from_numpy(video)

    def save_output(self, output: DiffusionOutput) -> None:
        audio = output.audio
        if isinstance(audio, list):
            audio_items = audio
        elif isinstance(audio, torch.Tensor) and audio.ndim >= 3:
            audio_items = list(audio)
        else:
            audio_items = [audio]

        for index, (video, pipe_args) in enumerate(output.get_outputs()):
            video = self._video_to_uint8(video)
            audio_item = audio_items[min(index, len(audio_items) - 1)]
            if not isinstance(audio_item, torch.Tensor):
                audio_item = torch.as_tensor(audio_item)
            output_name = self.get_output_name(pipe_args)
            output_path = f"{self.config.output_directory}/{output_name}_{index}.mp4"
            encode_video_with_audio(
                video,
                audio=audio_item.float().cpu(),
                audio_sample_rate=output.audio_sample_rate,
                fps=self.settings.fps,
                output_path=output_path,
            )
            log(f"Output video with audio saved to {output_path}")


@register_model("MiniMax-H3-Ref2VA")
class xFuserMiniMaxH3Ref2VAModel(xFuserMiniMaxH3Model):
    # The reference workflow shares the modular loading and QKV-fusion limitation.
    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )

    _transformer_component_name = "transformer_ref"
    settings = ModelSettings(
        model_name="MiniMaxAI/MiniMax-H3",
        output_name="minimax_h3_ref2va",
        model_output_type="video",
        fps=24,
        resolution_divisor=32,
        fp8_gemm_module_list=["transformer_ref.transformer_blocks"],
        fp8_gemm_include_suffixes=("attn.to_qkv", "ff.net.0.proj"),
        fp4_gemm_module_list=["transformer_ref.transformer_blocks"],
        fp8_precision_override_suffixes=(
            "attn.to_out.0",
            "ff.net.2",
            "adaln_proj.linear",
        ),
        fsdp_strategy={
            "transformer_ref": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
        },
        valid_tasks=["ref2va"],
    )

    def _load_model(self):
        from diffusers import ModularPipeline
        from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
            xFuserMiniMaxH3Transformer3DWrapper,
        )

        log(f"Loading {self.settings.model_name} Ref2VA components")
        if self.config.text_encoder_tp_degree <= 1:
            _patch_minimax_h3_text_encoder_broadcast()
        pipe = ModularPipeline.from_pretrained(
            self.settings.model_name,
            workflow="ref2va",
        )
        transformer = xFuserMiniMaxH3Transformer3DWrapper.from_pretrained(
            self.settings.model_name,
            subfolder="transformer_ref",
            dtype=torch.bfloat16,
        )
        pipe.update_components(transformer_ref=transformer)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.transformer_ref.fuse_qkv_projections()
        pipe.text_encoder.lm_head = None
        self._parallelize_text_encoder(pipe.text_encoder)
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

        prompt = input_args["prompt"]
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniMax-H3 Ref2VA supports one prompt per request.")
            prompt = prompt[0]

        images = input_args.get("input_images") or []
        if not images:
            raise ValueError(
                "MiniMax-H3 Ref2VA currently requires at least one image reference."
            )
        references = [MiniMaxH3ImageReference(image=image) for image in images]
        state = self.pipe(
            prompt=prompt,
            references=references,
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            generator=torch.Generator(device="cuda").manual_seed(
                input_args["seed"]
            ),
            output_type="np",
        )
        return MiniMaxH3DiffusionOutput(
            videos=state.get("videos"),
            audio=state.get("audio"),
            audio_sample_rate=int(state.get("sampling_rate")),
            pipe_args=input_args,
        )

    def _validate_args(self, input_args: dict) -> None:
        xFuserModel._validate_args(self, input_args)
        if input_args.get("negative_prompt") is not None:
            raise ValueError(
                "MiniMax-H3 is guidance-distilled and does not accept a negative prompt."
            )
        if not input_args.get("input_images"):
            raise ValueError(
                "MiniMax-H3 Ref2VA currently requires image references through "
                "--input_images."
            )
