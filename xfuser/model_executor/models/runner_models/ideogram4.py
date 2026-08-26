from __future__ import annotations

import json
import os

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)
from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
    DerivedTensor,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)
from xfuser.model_executor.pipelines.pipeline_ideogram4 import (
    get_ideogram4_pipeline_class,
)


FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# What the checkpoint appends to a weight's own name to name its scale, which is
# how the scale is found from the weight rather than the other way round
_SCALE_SUFFIX = "_scale"


def _resolve_pretrained_file(model_id: str, filename: str) -> str:
    if os.path.isdir(model_id):
        path = os.path.join(model_id, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_id, filename=filename)


def _load_sharded_safetensors(
    model_id: str,
    subfolder: str,
    basename: str = "diffusion_pytorch_model",
) -> dict[str, torch.Tensor]:
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import load_file

    index_filename = f"{subfolder}/{basename}.safetensors.index.json"
    try:
        index_path = _resolve_pretrained_file(model_id, index_filename)
    except (EntryNotFoundError, FileNotFoundError):
        model_path = _resolve_pretrained_file(
            model_id,
            f"{subfolder}/{basename}.safetensors",
        )
        return load_file(model_path, device="cpu")

    with open(index_path) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    state_dict = {}
    for shard_filename in sorted(set(weight_map.values())):
        shard_path = _resolve_pretrained_file(
            model_id,
            f"{subfolder}/{shard_filename}",
        )
        state_dict.update(load_file(shard_path, device="cpu"))
    return state_dict


def _is_fp8_checkpoint(model_id: str) -> bool:
    try:
        config_path = _resolve_pretrained_file(
            model_id,
            "text_encoder/config.json",
        )
        with open(config_path) as config_file:
            config = json.load(config_file)
        if config.get("ideogram_fp8_weight_only", False):
            return True
    except Exception:
        pass

    try:
        index_path = _resolve_pretrained_file(
            model_id,
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        )
        with open(index_path) as index_file:
            keys = json.load(index_file)["weight_map"]
        return any(key.endswith(FP8_SCALE_SUFFIX) for key in keys)
    except Exception:
        return False


def _convert_ideogram_fp8_transformer_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted = {}
    for key, tensor in state_dict.items():
        if key.endswith(".attention.qkv.weight_scale"):
            base = key.removesuffix(".attention.qkv.weight_scale")
            query, key_scale, value = tensor.chunk(3, dim=0)
            converted[f"{base}.attention.to_q{FP8_SCALE_SUFFIX}"] = query
            converted[f"{base}.attention.to_k{FP8_SCALE_SUFFIX}"] = key_scale
            converted[f"{base}.attention.to_v{FP8_SCALE_SUFFIX}"] = value
        elif key.endswith(".attention.qkv.weight"):
            base = key.removesuffix(".attention.qkv.weight")
            query, key_weight, value = tensor.chunk(3, dim=0)
            converted[f"{base}.attention.to_q.weight"] = query
            converted[f"{base}.attention.to_k.weight"] = key_weight
            converted[f"{base}.attention.to_v.weight"] = value
        elif key.endswith(".attention.o.weight_scale"):
            converted[
                key.replace(
                    ".attention.o.weight_scale",
                    ".attention.to_out.0.weight_scale",
                )
            ] = tensor
        elif key.endswith(".attention.o.weight"):
            converted[
                key.replace(
                    ".attention.o.weight",
                    ".attention.to_out.0.weight",
                )
            ] = tensor
        else:
            converted[key] = tensor
    return converted


def _dequantize_fp8_state_dict(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    result = {}
    for key, tensor in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            continue

        scale_key = f"{key}_scale"
        if tensor.dtype == FP8_WEIGHT_DTYPE and scale_key in state_dict:
            scale = state_dict[scale_key].to(torch.float32)
            result[key] = (tensor.to(torch.float32) * scale.unsqueeze(-1)).to(dtype)
        elif tensor.is_floating_point():
            result[key] = tensor.to(dtype)
        else:
            result[key] = tensor
    return result


def _dequantize_fp8(weight, scale, dtype: torch.dtype = torch.bfloat16):
    """One FP8 weight and its per-row scale, as the model's own dtype."""
    return (weight.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(dtype)


def _dequantize_fp8_chunk(weight, scale, *, index: int, chunks: int = 3):
    """One projection out of a fused one, dequantized from its slice of the scale."""
    return _dequantize_fp8(
        weight.chunk(chunks, dim=0)[index],
        scale.chunk(chunks, dim=0)[index],
    )


def _stored_tensor_names(path: str) -> list[str]:
    """The shard's tensor names, from its header, without reading a payload."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return list(handle.keys())


def _shard_paths(model_id: str, subfolder: str, basename: str) -> list[str]:
    from huggingface_hub.errors import EntryNotFoundError

    index_filename = f"{subfolder}/{basename}.safetensors.index.json"
    try:
        index_path = _resolve_pretrained_file(model_id, index_filename)
    except (EntryNotFoundError, FileNotFoundError):
        return [
            _resolve_pretrained_file(model_id, f"{subfolder}/{basename}.safetensors")
        ]
    with open(index_path) as index_file:
        weight_map = json.load(index_file)["weight_map"]
    return [
        _resolve_pretrained_file(model_id, f"{subfolder}/{shard}")
        for shard in sorted(set(weight_map.values()))
    ]


def _fp8_transformer_manifest(
    model_id: str,
    subfolder: str,
    basename: str = "diffusion_pytorch_model",
) -> CheckpointManifest:
    """Map this checkpoint's stored tensors onto the model's own parameter names.

    The same two conversions the eager path does in memory, expressed per tensor so
    a block can be filled without the whole state dict: the fused qkv weight is a
    third of itself for each of to_q, to_k and to_v, every FP8 weight is read with
    the scale stored beside it, and the attention output projection is named `o`
    here and `to_out.0` in the model.
    """
    from functools import partial

    weight_map: dict[str, str] = {}
    checkpoint_keys: dict[str, str] = {}
    derived: dict[str, DerivedTensor] = {}
    for path in _shard_paths(model_id, subfolder, basename):
        stored = set(_stored_tensor_names(path))
        for name in sorted(stored):
            if name.endswith(_SCALE_SUFFIX):
                # Read as part of the weight it scales, never on its own
                continue
            scale = f"{name}{_SCALE_SUFFIX}"
            if scale not in stored:
                weight_map[name] = path
                continue
            if name.endswith(".attention.qkv.weight"):
                base = name.removesuffix(".attention.qkv.weight")
                for index, projection in enumerate(("to_q", "to_k", "to_v")):
                    live = f"{base}.attention.{projection}.weight"
                    weight_map[live] = path
                    derived[live] = DerivedTensor(
                        sources=(name, scale),
                        build=partial(_dequantize_fp8_chunk, index=index),
                        description=f"{projection} third of a fused qkv weight",
                    )
                continue
            live = name
            if name.endswith(".attention.o.weight"):
                live = name.replace(
                    ".attention.o.weight", ".attention.to_out.0.weight"
                )
            weight_map[live] = path
            derived[live] = DerivedTensor(
                sources=(name, scale),
                build=_dequantize_fp8,
                description="FP8 weight read with its scale",
            )
    return CheckpointManifest(
        weight_map=weight_map,
        checkpoint_keys=checkpoint_keys,
        derived=derived,
        strict=True,
        label=f"{subfolder} (Ideogram FP8)",
    )


def _check_load_result(
    component_name: str,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load {component_name}: "
            f"missing keys={missing_keys[:10]}, unexpected keys={unexpected_keys[:10]}"
        )


def _default_guidance_schedule(num_inference_steps: int) -> list[float]:
    polish_steps = min(3, num_inference_steps)
    return [7.0] * (num_inference_steps - polish_steps) + [3.0] * polish_steps


@register_model("ideogram-ai/ideogram-v4")
@register_model("ideogram-ai/ideogram-4-nf4")
@register_model("ideogram-ai/ideogram-4-nf4-diffusers")
@register_model("ideogram-ai/ideogram-4-fp8")
@register_model("CalamitousFelicitousness/Ideogram-4-bf16-Diffusers")
@register_model("Ideogram-4")
class xFuserIdeogram4Model(xFuserModel):
    min_diffusers_version = "0.39.0"

    # The transformer config states this too, and runtime_state re-checks it against the loaded
    # model. Stated here so a Ulysses degree that cannot work is refused before the download.
    attention_heads = 18

    load_support = LoadSupport(
        meta_transformers=('transformer', 'unconditional_transformer'),
        # trust_remote_code hides text-encoder parameter names from manifest discovery.
        meta_text_encoders=(),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        fully_shard_degree=True,
        use_parallel_vae=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=48,
        max_sequence_length=2048,
    )
    settings = ModelSettings(
        model_name="ideogram-ai/ideogram-4-fp8",
        output_name="ideogram4",
        model_output_type="image",
        mod_value=16,
        resolution_divisor=16,
        fp8_gemm_module_list=[
            "transformer.layers",
            "unconditional_transformer.layers",
        ],
        fp4_gemm_module_list=[
            "transformer.layers",
            "unconditional_transformer.layers",
        ],
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["layers"]},
            "unconditional_transformer": {"wrap_attrs": ["layers"]},
        },
    )

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        heads = self.attention_heads
        if heads % config.ulysses_degree != 0:
            raise ValueError(
                f"Ideogram 4 has {heads} attention heads, so --ulysses_degree must "
                f"divide {heads}."
            )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        height = input_args["height"]
        width = input_args["width"]
        if height < 256 or width < 256:
            raise ValueError(
                f"Ideogram 4 requires height and width of at least 256, got {height}x{width}."
            )

    def _load_fp8_transformer(
        self,
        model_id: str,
        subfolder: str,
    ):
        from xfuser.model_executor.models.transformers.transformer_ideogram4 import (
            get_ideogram4_transformer_wrapper_class,
        )

        transformer_class = get_ideogram4_transformer_wrapper_class()
        transformer = transformer_class.from_config(
            transformer_class.load_config(model_id, subfolder=subfolder)
        )
        transformer._install_xfuser_processors()

        state_dict = _load_sharded_safetensors(model_id, subfolder)
        state_dict = _convert_ideogram_fp8_transformer_keys(state_dict)
        state_dict = _dequantize_fp8_state_dict(state_dict)
        missing_keys, unexpected_keys = transformer.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
        _check_load_result(subfolder, missing_keys, unexpected_keys)
        transformer.eval()
        log(f"Loaded {model_id}/{subfolder} through the Ideogram FP8 converter.")
        return transformer

    def _meta_fp8_transformer(self, transformer_class, subfolder: str):
        """Build this denoiser on meta and fill it per block from the FP8 checkpoint.

        The eager path above reads every tensor, converts the whole state dict and
        assigns it, so both denoisers exist in full before either is sharded. Here the
        conversion travels with the checkpoint map instead, as derived tensors, and the
        fill reads one block at a time.
        """

        return self.loader.load_transformer(
            transformer_class,
            subfolder=subfolder,
            weight_source=_fp8_transformer_manifest(
                self.settings.model_name, subfolder
            ),
        )

    def _load_fp8_text_encoder(self, model_id: str):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            model_id,
            subfolder="text_encoder",
            trust_remote_code=True,
        )
        text_encoder = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = _load_sharded_safetensors(
            model_id,
            "text_encoder",
            basename="model",
        )
        state_dict = _dequantize_fp8_state_dict(state_dict)
        missing_keys, unexpected_keys = text_encoder.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
        _check_load_result("text_encoder", missing_keys, unexpected_keys)
        text_encoder.eval()
        log(f"Loaded {model_id}/text_encoder through the Ideogram FP8 converter.")
        return text_encoder

    def _load_model(self) -> DiffusionPipeline:
        from xfuser.model_executor.models.transformers.transformer_ideogram4 import (
            get_ideogram4_transformer_wrapper_class,
        )

        # settings.model_name, not config.model: the latter can be a registry alias
        # such as "Ideogram-4", which is not a repo, and every checkpoint read here
        # goes to the hub or to a directory
        model_id = self.settings.model_name
        transformer_class = get_ideogram4_transformer_wrapper_class()

        if _is_fp8_checkpoint(model_id):
            if self.loader.fsdp_meta_load() or self.loader.replicated_broadcast_load():
                transformer = self._meta_fp8_transformer(
                    transformer_class, "transformer"
                )
                unconditional_transformer = self._meta_fp8_transformer(
                    transformer_class, "unconditional_transformer"
                )
            else:
                transformer = self._load_fp8_transformer(model_id, "transformer")
                unconditional_transformer = self._load_fp8_transformer(
                    model_id,
                    "unconditional_transformer",
                )
            text_encoder = self._load_fp8_text_encoder(model_id)
        else:
            transformer = transformer_class.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            unconditional_transformer = transformer_class.from_pretrained(
                model_id,
                subfolder="unconditional_transformer",
                torch_dtype=torch.bfloat16,
            )
            text_encoder = None

        pipeline_kwargs = {
            "transformer": transformer,
            "unconditional_transformer": unconditional_transformer,
            "torch_dtype": torch.bfloat16,
        }
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder

        try:
            from diffusers import Ideogram4PromptEnhancerHead

            pipeline_kwargs["prompt_enhancer_head"] = (
                Ideogram4PromptEnhancerHead.from_pretrained(
                    "diffusers/qwen3-vl-8b-instruct-lm-head",
                    torch_dtype=torch.bfloat16,
                )
            )
            log("Loaded the Ideogram 4 prompt enhancer head.")
        except Exception as error:
            log(
                "Prompt enhancer head is unavailable; plain text prompts will not "
                f"be upsampled automatically ({error})."
            )

        pipeline_class = get_ideogram4_pipeline_class()
        return pipeline_class.from_pretrained(
            model_id,
            **pipeline_kwargs,
        )

    @staticmethod
    def _is_json_prompt(prompt: str) -> bool:
        stripped = prompt.strip()
        return stripped.startswith("{") and stripped.endswith("}")

    def _should_upsample_prompt(self, prompt: str | list[str]) -> bool:
        if self.pipe.prompt_enhancer_head is None:
            return False
        prompts = [prompt] if isinstance(prompt, str) else prompt
        return all(not self._is_json_prompt(item) for item in prompts)

    def _get_compiled_pipe_components(self) -> list[str]:
        return ["transformer", "unconditional_transformer"]

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        guidance_scale = input_args.get("guidance_scale")
        guidance_schedule = (
            None
            if guidance_scale is not None
            else _default_guidance_schedule(input_args["num_inference_steps"])
        )
        output = self.pipe(
            prompt=input_args["prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            prompt_upsampling=self._should_upsample_prompt(input_args["prompt"]),
            max_sequence_length=input_args["max_sequence_length"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            output_type=input_args.get("output_type", "pil"),
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self.pipe.transformer._init_sp_state()
        self.pipe.unconditional_transformer._init_sp_state()
