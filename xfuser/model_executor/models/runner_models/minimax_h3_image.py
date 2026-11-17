"""Runner for MiniMaxAI/MiniMax-H3-Image.

Text-to-image, and editing when ``--input_images`` passes reference images. The
checkpoint is distilled to six sampling steps with one output head per step, and
guidance is baked into the weights, so there is no CFG branch to parallelize and no
negative prompt. Sequence parallelism is Ulysses over the packed sequence, as in
MiniMax-H3. The first rank encodes the prompt and broadcasts it, unless
``--text_encoder_tp_degree`` shards the Qwen3-VL text encoder, and
``--use_parallel_vae`` spreads the VAE's decode tiles over the ranks.
"""

from __future__ import annotations

import copy
import functools
import math
from types import MethodType

import torch

from xfuser.core.distributed import get_runtime_state
from xfuser.core.utils.runner_utils import log
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.layers import usp_sdma
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
from xfuser.model_executor.models.runner_models.text_encoder_utils import trim_text_encoder
from xfuser.model_executor.models.runner_models.loading.contracts import LoadRoute, LoadSupport
from xfuser.model_executor.models.runner_models.minimax_h3 import (
    _SUPPORTED_ATTN_BACKENDS,
    _SUPPORTED_ULYSSES_DEGREES,
    MiniMaxH3TextEncoderMixin,
    _load_minimax_h3_components,
    _patch_minimax_h3_text_encoder_broadcast,
)

MINIMAX_H3_IMAGE_STEPS = 6


def _decode_tiles(self, z: torch.Tensor) -> torch.Tensor:
    """MiniMax-H3-Image's tiled decode, with the tiles batched and spread over ranks.

    The VAE decodes and blends the same tiles as its own loop, but every tile has
    the same size (``_split_tiles`` widens the overlaps, not the tiles), so each
    rank stacks the tiles it owns into one decoder call instead of decoding them
    one at a time. A tile is a 16x16 latent window, so even a 16-megapixel canvas
    is a modest batch, and one call keeps a single batch shape for torch.compile.
    Tile ``t`` belongs to rank ``t % world_size``; the ranks all-gather the
    decoded tiles and each one stitches the full image.
    """
    if not self.use_tiling:
        return self._xfuser_original_decode(z)

    ratio = self.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = self._split_tiles(
        z.shape[-2] * ratio, self.tile_sample_min_height, self.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = self._split_tiles(
        z.shape[-1] * ratio, self.tile_sample_min_width, self.tile_sample_min_overlap_width
    )
    windows = [
        (y // ratio, (y + height) // ratio, x // ratio, (x + width) // ratio)
        for y, height in zip(y_indices, y_lengths)
        for x, width in zip(x_indices, x_lengths)
    ]

    group = self._xfuser_tile_group
    world_size = 1 if group is None else group.world_size
    rank = 0 if group is None else group.rank_in_group
    tiles_per_rank = math.ceil(len(windows) / world_size)
    owned = [z[..., top:bottom, left:right] for top, bottom, left, right in windows[rank::world_size]]

    batch_size = z.shape[0]
    tile_height, tile_width = y_lengths[0], x_lengths[0]
    decoded = torch.empty(
        (tiles_per_rank * batch_size, self.decoder.out_channels, tile_height, tile_width),
        dtype=self.decoder.proj_out.weight.dtype,
        device=z.device,
    )
    if owned:
        decoded[: len(owned) * batch_size] = self.decoder(self.post_quant_conv(torch.cat(owned)))
    if world_size > 1:
        decoded = group.all_gather(decoded, dim=0)

    def tile(index: int) -> torch.Tensor:
        position = (index % world_size) * tiles_per_rank + index // world_size
        return decoded[position * batch_size : (position + 1) * batch_size]

    columns = len(x_indices)
    rows = [[tile(i * columns + j) for j in range(columns)] for i in range(len(y_indices))]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def install_minimax_h3_image_tiled_decode(vae, group=None) -> None:
    """Batch MiniMax-H3-Image's tiled decode, spreading the tiles over ``group`` if given.

    Calling it again only changes the group, so the runner installs the batched
    decode at load time and the parallel VAE setup hands it the VAE group later.
    """
    if not hasattr(vae, "_xfuser_original_decode"):
        vae._xfuser_original_decode = vae._decode
        vae._decode = MethodType(_decode_tiles, vae)
    vae._xfuser_tile_group = group if group is not None and group.world_size > 1 else None
    if vae._xfuser_tile_group is not None:
        log(f"MiniMax-H3-Image parallel VAE: decode tiles spread over {group.world_size} ranks.")


@register_model("MiniMaxAI/MiniMax-H3-Image")
@register_model("MiniMax-H3-Image")
class xFuserMiniMaxH3ImageModel(MiniMaxH3TextEncoderMixin, xFuserModel):
    # From the registered checkpoint's transformer config.
    attention_heads = 56

    # diffusers main is the only source of MiniMax-H3-Image so far.
    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=MINIMAX_H3_IMAGE_STEPS,
        num_hybrid_attn_high_precision_steps=2,
    )

    settings = ModelSettings(
        model_name="MiniMaxAI/MiniMax-H3-Image",
        output_name="minimax_h3_image",
        model_output_type="image",
        resolution_divisor=32,
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp8_gemm_include_suffixes=("attn.to_qkv", "ff.net.0.proj"),
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
        },
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
        use_parallel_vae=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_hybrid_attn_schedule=True,
        enable_slicing=False,
        enable_tiling=False,
    )
    supported_attn_backends = _SUPPORTED_ATTN_BACKENDS

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        ulysses_degree = config.ulysses_degree or 1
        if ulysses_degree not in _SUPPORTED_ULYSSES_DEGREES:
            raise ValueError(
                "MiniMax-H3-Image Ulysses degree must divide both 56 attention heads "
                "and the 64-row packed-sequence alignment. Supported values: "
                f"{sorted(_SUPPORTED_ULYSSES_DEGREES)}."
            )
        text_encoder_tp_degree = config.text_encoder_tp_degree or 1
        if text_encoder_tp_degree not in (1, ulysses_degree):
            raise ValueError(
                "MiniMax-H3-Image text encoder TP reuses the Ulysses ranks, so "
                "--text_encoder_tp_degree must be 1 or match --ulysses_degree."
            )
        if config.use_parallel_vae and config.vae_parallel_size > 0:
            raise ValueError(
                "MiniMax-H3-Image parallel VAE spreads decode tiles over the model "
                "ranks and does not support dedicated VAE ranks; leave --vae_parallel_size at 0."
            )
        if config.enable_sequential_cpu_offload or config.enable_group_cpu_offload:
            raise ValueError(
                "MiniMax-H3-Image supports CPU offload only through --enable_model_cpu_offload, "
                "which streams the text encoder from host memory."
            )
        if config.enable_model_cpu_offload and (ulysses_degree > 1 or config.fully_shard_degree > 1):
            raise ValueError(
                "MiniMax-H3-Image --enable_model_cpu_offload is for single-GPU runs. Multi-GPU runs "
                "encode on the first rank; use --text_encoder_tp_degree if the encoder does not fit."
            )
        if config.batch_size is not None or config.dataset_path is not None:
            raise ValueError(
                "MiniMax-H3-Image packs one request into one sequence and does not "
                "support runner batching or dataset batching."
            )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        prompt = input_args["prompt"]
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("MiniMax-H3-Image supports one prompt per request.")
        if input_args.get("negative_prompt") is not None:
            raise ValueError("MiniMax-H3-Image is guidance-distilled and does not accept a negative prompt.")
        if input_args["num_inference_steps"] != MINIMAX_H3_IMAGE_STEPS:
            raise ValueError(
                f"MiniMax-H3-Image samples on a fixed {MINIMAX_H3_IMAGE_STEPS}-step grid, one output "
                f"head per step; got num_inference_steps={input_args['num_inference_steps']}."
            )

    _transformer_component_name = "transformer"

    def _load_model(self):
        from diffusers import ModularPipeline
        from diffusers.modular_pipelines.minimax_h3_image import encoders

        from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
            xFuserMiniMaxH3ImageTransformer2DWrapper,
        )

        log(f"Loading {self.settings.model_name} components")
        if self.config.text_encoder_tp_degree <= 1:
            _patch_minimax_h3_text_encoder_broadcast(encoders)
        pipe = ModularPipeline.from_pretrained(self.settings.model_name)
        # The repository also ships the original layout under transformer/; the
        # modular index names the diffusers one.
        subfolder = pipe.get_component_spec("transformer").subfolder
        transformer = xFuserMiniMaxH3ImageTransformer2DWrapper.from_pretrained(
            self.settings.model_name,
            dtype=torch.bfloat16,
            **self.loader.checkpoint_request(subfolder).from_pretrained_kwargs(),
            attention_backend=_parse_attention_backend(
                getattr(self.config, "attention_backend", None),
                "attention backend",
            ),
        )
        pipe.update_components(transformer=transformer)
        _load_minimax_h3_components(pipe, model_name="MiniMax-H3-Image")
        pipe.transformer.fuse_qkv_projections()
        # The fused to_qkv projections are new modules, created in training mode.
        pipe.transformer.eval()
        # The transformer conditions on hidden_states[text_encoder_layer]; the decoder
        # layers after it, the final norm and the LM head never reach it.
        # The config keeps its 64 layers: diffusers' get_qwen3vl_prompt_embeds checks the
        # config's layer count against text_encoder_layer, and the trimmed stack still
        # yields hidden_states[text_encoder_layer] unchanged.
        trim_text_encoder(pipe.text_encoder, pipe.text_encoder_layer)
        pipe.text_encoder.eval()
        self._parallelize_text_encoder(pipe.text_encoder)
        install_minimax_h3_image_tiled_decode(pipe.vae)
        return pipe

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self._expose_transformer_signature()

    def _expose_transformer_signature(self) -> None:
        """Show the denoise step the transformer's real forward parameters.

        The step passes the transformer only the layout fields its forward signature
        names. FSDP wraps the transformer in a module whose forward takes
        (*args, **kwargs), which would drop them all, as for FLUX.3.
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

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = input_args["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]
        pipe_kwargs = {
            "prompt": prompt,
            "height": input_args["height"],
            "width": input_args["width"],
            # The pipeline draws the noise from a CPU generator in the model card, so a
            # seed reproduces the documented diffusers output.
            "generator": torch.Generator().manual_seed(input_args["seed"]),
            "output": "images",
        }
        if input_args.get("input_images"):
            pipe_kwargs["image"] = input_args["input_images"]
        image = self.pipe(**pipe_kwargs)[0]
        return DiffusionOutput(images=[image], pipe_args=input_args)

    def _get_compile_mode(self) -> str:
        # CUDA graphs pay off once the forward is sharded, and cost nothing on one GPU.
        # Graphs re-capture pathologically on RDNA4, as for FLUX.2.
        if PACKAGES_CHECKER._on_rdna4():
            return "default"
        return "reduce-overhead"

    def _compile_model(self, input_args: dict) -> None:
        mode = self._get_compile_mode()
        # Without CUDA graphs: the decoder's blocks replay one graph back to back, so
        # each replay would overwrite the output the next block is still reading.
        self.pipe.vae.compile_repeated_blocks(mode="default", fullgraph=False)
        self._compile_text_encoder()
        if self.config.fully_shard_degree > 1:
            super()._compile_model(input_args)
            return
        self._enable_compute_comm_overlap()
        # Compile everything up to the output norm. The per-step output head stays
        # eager: its index is a Python int and would specialize a graph per step.
        # forward itself is left alone, so the denoise block still reads the
        # transformer's signature to pick its inputs.
        transformer = self.pipe.transformer
        transformer._forward_body = torch.compile(
            transformer._forward_body,
            # Host-issued SDMA copies cannot be replayed by CUDA graphs.
            mode=usp_sdma.compile_mode(mode),
            fullgraph=not get_runtime_state().has_attention_schedule(),
        )
        self._run_timed_pipe(copy.deepcopy(input_args))

    def _compile_text_encoder(self) -> None:
        """Compile the Qwen3-VL decoder stack that encodes the prompt.

        Encoding a prompt is launch-bound in eager mode. The prompt length varies
        per request, so the graph is dynamic in it. The tensor-parallel encoder
        all-reduces inside its layers, and offload moves it between devices, so
        both stay eager.
        """
        if self.config.text_encoder_tp_degree > 1 or self.config.enable_model_cpu_offload:
            return
        self.pipe.text_encoder.model.language_model.compile(dynamic=True)
