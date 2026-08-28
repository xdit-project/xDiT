#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Self-contained xDiT + AMD Quark **native inference** example for FLUX.1.

Unlike plain "fake-quantized" inference (where quantized layers still run the
high-precision GEMM with simulated quant/dequant on the fly), *native inference*
freezes the model and converts every ``QuantLinear`` into a real low-precision
kernel path:

    nn.Linear  ->  QuantLinear  ->  QParamsLinear  ->  Aiter native linear

The conversion is driven by ``ModelQuantizer.freeze(..., runtime_options=...)``
with ``RuntimeOptions(native_linear_mode=...)``:

    * ``fp8_per_tensor``  for FP8 E4M3   (mode ``fp8``)
    * ``mxfp4``           for OCP MXFP4  (mode ``mxfp4``)

The resulting modules hold real quantized weights and dispatch to the Aiter GEMM
kernels, so the transformer runs faster and uses less memory than the
fake-quantized path. Because the frozen modules are ``QParamsLinear`` subclasses,
the model can also be exported directly to HF safetensors (``--quark_export_dir``).

This script keeps xDiT unmodified: everything Quark-related lives here.

Prerequisites
-------------
    pip install -e .            # xDiT
    pip install amd-quark       # AMD Quark
    # Aiter kernels are required for the native linear path on AMD GPUs (ROCm).

Usage
-----
    # Single GPU, FP8 native inference
    python examples/flux_quark_example.py \
        --model /path/to/FLUX.1-dev \
        --prompt "a cat holding a sign that says hello world" \
        --height 1024 --width 1024 --num_inference_steps 28 \
        --quark_quantization_mode fp8

    # Single GPU, MXFP4 native inference
    python examples/flux_quark_example.py \
        --model /path/to/FLUX.1-dev \
        --prompt "a cat" --quark_quantization_mode mxfp4

    # Multi-GPU (2 GPUs) with Ulysses sequence parallel
    torchrun --nproc_per_node 2 examples/flux_quark_example.py \
        --model /path/to/FLUX.1-dev --prompt "a cat" \
        --ulysses_degree 2 --quark_quantization_mode fp8

    # Also export the frozen native model to HF safetensors
    python examples/flux_quark_example.py \
        --model /path/to/FLUX.1-dev --prompt "a cat" \
        --quark_quantization_mode fp8 --quark_export_dir ./flux_fp8_native
"""

import logging
import time

import torch

from xfuser import xFuserArgs, xFuserFluxPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    get_world_group,
)

# AMD Quark imports. Kept at module scope so a missing install fails fast with a
# clear message rather than deep inside the pipeline.
try:
    from quark.torch import export_safetensors
    from quark.torch.quantization.api import ModelQuantizer
    from quark.torch.quantization.config.config import (
        FP8E4M3PerTensorSpec,
        OCP_MXFP4Spec,
        QConfig,
        QLayerConfig,
    )
    from quark.torch.quantization.utils import RuntimeOptions
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "AMD Quark is required for this example. Install it with: pip install amd-quark"
    ) from exc


def quantize_transformer_native(
    transformer: torch.nn.Module,
    quant_mode: str,
    device: torch.device,
    export_dir: str | None = None,
) -> torch.nn.Module:
    """
    Quantize a diffusion transformer with AMD Quark and freeze it for native inference.

    The flow is:

        1. Build a per-tensor FP8 (or MXFP4) spec for both weights and activations.
        2. ``quantize_model`` inserts fake-quant ``QuantLinear`` modules (dynamic
           activation quant -> no calibration data required).
        3. ``freeze(..., runtime_options=RuntimeOptions(native_linear_mode=...))``
           converts ``QuantLinear -> QParamsLinear -> Aiter native linear`` so the
           model runs the real low-precision GEMM at inference time.
        4. Optionally export the frozen model to HF safetensors.

    :param torch.nn.Module transformer: The diffusion transformer to quantize.
    :param str quant_mode: ``"fp8"`` (FP8 E4M3) or ``"mxfp4"`` (OCP MXFP4).
    :param torch.device device: Target device for the quantized model.
    :param str | None export_dir: If set, export the frozen model to this directory.

    :return: The frozen, native-inference-ready transformer.
    :rtype: torch.nn.Module
    """
    mode = quant_mode.lower()
    if mode == "fp8":
        quant_spec = FP8E4M3PerTensorSpec(
            observer_method="min_max",
            scale_type="float",
            is_dynamic=True,
        ).to_quantization_spec()
        native_linear_mode = "fp8_per_tensor"
        logging.info("Quark: using FP8 E4M3 per-tensor (dynamic) quantization")
    elif mode == "mxfp4":
        quant_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        native_linear_mode = "mxfp4"
        logging.info("Quark: using OCP MXFP4 quantization")
    else:
        raise ValueError(f"Unsupported quark_quantization_mode: {quant_mode!r}. Choose 'fp8' or 'mxfp4'.")

    layer_config = QLayerConfig(weight=quant_spec, input_tensors=quant_spec)
    quantizer = ModelQuantizer(QConfig(global_quant_config=layer_config))

    logging.info("Quark: inserting quantized layers (dynamic quantization, no calibration)...")
    quantized = quantizer.quantize_model(transformer)
    quantized = quantized.to(device)

    logging.info(f"Quark: freezing for native inference (native_linear_mode='{native_linear_mode}')...")
    runtime_options = RuntimeOptions(native_linear_mode=native_linear_mode)
    quantized = quantizer.freeze(quantized, runtime_options=runtime_options)

    if export_dir is not None:
        logging.info(f"Quark: exporting frozen native model to safetensors at {export_dir} ...")
        with torch.no_grad():
            export_safetensors(
                model=quantized,
                output_dir=export_dir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="reorder",
            )

    logging.info("Quark: native inference model ready")
    return quantized


def main():
    parser = FlexibleArgumentParser(description="xFuser FLUX + Quark native inference")
    parser = xFuserArgs.add_cli_args(parser)

    # Quark-specific options (self-contained; not part of xDiT's argument set).
    parser.add_argument(
        "--quark_quantization_mode",
        type=str,
        default="fp8",
        choices=["fp8", "mxfp4"],
        help="Native inference quantization format: 'fp8' (FP8 E4M3) or 'mxfp4' (OCP MXFP4). Default: fp8.",
    )
    parser.add_argument(
        "--quark_export_dir",
        type=str,
        default=None,
        help="Optional directory to export the frozen native model as HF safetensors.",
    )

    args = parser.parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    # Build the FLUX pipeline (unmodified xDiT path).
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    # Apply Quark native-inference quantization to the transformer in place,
    # after the pipeline is on device but before prepare_run / inference.
    device = torch.device(f"cuda:{local_rank}")
    pipe.transformer = quantize_transformer_native(
        transformer=pipe.transformer,
        quant_mode=args.quark_quantization_mode,
        device=device,
        export_dir=args.quark_export_dir,
    )

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        import os

        os.makedirs("./results", exist_ok=True)
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = (
                    f"flux_quark_{args.quark_quantization_mode}_native_"
                    f"result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                )
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, "
            f"parameter memory: {parameter_peak_memory / 1e9:.2f} GB, "
            f"memory: {peak_memory / 1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
