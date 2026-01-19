import os
import csv
import logging
import torch
import numpy as np
from PIL.Image import Image
from typing import Callable, Optional
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_, _is_linear
from torchao.quantization.quantize_.common import KernelPreference

logger = logging.getLogger(__name__)

def log(message: str, debug=False) -> None:
    """Log message only from the last process to avoid duplicates."""
    if is_first_process():
        if debug:
            logger.debug(message)
        else:
            logger.info(message)

def is_first_process() -> bool:
    """ Checks based on env rank and world size if this is last process in """
    rank = int(os.environ.get("RANK"))
    return rank == 0

def resize_image_to_max_area(image: Image, input_height: int, input_width: int, mod_value: int) -> Image:
    """ Resize image to fit within max area while retaining aspect ratio """

    max_area = input_height * input_width
    width, height = image.size
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area /aspect_ratio)) // mod_value * mod_value

    image = image.resize((width, height))
    log(f"Resized image to {image.width}x{image.height} to fit within max area {width}x{height}")
    return image

def resize_and_crop_image(image: Image, target_height: int, target_width: int, mod_value: int) -> Image:
        """ Resize and center-crop image to target dimensions """

        target_height_aligned = target_height // mod_value * mod_value
        target_width_aligned = target_width // mod_value * mod_value

        log("Force output size mode enabled.")
        log(f"Input image resolution: {image.height}x{image.width}")
        log(f"Requested output resolution: {target_height}x{target_width}")
        log(f"Aligned output resolution (multiple of {mod_value}): {target_height_aligned}x{target_width_aligned}")

        # Step 1: Resize image maintaining aspect ratio so both dimensions >= target
        img_width, img_height = image.size

        # Calculate scale factor to ensure both dimensions are at least target size
        scale_width = target_width_aligned / img_width
        scale_height = target_height_aligned / img_height
        scale = max(scale_width, scale_height)  # Use max to ensure both dims are >= target

        # Resize with aspect ratio preserved
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        image = image.resize((new_width, new_height))

        log(f"Resized image to: {new_height}x{new_width} (maintaining aspect ratio)")

        # Step 2: Crop from center to get exact target dimensions
        left = (new_width - target_width_aligned) // 2
        top = (new_height - target_height_aligned) // 2
        image = image.crop((left, top, left + target_width_aligned, top + target_height_aligned))

        log(f"Cropped from center to: {target_height_aligned}x{target_width_aligned}")
        return image

def quantize_linear_layers_to_fp8(module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    filter_fn: Callable[[torch.nn.Module, str], bool] = _is_linear,
    device: Optional[torch.device] = None) -> None:
    config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                set_inductor_config=False,
                kernel_preference=KernelPreference.AUTO
        )
    """Quantize all linear layers in the given module or module list to FP8."""
    log("Quantizing linear layers to FP8...")
    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        quantize_(module_or_module_list_to_quantize, config=config, filter_fn=filter_fn, device=device)
    elif isinstance(module_or_module_list_to_quantize, torch.nn.ModuleList):
        for module in module_or_module_list_to_quantize:
            quantize_(module, config=config, filter_fn=filter_fn, device=device)
    else:
        raise ValueError(f"Failed to quantize linear layers. Invalid module type: {type(module_or_module_list_to_quantize)}, expected torch.nn.Module or torch.nn.ModuleList")


def load_dataset_prompts(dataset_path: str) -> list[str]:
    """ load prompts from a csv dataset file """
    prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row['prompt'])
    log(f"Loaded {len(prompts)} prompts from dataset at {dataset_path}")
    return prompts