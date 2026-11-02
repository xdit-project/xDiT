"""Faster ``output_type="pil"`` post-processing for diffusers image pipelines."""

import functools
from typing import Iterable, List

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image, fromarray


def images_to_pil(images: torch.Tensor) -> List[Image]:
    """PIL images from an ``output_type="pt"`` batch, ``[B, C, H, W]`` in ``[0, 1]``.

    Gives the same pixels as diffusers' ``output_type="pil"``, but quantizes to ``uint8`` and lays the
    pixels out row by row where the tensor is, so less crosses to the host and PIL needs no strided copy.
    """
    pixels = images.float().mul(255).round_().clamp_(0, 255).to(torch.uint8)
    pixels = pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [fromarray(p.squeeze(-1) if p.shape[-1] == 1 else p) for p in pixels]


def _with_fast_pil(postprocess):
    @functools.wraps(postprocess)
    def wrapped(image, output_type="pil", *args, **kwargs):
        if output_type != "pil":
            return postprocess(image, output_type, *args, **kwargs)
        return images_to_pil(postprocess(image, "pt", *args, **kwargs))

    return wrapped


def enable_fast_pil_postprocess(pipes: Iterable) -> None:
    """Route ``output_type="pil"`` through ``images_to_pil`` for each pipeline's ``VaeImageProcessor``.

    Covers diffusers pipelines and xDiT's wrappers, whose diffusers pipeline is ``module``. Other output
    types still go to diffusers.
    """
    processors = {}
    for pipe in pipes:
        for owner in (pipe, getattr(pipe, "module", None)):
            processor = getattr(owner, "image_processor", None)
            if isinstance(processor, VaeImageProcessor):
                processors[id(processor)] = processor
    for processor in processors.values():
        processor.postprocess = _with_fast_pil(processor.postprocess)
