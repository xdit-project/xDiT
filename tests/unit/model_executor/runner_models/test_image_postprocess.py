from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from xfuser.model_executor.models.runner_models.image_postprocess import enable_fast_pil_postprocess


def _decoded(channels):
    """A decoded VAE batch in [-1, 1], slightly out of range so clamping is covered."""
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(2, channels, 9, 13, generator=generator) * 2.2 - 1.1).bfloat16()


@pytest.mark.parametrize("processor_cls", [VaeImageProcessor, Flux2ImageProcessor])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_pil_output_matches_diffusers(processor_cls, channels):
    pipe = SimpleNamespace(image_processor=processor_cls())
    enable_fast_pil_postprocess([pipe])

    actual = pipe.image_processor.postprocess(_decoded(channels), output_type="pil")
    expected = processor_cls().postprocess(_decoded(channels), output_type="pil")

    assert [image.mode for image in actual] == [image.mode for image in expected]
    for ours, theirs in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


@pytest.mark.parametrize("output_type", ["np", "pt", "latent"])
def test_other_output_types_are_unchanged(output_type):
    pipe = SimpleNamespace(image_processor=VaeImageProcessor())
    enable_fast_pil_postprocess([pipe])

    actual = pipe.image_processor.postprocess(_decoded(3), output_type=output_type)
    expected = VaeImageProcessor().postprocess(_decoded(3), output_type=output_type)

    torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected), rtol=0, atol=0)


def test_wrapped_pipelines_and_shared_processors_are_patched_once():
    """xDiT's wrappers keep the diffusers pipeline in ``module``, often sharing its processor."""
    calls = []
    processor = VaeImageProcessor()
    original = processor.postprocess

    def counting_postprocess(*args, **kwargs):
        calls.append(kwargs.get("output_type", args[1] if len(args) > 1 else "pil"))
        return original(*args, **kwargs)

    processor.postprocess = counting_postprocess
    module = SimpleNamespace(image_processor=processor)
    enable_fast_pil_postprocess([SimpleNamespace(image_processor=processor, module=module), None])

    images = module.image_processor.postprocess(_decoded(3), output_type="pil")

    assert calls == ["pt"]
    assert len(images) == 2
