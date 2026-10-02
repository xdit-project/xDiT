import pytest
from PIL import Image

pytest.importorskip(
    "diffusers.pipelines.qwenimage21.pipeline_qwenimage21",
    reason="installed diffusers does not include Qwen-Image-2.1",
)

from xfuser.model_executor.models.runner_models.qwen import _qwen_image21_output_size


def test_explicit_size_wins_over_images():
    assert _qwen_image21_output_size([Image.new("RGB", (1600, 900))], 512, 768) == (512, 768)


def test_text_only_defaults_to_square():
    assert _qwen_image21_output_size([], None, None) == (1024, 1024)


def test_landscape_image_sets_landscape_output():
    assert _qwen_image21_output_size([Image.new("RGB", (1600, 900))], None, None) == (768, 1376)


def test_last_image_decides_the_aspect_ratio():
    images = [Image.new("RGB", (1600, 900)), Image.new("RGB", (900, 1600))]
    assert _qwen_image21_output_size(images, None, None) == (1376, 768)


def test_only_one_dimension_is_rejected():
    with pytest.raises(ValueError, match="both --height and --width"):
        _qwen_image21_output_size([], 1024, None)
