"""The quality gate has to fail the images that mean a load is broken, and pass the ones that do not.

Its floors are calibrated against measurements on Z-Image-Turbo at that model's own sampling, all
against the eager bf16 reference at one rank: sharding alone scores 0.9935 SSIM, quantizing the
weights and the text encoder scores 0.7244 on an image a person confirmed is the same picture, and a
collapsed render scores 0.0006. So these check the gate catches gross breakage without being
sensitive to the band a working case lives in, which is the only claim it can honestly make.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quality():
    return _load("image_quality", ROOT / "tools/image_quality.py")


@pytest.fixture
def write_png(tmp_path):
    def _write(name, array):
        from PIL import Image

        path = tmp_path / name
        Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(path)
        return path

    return _write


def _picture(seed=0, size=64):
    """Something with structure, since flat fields make every similarity metric agree."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    base = np.stack([x, y, (x + y) / 2], axis=2)
    return np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1)


def test_an_image_scores_perfectly_against_itself(quality, write_png):
    """The deterministic scoring path makes an exact reproduction normal, so it must not be a special
    case that produces infinities or crashes."""
    path = write_png("a.png", _picture())

    scores = quality.score_images(path, path)

    assert scores["comparable"] and scores["identical"]
    assert scores["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert scores["mse"] == 0.0
    assert scores["psnr"] == 99.0, "a perfect score has to stay a JSON number"
    assert quality.verdict(scores)["verdict"] == "pass"


def test_a_slightly_noisier_render_still_passes(quality, write_png):
    """Quantization moves an image a little, and that is the thing the gate must not fail on."""
    reference = _picture()
    reference_path = write_png("ref.png", reference)
    rng = np.random.default_rng(7)
    actual_path = write_png("act.png", np.clip(reference + rng.normal(0, 0.01, reference.shape), 0, 1))

    scores = quality.score_images(reference_path, actual_path)

    assert scores["ssim"] > quality.DEFAULT_THRESHOLDS["ssim_min"]
    assert quality.verdict(scores)["verdict"] == "pass"


@pytest.mark.parametrize(
    "name, breakage",
    [
        ("black frame", lambda ref: np.zeros_like(ref)),
        ("pure noise", lambda ref: np.random.default_rng(3).random(ref.shape)),
        ("wrong content", lambda ref: ref[::-1, ::-1]),
    ],
)
def test_a_broken_render_fails(quality, write_png, name, breakage):
    """These are the outcomes that mean a quantized load destroyed the model."""
    reference = _picture()
    reference_path = write_png("ref.png", reference)
    actual_path = write_png("act.png", breakage(reference))

    result = quality.verdict(quality.score_images(reference_path, actual_path))

    assert result["verdict"] == "fail", f"{name} must not pass the gate"
    assert result["failures"], "a failure has to say which threshold was missed"


def test_a_nan_render_fails_rather_than_scoring_nan(quality, write_png):
    """NaNs reaching the metric would compare false everywhere and could read as a pass."""
    reference = _picture()
    reference_path = write_png("ref.png", reference)
    # NaN does not survive a PNG round trip, so it arrives as whatever the cast produced. The point
    # is that a run which produced NaNs cannot come out the other side looking fine.
    actual_path = write_png("act.png", np.full_like(reference, np.nan))

    result = quality.verdict(quality.score_images(reference_path, actual_path))

    assert result["verdict"] == "fail"


def test_differently_sized_images_are_not_scored_into_agreement(quality, write_png):
    """Every case runs at one height and width, so a size difference means the run misbehaved."""
    reference_path = write_png("ref.png", _picture(size=64))
    actual_path = write_png("act.png", _picture(size=32))

    scores = quality.score_images(reference_path, actual_path)
    result = quality.verdict(scores)

    assert not scores["comparable"]
    assert result["verdict"] == "fail"
    assert "sizes differ" in result["failures"][0]


def test_the_thresholds_bracket_the_measured_reality(quality):
    """Pins the calibration, so nobody tightens a floor towards the real score and starts failing.

    Measured on Z-Image-Turbo against the eager bf16 reference at one rank: sharding alone scores
    0.9935 SSIM and 40.5 PSNR, quantization scores 0.7244 and 19.0, and a collapsed render scores
    0.0006. Each floor has to sit between the result it judges and breakage, with room on both sides,
    since tightening one buys no detection.
    """
    sharded = {"comparable": True, "ssim": 0.9935, "psnr": 40.48, "mse": 8.96e-5}
    quantized = {"comparable": True, "ssim": 0.7244, "psnr": 18.98, "mse": 1.264e-2}
    collapsed = {"comparable": True, "ssim": 0.0006, "psnr": 5.0, "mse": 0.32}

    assert quality.verdict(sharded)["verdict"] == "pass"
    assert quality.verdict(quantized, quality.QUANTIZED_THRESHOLDS)["verdict"] == "pass"
    for floors in (None, quality.QUANTIZED_THRESHOLDS):
        assert quality.verdict(collapsed, floors)["verdict"] == "fail"
    assert quality.DEFAULT_THRESHOLDS["ssim_min"] < 0.9935 - 0.05, "no margin against noise"
    assert quality.QUANTIZED_THRESHOLDS["ssim_min"] < 0.7244 - 0.05, "no margin against noise"
    assert quality.QUANTIZED_THRESHOLDS["ssim_min"] > 0.5, "would not catch a wrong image"


def test_an_unquantized_case_is_held_to_the_tighter_floor(quality):
    """Changing only where the weights live should not move the image, and 0.72 would say it may.

    This is the branch's own claim, so the sharded cases are the ones the gate has to be strict
    about: at the quantized floor a shard that rendered a different picture would still pass.
    """
    assert (
        quality.DEFAULT_THRESHOLDS["ssim_min"]
        > quality.QUANTIZED_THRESHOLDS["ssim_min"] + 0.2
    )
    quantized_result = {"comparable": True, "ssim": 0.7244, "psnr": 18.98, "mse": 1.264e-2}

    assert quality.verdict(quantized_result)["verdict"] == "fail"


def test_a_verdict_records_the_thresholds_it_used(quality, write_png):
    """A bare pass recorded months ago is not interpretable once the thresholds move."""
    path = write_png("a.png", _picture())

    result = quality.verdict(quality.score_images(path, path), {"ssim_min": 0.5})

    assert result["thresholds"]["ssim_min"] == 0.5
    assert result["thresholds"]["psnr_min"] == quality.DEFAULT_THRESHOLDS["psnr_min"]


def test_structure_is_compared_and_not_just_overall_brightness(quality, write_png):
    """A global comparison passes images whose statistics match but whose content does not.

    Shuffling the pixels preserves the mean and variance exactly, so a gate built on those alone
    would call this identical.
    """
    reference = _picture()
    rng = np.random.default_rng(11)
    shuffled = reference.reshape(-1, 3)[rng.permutation(reference[:, :, 0].size)].reshape(
        reference.shape
    )
    reference_path = write_png("ref.png", reference)
    actual_path = write_png("act.png", shuffled)

    assert quality.verdict(quality.score_images(reference_path, actual_path))["verdict"] == "fail"
