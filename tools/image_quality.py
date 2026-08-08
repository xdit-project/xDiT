"""Score a validation image against a reference one.

Exists to answer one question about a quantized or sharded load: did it still draw the picture. It
is deliberately not a fine-grained quality metric, because at the configurations this matrix runs it
cannot be one. Measured on Z-Image-Turbo at 4 steps and 512x512, fp8 scores 0.9825 SSIM against bf16
at the same rank count, while the same fp8 case scores 0.9809 against itself when torch.compile
picks different kernels. Quantization damage and kernel choice are the same size, so a threshold
placed between them would report which kernel won. Scoring runs therefore disable compile, which
makes both sides deterministic.

What survives that is a gross-correctness gate. NaNs, black frames, wrong content and quantization
that destroys the model all land below 0.5, nowhere near where a working case sits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Calibrated against what the gate actually compares: a candidate at eight ranks against an eager
# bf16 reference at one, so a passing case differs by parallelism and quantization together rather
# than by quantization alone. fp8 measures 0.9227 SSIM and 24.9 PSNR that way, and gross breakage
# measures below 0.5, so these floors clear the real result by a wide margin and still fail breakage.
# Tightening them towards the measured value would buy no detection and start failing on noise.
DEFAULT_THRESHOLDS = {"ssim_min": 0.80, "psnr_min": 15.0}


def load_image(path: str | Path) -> np.ndarray:
    """An HxWx3 array in [0, 1]. Alpha is dropped, since the artifacts are opaque renders."""
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0


# A rendered picture of anything has spread. Every artifact this matrix has produced and a human has
# confirmed measures at least 0.15 standard deviation over [0, 1], and a collapsed render measures
# exactly 0, so this floor sits an order of magnitude below the real results and still catches a flat
# frame. Deliberately not a mean floor: a legitimately dark render has a low mean, while no
# legitimate render is uniform.
BLANK_LIMITS = {"std_min": 0.01}

READABLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def describe_content(path: str | Path) -> dict[str, Any]:
    """What is in one artifact, judged without a reference to compare it against.

    A reference comparison can only speak about models whose sample is stable enough to compare, so
    it cannot be asked whether an image exists at all. This can, for every model: a run that wrote a
    black frame and exited zero is a failure no matter what the model would have drawn.
    """
    suffix = Path(path).suffix.casefold()
    if suffix not in READABLE_SUFFIXES:
        return {
            "measured": False,
            "reason": f"no still-image reader for {suffix or 'this artifact'}",
        }
    image = load_image(path)
    return {
        "measured": True,
        "mean": round(float(image.mean()), 6),
        "std": round(float(image.std()), 6),
        "levels": int(np.unique(np.round(image * 255.0)).size),
    }


def blank_verdict(
    content: dict[str, Any], limits: dict[str, float] | None = None
) -> dict[str, Any]:
    """Whether the artifact is flat, or a statement that it could not be measured."""
    floors = {**BLANK_LIMITS, **(limits or {})}
    if not content.get("measured"):
        return {
            "verdict": "unmeasured",
            "failures": [],
            "thresholds": floors,
        }
    failures = []
    if content["std"] < floors["std_min"]:
        failures.append(
            f"uniform image: standard deviation {content['std']:.4f} below "
            f"{floors['std_min']}, {content['levels']} distinct levels"
        )
    return {
        "verdict": "fail" if failures else "pass",
        "failures": failures,
        "thresholds": floors,
    }


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    offsets = np.arange(size, dtype=np.float64) - (size - 1) / 2
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _blur(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable gaussian blur, valid region only.

    Separable because an 11x11 window over a 512x512 plane is otherwise the slowest part of scoring
    a run, and cropping to the valid region keeps the edges from being compared against padding
    that neither image actually contains.
    """
    rows = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, plane)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, rows)


def ssim(reference: np.ndarray, actual: np.ndarray) -> float:
    """Mean SSIM over channels, windowed rather than global.

    A single global comparison passes images whose overall statistics match but whose content does
    not, which is exactly the failure this is meant to catch.
    """
    kernel = _gaussian_kernel()
    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for channel in range(reference.shape[2]):
        x, y = reference[:, :, channel], actual[:, :, channel]
        mu_x, mu_y = _blur(x, kernel), _blur(y, kernel)
        sigma_x = _blur(x * x, kernel) - mu_x**2
        sigma_y = _blur(y * y, kernel) - mu_y**2
        sigma_xy = _blur(x * y, kernel) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        scores.append(float((numerator / denominator).mean()))
    return float(np.mean(scores))


def score_images(reference_path: str | Path, actual_path: str | Path) -> dict[str, Any]:
    """Compare two artifacts, or explain why they are not comparable.

    Mismatched dimensions are reported rather than resized into agreement: the matrix runs every
    case at one height and width, so a size difference means the run did not do what was asked and
    a similarity score would paper over it.
    """
    reference = load_image(reference_path)
    actual = load_image(actual_path)
    if reference.shape != actual.shape:
        return {
            "comparable": False,
            "reason": (
                f"image sizes differ: reference {reference.shape[1]}x{reference.shape[0]} "
                f"against {actual.shape[1]}x{actual.shape[0]}"
            ),
        }
    mse = float(((reference - actual) ** 2).mean())
    return {
        "comparable": True,
        "mse": mse,
        # Capped rather than infinite so the value stays JSON-representable when a run reproduces
        # its reference exactly, which the deterministic scoring path makes a normal outcome.
        "psnr": 99.0 if mse < 1e-12 else round(-10.0 * math.log10(mse), 3),
        "ssim": round(ssim(reference, actual), 6),
        "identical": mse == 0.0,
    }


def verdict(scores: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Turn scores into pass or fail, naming every threshold that was not met.

    Reports the thresholds alongside the outcome because a bare pass or fail recorded months ago is
    not interpretable once the numbers move.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not scores.get("comparable", False):
        return {
            "verdict": "fail",
            "failures": [scores.get("reason", "images are not comparable")],
            "thresholds": limits,
        }
    failures = []
    if scores["ssim"] < limits["ssim_min"]:
        failures.append(f"ssim {scores['ssim']:.4f} below {limits['ssim_min']}")
    if scores["psnr"] < limits["psnr_min"]:
        failures.append(f"psnr {scores['psnr']:.1f} below {limits['psnr_min']}")
    return {
        "verdict": "fail" if failures else "pass",
        "failures": failures,
        "thresholds": limits,
    }
