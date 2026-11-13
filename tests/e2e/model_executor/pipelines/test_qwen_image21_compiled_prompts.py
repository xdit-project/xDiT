"""Compiled Qwen-Image-2.1 serves prompts of different lengths in one process.

Opt-in: ``XDIT_QWEN_IMAGE21_E2E=1 python -m pytest tests/e2e/model_executor/pipelines/test_qwen_image21_compiled_prompts.py``
with the Qwen/Qwen-Image-2.1 weights available and AITER installed. A second prompt length recompiles the
denoising blocks with dynamic shapes, which is where a compile failure would surface.
"""

import os

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.accelerator,
    pytest.mark.rocm,
    pytest.mark.skipif(
        os.environ.get("XDIT_QWEN_IMAGE21_E2E") != "1",
        reason="Set XDIT_QWEN_IMAGE21_E2E=1 and provide the Qwen/Qwen-Image-2.1 weights",
    ),
]


def test_compiled_runner_generates_for_two_prompt_lengths(tmp_path):
    import torch
    from PIL import Image

    from xfuser.runner import xFuserModelRunner

    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU with enough memory for Qwen-Image-2.1")
    condition = tmp_path / "condition.png"
    Image.new("RGB", (512, 512), (90, 140, 200)).save(condition)
    config = {
        "model": "Qwen/Qwen-Image-2.1",
        "use_torch_compile": True,
        "attention_backend": "aiter_bf16",
        "batch_size": 1,
        "output_directory": str(tmp_path),
        "warmup_calls": 0,
        "num_iterations": 1,
    }
    runner = xFuserModelRunner(config)
    args = vars(runner.config).copy()
    args.update(
        prompt=[
            "Change the background to a sunset beach with gentle waves and a warm orange sky, and keep the rest.",
            "Make the sky purple.",
        ],
        height=1024,
        width=1024,
        num_inference_steps=4,
        seed=42,
        input_images=[str(condition)],
    )
    args = runner.preprocess_args(args)
    try:
        runner.initialize(args)
        output, _ = runner.run(args)
    finally:
        runner.cleanup()

    assert len(output.images) == 2
