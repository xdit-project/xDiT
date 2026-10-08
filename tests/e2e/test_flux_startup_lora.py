"""Opt-in generation with official Flux weights and a community LoRA."""

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        os.environ.get("XDIT_FLUX_LORA_E2E") != "1",
        reason="Set XDIT_FLUX_LORA_E2E=1 and provide access to FLUX.1-dev weights",
    ),
]


def test_flux_startup_lora_generates_image(tmp_path):
    pytest.importorskip("peft")
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from xfuser.runner import xFuserModelRunner

    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU with enough memory for FLUX.1-dev")
    model_name = "black-forest-labs/FLUX.1-dev"
    adapter_name = "alvdansen/frosting_lane_flux"
    weight_name = "flux_dev_frostinglane_araminta_k.safetensors"
    config = {
        "model": model_name,
        "lora_path": adapter_name,
        "lora_weight_name": weight_name,
        "lora_scale": 0.75,
        "output_directory": str(tmp_path),
        "warmup_calls": 0,
        "num_iterations": 1,
    }
    runner = xFuserModelRunner(config)
    args = vars(runner.config).copy()
    args.update(
        prompt="A small robot, frstingln illustration",
        height=256,
        width=256,
        num_inference_steps=2,
        max_sequence_length=64,
        seed=0,
        input_images=[],
    )
    args = runner.preprocess_args(args)
    try:
        runner.initialize(args)
        key = "transformer_blocks.0.attn.to_q.weight"
        index = hf_hub_download(model_name, "diffusion_pytorch_model.safetensors.index.json", subfolder="transformer")
        shard = json.loads(Path(index).read_text())["weight_map"][key]
        base_path = hf_hub_download(model_name, shard, subfolder="transformer")
        adapter_path = hf_hub_download(adapter_name, weight_name)
        with safe_open(base_path, framework="pt") as base:
            original = base.get_tensor(key).to(torch.bfloat16).float()
        with safe_open(adapter_path, framework="pt") as adapter:
            a = adapter.get_tensor("transformer.transformer_blocks.0.attn.to_q.lora_A.weight").float()
            b = adapter.get_tensor("transformer.transformer_blocks.0.attn.to_q.lora_B.weight").float()
        projection = runner.model.pipe.transformer.transformer_blocks[0].attn.to_q
        observed = projection.weight.detach().cpu().float() - original
        expected = 0.75 * (b @ a)
        assert torch.nn.functional.cosine_similarity(observed.flatten(), expected.flatten(), dim=0) > 0.9
        assert not any("lora_" in name for name, _ in runner.model.pipe.transformer.named_parameters())
        output, _ = runner.run(args)
        assert len(output.images) == 1
        pixels = np.asarray(output.images[0])
        assert pixels.shape == (256, 256, 3)
        assert pixels.std() > 0
        output.images[0].save(tmp_path / "flux_lora.png")
    finally:
        runner.cleanup()
