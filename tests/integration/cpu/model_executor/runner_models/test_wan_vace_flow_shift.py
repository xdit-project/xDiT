"""Exercise VACE's public runner lifecycle with the real Diffusers denoising loop.

Only checkpoint loading, neural-network computation and process/device boundaries
are replaced. The runner hooks, pipeline, image processing and UniPC steps are real.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def vace_runner(monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("ftfy")
    if not hasattr(diffusers, "WanVACEPipeline"):
        pytest.skip("WanVACEPipeline requires diffusers >= 0.35.2")

    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    from xfuser import xFuserArgs
    from xfuser.core.distributed import parallel_state
    from xfuser.model_executor.models.runner_models import base_model, wan
    from xfuser.model_executor.models.transformers.transformer_wan_vace import (
        xFuserWanVACETransformer3DWrapper,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    group = SimpleNamespace(world_size=1, local_rank=0, rank=0, rank_in_group=0, barrier=lambda: None)
    monkeypatch.setattr(parallel_state, "_WORLD", group)
    monkeypatch.setattr(base_model, "get_model_replica_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(base_model, "initialize_runtime_state", lambda *args: None)
    monkeypatch.setattr(base_model, "runtime_state_is_initialized", lambda: False)
    monkeypatch.setattr(wan, "get_runtime_state", lambda: Mock())
    # Bypass GPU timing only; all tensors and scheduler computation stay on CPU.
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Mock(elapsed_time=lambda end: 1.0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    transformer = diffusers.WanVACETransformer3DModel(
        num_attention_heads=1,
        attention_head_dim=8,
        in_channels=1,
        out_channels=1,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
        vace_layers=[0],
        vace_in_channels=66,
    )
    observed_timesteps = []

    def predict(hidden_states, timestep, **kwargs):
        observed_timesteps.append(timestep.detach().clone())
        return (hidden_states * 0.1,)

    monkeypatch.setattr(transformer, "forward", predict)
    vae = diffusers.AutoencoderKLWan(
        base_dim=4,
        z_dim=1,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        latents_mean=[0.0],
        latents_std=[1.0],
    )

    def encode(video):
        latent = video[:, :1, ::4, ::8, ::8]
        return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(torch.cat([latent, latent], dim=1)))

    def decode(latent, **kwargs):
        video = torch.nn.functional.interpolate(
            latent.repeat(1, 3, 1, 1, 1),
            size=((latent.shape[2] - 1) * 4 + 1, latent.shape[3] * 8, latent.shape[4] * 8),
            mode="nearest",
        )
        return (video.tanh(),)

    monkeypatch.setattr(vae, "encode", encode)
    monkeypatch.setattr(vae, "decode", decode)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    text_encoder = transformers.UMT5EncoderModel(
        transformers.UMT5Config(vocab_size=2, d_model=8, d_ff=16, d_kv=8, num_heads=1, num_layers=1)
    ).eval()
    scheduler = diffusers.UniPCMultistepScheduler(
        flow_shift=3.0,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        final_sigmas_type="zero",
    )
    pipeline = diffusers.WanVACEPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    monkeypatch.setattr(xFuserWanVACETransformer3DWrapper, "from_pretrained", lambda *args, **kwargs: transformer)
    monkeypatch.setattr(diffusers.WanVACEPipeline, "from_pretrained", lambda *args, **kwargs: pipeline)
    # Retain pipeline placement, changing its accelerator destination to CPU.
    pipeline_to = pipeline.to
    monkeypatch.setattr(pipeline, "to", lambda *args, **kwargs: pipeline_to("cpu"))
    runner = wan.xFuserWan21VACEModel(xFuserArgs(model="Wan2.1-VACE-1.3B", warmup_calls=0, num_iterations=1))
    return runner, observed_timesteps, scheduler


@pytest.mark.parametrize(
    ("height", "width", "requested", "expected"),
    [
        (None, None, None, 5.0),
        (480, 832, None, 3.0),
        (480, 832, 4.0, 4.0),
        (720, 1280, 3.0, 3.0),
    ],
)
def test_vace_inference_uses_requested_or_resolution_flow_shift(vace_runner, height, width, requested, expected):
    runner, observed_timesteps, checkpoint_scheduler = vace_runner
    image = Image.new("RGB", (16, 16), (128, 128, 128))
    input_args = runner.preprocess_args(
        dict(
            height=height,
            width=width,
            flow_shift=requested,
            prompt="a moving scene",
            dataset_path=None,
            input_images=[image, image],
            num_frames=5,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
        )
    )
    runner.initialize(input_args)
    output, _ = runner.run(input_args)

    reference = type(checkpoint_scheduler).from_config(checkpoint_scheduler.config, flow_shift=expected)
    reference.set_timesteps(input_args["num_inference_steps"], device="cpu")
    # These are the timesteps consumed by the denoiser in the real pipeline loop,
    # rather than an assertion on the scheduler's stored configuration.
    torch.testing.assert_close(torch.cat(observed_timesteps), reference.timesteps)
    assert output.videos[0].shape == (5, input_args["height"], input_args["width"], 3)
    assert np.isfinite(output.videos[0]).all()
