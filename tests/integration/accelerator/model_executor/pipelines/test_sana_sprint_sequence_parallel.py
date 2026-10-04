"""Sana Sprint under sequence parallelism must denoise like diffusers on one device.

Two things kept it from doing so. Without flash-attn, Sana's linear-attention
processor built its Ulysses layer with yunchang's AttnType.TORCH, which released
yunchang no longer defines, so the model failed to load. And xDiT's cross-attention
processor dropped the prompt's attention mask, so image tokens also attended to the
prompt's padding tokens. Two ranks with Ulysses attention run a tiny Sana Sprint
with a padded prompt for two steps and compare with diffusers.
"""

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import init_distributed_environment
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel


def _tiny_sana_sprint():
    from diffusers import SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler

    torch.manual_seed(0)
    transformer = SanaTransformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        sample_size=32,
        qk_norm="rms_norm_across_heads",
        guidance_embeds=True,
    )
    return SanaSprintPipeline(
        tokenizer=None, text_encoder=None, vae=None, transformer=transformer, scheduler=SCMScheduler()
    )


def _call_kwargs(device):
    generator = torch.Generator().manual_seed(1)
    return {
        "prompt_embeds": torch.randn(1, 6, 8, generator=generator).to(device),
        # A padded prompt, as the tokenizer produces: the last two tokens are padding.
        "prompt_attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]], device=device),
        "latents": torch.randn(1, 4, 16, 16, generator=generator).to(device),
        "height": 512,
        "width": 512,
        "num_inference_steps": 2,
        "output_type": "latent",
        "use_resolution_binning": False,
        "generator": torch.Generator().manual_seed(3),
    }


def _worker(rank, world_size, init_method):
    from xfuser.model_executor.pipelines.pipeline_sana_sprint import xFuserSanaSprintPipeline

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    args = xFuserArgs(model="tiny", ulysses_degree=world_size, attention_backend="sdpa")
    engine_config, _ = args.create_config()
    engine_config.runtime_config.dtype = torch.float32

    pipeline = _tiny_sana_sprint().to(device)
    with torch.no_grad():
        expected = pipeline(**_call_kwargs(device))[0]
    output = xFuserSanaSprintPipeline(pipeline, engine_config)(**_call_kwargs(device))

    # Only the last rank of the data-parallel group returns the images.
    if rank == world_size - 1:
        torch.testing.assert_close(output[0], expected, rtol=1e-3, atol=1e-3)
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.multi_gpu
def test_sana_sprint_sequence_parallel_matches_diffusers(accelerator_ranks):
    accelerator_ranks(_worker, world_size=2, timeout=240, init_filename="sana-sprint-sp")
