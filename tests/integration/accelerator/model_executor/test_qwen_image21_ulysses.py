"""Qwen-Image-2.1 Ulysses parity against stock diffusers on a tiny random transformer."""

import pytest

pytest.importorskip(
    "diffusers.models.transformers.transformer_qwenimage21",
    reason="installed diffusers does not include Qwen-Image-2.1",
)

pytestmark = pytest.mark.multi_gpu

NUM_LAYERS = 2
NUM_HEADS = 8


def _tiny_model(device):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    torch.manual_seed(0)
    return (
        QwenImage21Transformer2DModel(
            num_layers=NUM_LAYERS,
            attention_head_dim=16,
            num_attention_heads=NUM_HEADS,
            context_in_dim=8,
            in_channels=4,
            out_channels=4,
            axes_dims_rope=(4, 6, 6),
        )
        .to(device)
        .eval()
    )


def _prefill_then_decode(model, device, encoder_hidden_states_mask):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    # 3 text, a 2x2 condition image (one VLM slot), 3 text, then a 2x6 target (three slots): a 22-token
    # prefill and a 12-token decode, so both are padded at some of the tested degrees.
    generator = torch.Generator().manual_seed(1)
    inputs = {
        "hidden_states": torch.randn(1, 4 + 12, 4, generator=generator).to(device),
        "encoder_hidden_states": torch.randn(1, 7, 8, generator=generator).to(device),
        "img_shapes": [[(1, 2, 2), (1, 2, 6)]],
        "img_mask": torch.tensor([[False] * 3 + [True] + [False] * 3 + [True] * 3], device=device),
    }
    kv_cache = QwenImage21KVCache(NUM_LAYERS)
    outputs = []
    for timestep, mode in ((0.9, "extract"), (0.4, "cached")):
        with torch.no_grad():
            outputs.append(
                model(
                    **inputs,
                    timestep=torch.tensor([timestep], device=device),
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    kv_cache=kv_cache,
                    kv_cache_mode=mode,
                ).sample
            )
    return outputs


def _ulysses_worker(rank, world_size, init_method):
    from types import SimpleNamespace

    import torch
    from diffusers.models.transformers import transformer_qwenimage21 as qwenimage21

    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp
    from xfuser.model_executor.models.transformers import transformer_qwenimage21 as xfuser_qwenimage21

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ulysses_degree=world_size, ring_degree=1)
    usp.get_runtime_state = lambda: SimpleNamespace(attention_backend=AttentionBackendType.SDPA)
    try:
        processors = [xfuser_qwenimage21.xFuserQwenImage21AttnProcessor]
        if qwenimage21._FLEX_AVAILABLE:
            processors.append(xfuser_qwenimage21.xFuserQwenImage21FlexAttnProcessor)
        padded_prompt = torch.ones(1, 7, dtype=torch.bool, device=device)
        padded_prompt[0, -1] = False

        for processor_cls in processors:
            for mask in (None, padded_prompt):
                stock = _tiny_model(device)
                reference = _prefill_then_decode(stock, device, mask)

                wrapper = xfuser_qwenimage21.xFuserQwenImage21TransformerWrapper.from_config(stock.config)
                wrapper = wrapper.to(device).eval()
                wrapper.load_state_dict(stock.state_dict())
                for block in wrapper.transformer_blocks:
                    block.attn.set_processor(processor_cls())
                actual = _prefill_then_decode(wrapper, device, mask)

                for ref, out in zip(reference, actual):
                    torch.testing.assert_close(
                        out,
                        ref,
                        rtol=1e-4,
                        atol=1e-4,
                        msg=lambda m: f"{processor_cls.__name__}, padded prompt={mask is not None}: {m}",
                    )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_matches_single_gpu_diffusers(accelerator_ranks, world_size):
    accelerator_ranks(_ulysses_worker, world_size=world_size, timeout=300)
