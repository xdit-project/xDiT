"""A Qwen-Image batch of prompts with different lengths matches one call per prompt, under Ulysses."""

import pytest

pytestmark = pytest.mark.multi_gpu

HEADS = 8
HEAD_DIM = 128


def _tiny_wrapper(device):
    import torch

    from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper

    torch.manual_seed(0)
    model = xFuserQwenImageTransformerWrapper(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        joint_attention_dim=32,
        axes_dims_rope=(16, 56, 56),
    )
    # At default init two different prompts move the output by ~1e-2, too little to tell a pad apart.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 2:
                parameter.mul_(4)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _call(model, hidden_states, prompt, img_shapes, attention_kwargs=None):
    import torch

    batch = hidden_states.shape[0]
    with torch.no_grad():
        return model(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt,
            timestep=torch.full((batch,), 0.5, device=hidden_states.device, dtype=hidden_states.dtype),
            img_shapes=img_shapes * batch,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]


def _worker(rank, world_size, init_method):
    from types import SimpleNamespace

    import torch

    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp
    from xfuser.model_executor.models.transformers import transformer_qwen
    from xfuser.model_executor.models.transformers.transformer_qwen import per_sample_key_packing

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size, ring_degree=1)
    backends = [
        b for b in (AttentionBackendType.AITER, AttentionBackendType.AITER_BF16) if registry.available(b) is None
    ]
    try:
        if not backends:
            pytest.skip("neither AITER FAv3 nor AITER MHA v4 BF16 is available")
        for backend in backends:
            state = SimpleNamespace(
                attention_backend=backend,
                fp8_comms=None,
                runtime_config=SimpleNamespace(use_sparseattn_head_balance=False),
            )
            usp.get_runtime_state = lambda: state
            transformer_qwen.get_runtime_state = lambda: state
            registry.prepare(backend)
            model = _tiny_wrapper(device)
            # (frames, rows, cols) of the target and the reference image: 2x(4x8) = 64 tokens, or 2x(4x9)
            # = 72 tokens, which needs SP pad at 8 and so takes the reordered layout.
            for cols in (8, 9):
                img_shapes = [[(1, 4, cols), (1, 4, cols)]]
                image_tokens = 2 * 4 * cols
                generator = torch.Generator().manual_seed(1)
                hidden_states = torch.randn(1, image_tokens, 16, generator=generator).to(device, torch.bfloat16)
                positive = torch.randn(1, 24, 32, generator=generator).to(device, torch.bfloat16)
                negative = torch.randn(1, 13, 32, generator=generator).to(device, torch.bfloat16)

                reference = [_call(model, hidden_states, p, img_shapes) for p in (positive, negative)]

                def error(a, b):
                    return (a.float() - b.float()).abs().max().item()

                # Errors are judged against how far the prompt alone moves the output.
                prompt_effect = error(reference[0], reference[1])
                assert prompt_effect > 1.0, f"the model barely depends on its prompt ({prompt_effect})"
                tolerance = 0.05 * prompt_effect

                padded = torch.cat([positive, torch.nn.functional.pad(negative, (0, 0, 0, 11))])
                packing = per_sample_key_packing([24, 13], 24, image_tokens, world_size, device)
                assert packing is not None
                batched = _call(model, hidden_states.repeat(2, 1, 1), padded, img_shapes, dict(packing))

                where = f"{backend.name}, {image_tokens} image tokens"
                for row, ref in enumerate(reference):
                    assert error(batched[row : row + 1], ref) < tolerance, (where, row, prompt_effect)
                # The packing, not the pad's content, keeps the row exact: a random pad changes nothing
                # with it and moves the unconditional row without it.
                noise = torch.randn(1, 11, 32, generator=generator).to(device, torch.bfloat16) * 4
                loud = torch.cat([positive, torch.cat([negative, noise], dim=1)])
                loud_packed = _call(model, hidden_states.repeat(2, 1, 1), loud, img_shapes, dict(packing))
                assert error(loud_packed[1:], reference[1]) < tolerance, (where, prompt_effect)
                unpacked = _call(model, hidden_states.repeat(2, 1, 1), loud, img_shapes)
                assert error(unpacked[1:], reference[1]) > 0.25 * prompt_effect, (where, prompt_effect)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_a_padded_prompt_batch_matches_one_call_per_prompt(accelerator_ranks, world_size):
    accelerator_ranks(_worker, world_size=world_size, timeout=300)
