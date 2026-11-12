"""Wan's FlyDSL QK norm and RoPE, writing Q/K packed for the SDMA Ulysses exchange."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _setup(rank, world_size, init_method):
    """One Ulysses group of every rank, with the runtime state the SDMA path reads."""
    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size)
    engine_config, _ = xFuserArgs(attention_backend="AITER", ulysses_degree=world_size).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend("AITER")
    return torch.device("cuda", rank)


def _packed_worker(rank, world_size, init_method):
    import torch
    from yunchang.globals import PROCESS_GROUP
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers.fused_qk_norm_rope_wan_flydsl import fused_qk_norm_rope, fused_qk_norm_rope_packed

    try:
        device = _setup(rank, world_size, init_method)
        generator = torch.Generator(device=device).manual_seed(rank)
        norm = torch.nn.RMSNorm(5120, eps=1e-6, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(5120, generator=generator, device=device, dtype=torch.bfloat16))
            for rows in (257, 1024, 257):
                query, key = [
                    torch.randn(1, rows, 5120, generator=generator, device=device, dtype=torch.bfloat16)
                    for _ in range(2)
                ]
                angle = torch.randn(1, rows, 1, 128, generator=generator, device=device)
                rotary = angle.cos(), angle.sin()
                expected_q, expected_k = fused_qk_norm_rope(query, key, norm, norm, *rotary, 40)
                for peers in (2, 4, 8):
                    expected = ulysses.pack_group(
                        [
                            ulysses.pieces_from_heads(tensor.transpose(1, 2), peers, 0, 40 // peers)
                            for tensor in (expected_q, expected_k)
                        ]
                    )
                    actual = fused_qk_norm_rope_packed(query, key, norm, norm, *rotary, 40, peers)
                    torch.testing.assert_close(actual.reshape(-1), expected, rtol=0, atol=0)
                group = PROCESS_GROUP.ULYSSES_PG
                exchange = ulysses.Exchange.make(
                    world_size, 1, 128, query.dtype, [rows] * 3, [40 // world_size] * 3, [(0, 0)] * 3, 1
                )
                ulysses.begin(group, exchange)
                packed_qk = fused_qk_norm_rope_packed(query, key, norm, norm, *rotary, 40, world_size)
                ulysses.send(group, 0, packed_qk, (0, 1))
                packed_value = ulysses.pack_group(
                    [ulysses.pieces_from_heads(expected_k.transpose(1, 2), world_size, 0, 40 // world_size)]
                )
                ulysses.send(group, 0, packed_value, (2,))
                output = ulysses.finish(group, lambda query, key, value: query * 2 + 1)
                torch.testing.assert_close(output, expected_q * 2 + 1, rtol=0, atol=0)
            norm.float()
            expected_q, expected_k = fused_qk_norm_rope(query, key, norm, norm, *rotary, 40)
            expected = ulysses.pack_group(
                [
                    ulysses.pieces_from_heads(tensor.transpose(1, 2), world_size, 0, 40 // world_size)
                    for tensor in (expected_q, expected_k)
                ]
            )
            actual = fused_qk_norm_rope_packed(query, key, norm, norm, *rotary, 40, world_size)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_packed_qk_matches_the_norm_rope_and_pack(accelerator_ranks):
    """Q/K normed, rotated and packed peer by peer in one kernel: as the unpacked kernel then
    the exchange's packing, and through a Q/K-first exchange."""
    accelerator_ranks(_packed_worker, world_size=2, timeout=600)
