"""Ulysses attention with its all-to-alls on SDMA engines: the exchange, and USP on it."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

# The backend the SDMA path runs with in production. SDPA, reading the bshd inbox views
# directly, has given rare run-to-run differences under heavy neighbour traffic even though
# the views it read held the right data before and after.
BACKEND = "AITER"
HEAD_DIM = 64


def _setup(rank, world_size, init_method, ulysses, cfg=1):
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
    initialize_model_parallel(classifier_free_guidance_degree=cfg, ulysses_degree=ulysses)
    engine_config, _ = xFuserArgs(
        attention_backend=BACKEND, ulysses_degree=ulysses, use_cfg_parallel=cfg > 1
    ).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend(BACKEND)
    return torch.device("cuda", rank)


def _set_config(config):
    """Make `config` the runtime's Ulysses all-to-all options; returns the previous ones."""
    from xfuser.core.distributed import get_runtime_state

    runtime_config = get_runtime_state().runtime_config
    previous, runtime_config.ulysses_a2a = runtime_config.ulysses_a2a, config
    return previous


def _configure(mode, head_groups, **options):
    from xfuser.config.ulysses_a2a import UlyssesA2AConfig

    return _set_config(UlyssesA2AConfig(mode, head_groups, **options))


def _bhsd(generator, device, batch, seq, heads):
    """A [b, H, s, d] view of a [b, s, H, d] tensor, as models hand Q/K/V to USP."""
    import torch

    x = torch.randn(batch, seq, heads, HEAD_DIM, generator=generator, device=device, dtype=torch.bfloat16)
    return x.transpose(1, 2)


def exchange_mismatches(group, case, seed, rccl=False, early=False):
    """Run one exchange and count wrong elements, over every view attention sees and the output.

    Each rank all-gathers the inputs to know what each head group's views must hold, joint rows
    included; the stand-in for attention returns 2 * Q + 1, exact in any layout, so the output
    of this rank must be 2 * Q + 1 for its own rows.
    """
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.sdma import ulysses

    batch, s_q, s_kv, hq, hkv, extras, joint_rows, front, groups, concurrent = case
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + dist.get_rank())
    q = _bhsd(generator, device, batch, s_q, hq * world)
    k, v = (_bhsd(generator, device, batch, s_kv, hkv * world) for _ in range(2))
    tensors = [q, k, v, *(_bhsd(generator, device, batch, s_q, hq * world) for _ in range(extras))]
    joint = [_bhsd(generator, device, batch, joint_rows, hkv).contiguous() for _ in range(2)] if joint_rows else None
    full = []
    for x in tensors:
        parts = [torch.empty_like(x.contiguous()) for _ in range(world)]
        dist.all_gather(parts, x.contiguous(), group=group)
        full.append(torch.cat(parts, dim=2))
    heads = [x.shape[1] // world for x in tensors]
    sizes = ulysses.head_split(heads, groups)
    counts = [torch.zeros((), dtype=torch.int64, device=device) for _ in sizes]
    calls = []

    def attend(*views):
        gi = len(calls)
        calls.append(gi)
        for t, view in enumerate(views):
            off, h = sum(sizes[i][t] for i in range(gi)), sizes[gi][t]
            want = full[t][:, rank * heads[t] + off : rank * heads[t] + off + h]
            if joint and t in (1, 2):
                own = joint[t - 1][:, off : off + h]
                want = torch.cat([own, want] if front else [want, own], dim=2)
            counts[gi].add_((view != want).sum())
        return views[0] * 2 + 1

    kv = (joint_rows, 0) if front else (0, joint_rows)
    reserved = [(0, 0), kv, kv] + [(0, 0)] * extras
    local = [None, *joint] + [None] * extras if joint else None
    if early:
        # Each head group in two sends: Q and K first, then the rest.
        rows = [x.shape[2] for x in tensors]
        exchange = ulysses.Exchange.make(world, batch, HEAD_DIM, q.dtype, rows, heads, reserved, groups)
        ulysses.begin(group, exchange)
        offsets = [0] * len(tensors)
        for gi, group_sizes in enumerate(sizes):
            pieces = [ulysses.pieces_from_heads(x, world, offsets[t], group_sizes[t]) for t, x in enumerate(tensors)]
            offsets = [o + h for o, h in zip(offsets, group_sizes)]
            ulysses.send(group, gi, ulysses.pack_group(pieces[:2]), (0, 1))
            ulysses.send(group, gi, ulysses.pack_group(pieces[2:]))
        out = ulysses.finish(group, attend, local, concurrent)
    else:
        out = ulysses.attention(tensors, attend, group, groups, concurrent, reserved, local, rccl)
    wrong = (out != q.transpose(1, 2) * 2 + 1).sum() + sum(counts)
    return int(wrong.item())


# (batch, Q rows, K/V rows, Q heads per rank, K/V heads per rank, extra tensors, joint rows,
#  joint in front, head groups, group streams)
EXCHANGES = [
    (1, 96, 96, 5, 5, 0, 0, True, 1, False),
    (1, 96, 96, 5, 5, 0, 0, True, 2, False),
    (1, 96, 96, 5, 5, 0, 0, True, 5, False),
    (1, 96, 96, 5, 5, 0, 0, True, 2, True),
    (2, 64, 64, 6, 6, 0, 0, True, 3, False),
    # Joint text in front and behind: Q carries the joint rows, K/V get them locally.
    (1, 72, 64, 6, 6, 0, 8, True, 2, False),
    (1, 72, 64, 6, 6, 0, 8, False, 3, True),
    # Grouped-query K/V (4 query heads per K/V head) and an extra Q-shaped tensor.
    (1, 64, 64, 8, 2, 1, 0, True, 2, False),
    # Larger than every call before it: the shared buffers grow.
    (1, 1024, 1024, 5, 5, 0, 0, True, 2, False),
]


def _exchange_worker(rank, world_size, init_method, cfg, rccl):
    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        _setup(rank, world_size, init_method, world_size // cfg, cfg)
        for index, case in enumerate(EXCHANGES):
            # Several calls in a row, as layers make them, each with new data.
            for call in range(3):
                wrong = exchange_mismatches(PROCESS_GROUP.ULYSSES_PG, case, 10 * index + call, rccl)
                assert wrong == 0, f"{case}: {wrong} wrong elements"
        # Then every call a different shape, head-group count and stream setting than the one
        # before, as a model's different attention layers make them: the layout, copy plans and
        # flag words change between consecutive calls.
        for sweep in range(2):
            for index, case in enumerate(EXCHANGES):
                wrong = exchange_mismatches(PROCESS_GROUP.ULYSSES_PG, case, 1000 + 100 * sweep + index, rccl)
                assert wrong == 0, f"interleaved {case}: {wrong} wrong elements"
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_exchange(accelerator_ranks, world_size):
    accelerator_ranks(_exchange_worker, world_size=world_size, timeout=600, args=(1, False))


def _wan_early_worker(rank, world_size, init_method):
    from yunchang.globals import PROCESS_GROUP
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        _setup(rank, world_size, init_method, world_size)
        for call in range(12):
            case = (1, 96 if call % 3 else 1024, 96 if call % 3 else 1024, 5, 5, 0, 0, True, 1 + call % 2, False)
            for early in (False, True):
                wrong = exchange_mismatches(PROCESS_GROUP.ULYSSES_PG, case, call, early=early)
                assert wrong == 0, f"early={early}, call={call}: {wrong} wrong elements"
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_wan_early_qk_exchange(accelerator_ranks):
    accelerator_ranks(_wan_early_worker, world_size=2, timeout=300)


def _wan_early_compiled_worker(rank, world_size, init_method):
    import torch
    from yunchang.globals import PROCESS_GROUP
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers.usp import USP

    try:
        device = _setup(rank, world_size, init_method, world_size)
        group = PROCESS_GROUP.ULYSSES_PG

        @torch.compile(fullgraph=True)
        def layer(hidden, weight, early):
            query = torch.nn.functional.linear(hidden, weight).unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)
            key = query * 2
            if early:
                spec = ulysses.encode(world_size, 1, HEAD_DIM, [hidden.shape[1]] * 3, [5] * 3, [(0, 0)] * 3, 1)
                token = torch.ops.xfuser.sdma_ulysses_begin(query, spec, group.group_name, "sdma")
                packed_qk = ulysses.pack_group(
                    [ulysses.pieces_from_heads(tensor, world_size, 0, 5) for tensor in (query, key)]
                )
                token = torch.ops.xfuser.sdma_ulysses_send(token, packed_qk, 0, [0, 1], group.group_name)
                value = torch.ops.xfuser.sdma_ulysses_linear(token, hidden, weight, None) * 3
                value = value.unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)
                packed_value = ulysses.pack_group([ulysses.pieces_from_heads(value, world_size, 0, 5)])
                token = torch.ops.xfuser.sdma_ulysses_send(token, packed_value, 0, [2], group.group_name)
                return usp_sdma.finish(
                    token,
                    [packed_qk, packed_value],
                    spec,
                    joint_key=None,
                    joint_value=None,
                    extra_names=(),
                    backend=BACKEND,
                    dropout_p=0.0,
                    is_causal=False,
                    attention_kwargs={},
                    kv_head_repeat=1,
                    group=group,
                )
            value = (torch.nn.functional.linear(hidden, weight) * 3).unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)
            return USP(query, key, value)

        generator = torch.Generator(device=device).manual_seed(rank)
        weight = (
            torch.randn(5 * world_size * HEAD_DIM, 128, generator=generator, device=device, dtype=torch.bfloat16) * 0.05
        )
        for call in range(8):
            hidden = torch.randn(1, 256, 128, generator=generator, device=device, dtype=torch.bfloat16)
            expected = _usp_call(lambda inputs, weights: layer(inputs, weights, False), hidden, weight, "packed", 1)
            actual = layer(hidden, weight, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda message: f"call {call}: {message}")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_wan_early_compiled_matches_packed(accelerator_ranks):
    accelerator_ranks(_wan_early_compiled_worker, world_size=2, timeout=600)


def _wan_packed_qk_worker(rank, world_size, init_method):
    import torch
    from yunchang.globals import PROCESS_GROUP
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers.fused_qk_norm_rope_wan_flydsl import fused_qk_norm_rope, fused_qk_norm_rope_packed

    try:
        device = _setup(rank, world_size, init_method, world_size)
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


def test_wan_packed_qk_matches_original_norm_and_pack(accelerator_ranks):
    accelerator_ranks(_wan_packed_qk_worker, world_size=2, timeout=600)


def _wan_processor_worker(rank, world_size, init_method, backend):
    from types import SimpleNamespace

    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention
    from xfuser.core.distributed import get_runtime_state
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.models.transformers.transformer_wan import (
        xFuserWanAttnProcessor,
        xFuserWanSdmaAttnProcessor,
    )

    graph = None
    original_send = ulysses.send
    calls = 0

    def counted_send(process_group, gi, packed, tensors=None):
        # Q/K sent on their own: the split path ran.
        nonlocal calls
        calls += tensors is not None and tuple(tensors) == (0, 1)
        return original_send(process_group, gi, packed, tensors)

    try:
        device = _setup(rank, world_size, init_method, world_size)
        get_runtime_state().set_attention_backend(backend)
        torch.manual_seed(0)
        attn = WanAttention(256, heads=40, dim_head=128, processor=xFuserWanAttnProcessor()).to(device, torch.bfloat16)
        attn.eval()
        # As the runner prepares the transformer: the SDMA processor replaces the standard one.
        _configure("sdma", 1, overlap_qkv=True)
        usp_sdma.prepare(SimpleNamespace(attn=attn), ["attn"], cpu_offload=False)
        assert type(attn.processor) is xFuserWanSdmaAttnProcessor
        generator = torch.Generator(device=device).manual_seed(rank)
        hidden = torch.randn(1, 257, 256, generator=generator, device=device, dtype=torch.bfloat16)
        angle = torch.randn(1, 257, 1, 128, generator=generator, device=device)
        rotary = angle.cos(), angle.sin()
        ulysses.send = counted_send
        compiled = torch.compile(attn, fullgraph=True)
        with torch.no_grad():
            for run in (attn, compiled):
                _configure("packed", 1)
                expected = run(hidden, rotary_emb=rotary)
                _configure("sdma", 1, overlap_qkv=True)
                before = calls
                actual = run(hidden, rotary_emb=rotary)
                assert calls == before + 1, "production split-QK path did not execute"
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _configure("sdma", 1, overlap_qkv=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attn(hidden, rotary_emb=rotary)
            for repeat in range(3):
                hidden.copy_(torch.randn(hidden.shape, generator=generator, device=device, dtype=hidden.dtype))
                graph.replay()
                _configure("packed", 1)
                expected = attn(hidden, rotary_emb=rotary)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            # The head-group count does not apply: Wan self-attention splits Q/K in one group.
            _configure("sdma", 2, overlap_qkv=True)
            before = calls
            actual = attn(hidden, rotary_emb=rotary)
            assert calls == before + 1, "the split-QK path must run with --ulysses_sdma_head_groups 2"
            _configure("packed", 1)
            torch.testing.assert_close(actual, attn(hidden, rotary_emb=rotary), rtol=0, atol=0)
            # Unsupported: FP32 QK norms keep the usual path.
            attn.norm_q.to(torch.float32)
            attn.norm_k.to(torch.float32)
            _configure("sdma", 1, overlap_qkv=True)
            before = calls
            actual = attn(hidden, rotary_emb=rotary)
            assert calls == before, "unsupported settings must not split Q/K"
            _configure("sdma", 1)
            torch.testing.assert_close(actual, attn(hidden, rotary_emb=rotary), rtol=0, atol=0)
    finally:
        graph = None
        ulysses.send = original_send
        _configure("default", 2)
        torch.compiler.reset()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("backend", ["AITER_BF16", "AITER_FP8"])
def test_wan_processor_split_qk_matches_packed(accelerator_ranks, backend):
    accelerator_ranks(_wan_processor_worker, world_size=2, timeout=900, args=(backend,))


@pytest.mark.parametrize("world_size", [4, 8])
def test_sdma_exchange_under_cfg_parallel(accelerator_ranks, world_size):
    accelerator_ranks(_exchange_worker, world_size=world_size, timeout=600, args=(2, False))


def test_packed_exchange_over_rccl(accelerator_ranks):
    accelerator_ranks(_exchange_worker, world_size=4, timeout=600, args=(1, True))


def _lifetime_worker(rank, world_size, init_method):
    import gc
    import weakref

    import torch
    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sdma import ulysses

    try:
        device = _setup(rank, world_size, init_method, world_size)
        group = PROCESS_GROUP.ULYSSES_PG
        generator = torch.Generator(device=device).manual_seed(rank)
        q, k, v = (_bhsd(generator, device, 1, 256, 4 * world_size) for _ in range(3))
        ex = ulysses.Exchange.make(world_size, 1, HEAD_DIM, q.dtype, [256] * 3, [4] * 3, [(0, 0)] * 3, 1)
        ulysses.begin(group, ex)
        packed = ulysses.pack_group([ulysses.pieces_from_heads(x, world_size, 0, 4) for x in (q, k, v)])
        alive = weakref.ref(packed)
        ulysses.send(group, 0, packed)
        # The caller drops its tensor, as compiled code may right after the send: the engines may
        # still be reading it, so the call must keep it until finish.
        del packed
        gc.collect()
        assert alive() is not None
        out = ulysses.finish(group, lambda q, k, v: q * 2 + 1)
        gc.collect()
        assert alive() is None
        torch.testing.assert_close(out, q.transpose(1, 2) * 2 + 1, rtol=0, atol=0)
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_packed_tensors_live_until_finish(accelerator_ranks):
    accelerator_ranks(_lifetime_worker, world_size=2, timeout=300)


def _compiled_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        device = _setup(rank, world_size, init_method, world_size)
        torch._dynamo.config.recompile_limit = 64

        @torch.compile
        def layer(x, w):
            # A projection and the exchange in one graph, as a compiled block runs them; the pack
            # fuses into the kernel producing Q/K/V.
            q, k, v = (torch.nn.functional.linear(x, w) * (i + 1) for i in range(3))
            q, k, v = (t.unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2) for t in (q, k, v))
            from xfuser.model_executor.layers.usp import USP

            return USP(q, k, v)

        generator = torch.Generator(device=device).manual_seed(rank)
        heads = 6 * world_size
        w = torch.randn(heads * HEAD_DIM, 256, generator=generator, device=device, dtype=torch.bfloat16) * 0.05
        # Head groups change from call to call, which changes how buffers are laid out and reused.
        for call, groups in enumerate((1, 2, 1, 3, 2, 1, 2, 1)):
            x = torch.randn(1, 384, 256, generator=generator, device=device, dtype=torch.bfloat16)
            expected = _usp_call(layer, x, w, "packed", groups)
            actual = _usp_call(layer, x, w, "sdma", groups)
            # The same layout either way, so the backend rounds both the same.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda m: f"call {call}: {m}")
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _usp_call(fn, x, w, mode, head_groups):

    previous = _configure(mode, head_groups)
    try:
        return fn(x, w)
    finally:
        _set_config(previous)


def test_compiled_sdma_matches_its_layout_over_rccl(accelerator_ranks):
    accelerator_ranks(_compiled_worker, world_size=4, timeout=900)


def _cuda_graph_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sdma import transport
    from xfuser.model_executor.layers.usp import USP

    original_begin = transport.PeerGroup.begin
    calls = []

    def counted_begin(group):
        calls.append(1)
        return original_begin(group)

    try:
        device = _setup(rank, world_size, init_method, world_size)
        torch._dynamo.config.recompile_limit = 64
        transport.PeerGroup.begin = counted_begin

        def layer(x, w):
            q, k, v = (torch.nn.functional.linear(x, w) * (i + 1) for i in range(3))
            q, k, v = (t.unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2) for t in (q, k, v))
            return USP(q, k, v)

        # CUDA graph trees run their warmup in the graph's memory pool: the ops must not leave an
        # allocation of theirs alive there (this process has not used the SDMA transport yet).
        graphed = torch.compile(layer, mode="reduce-overhead")
        generator = torch.Generator(device=device).manual_seed(rank)
        w = torch.randn(6 * world_size * HEAD_DIM, 256, generator=generator, device=device, dtype=torch.bfloat16)
        w = w * 0.05
        for call in range(4):
            x = torch.randn(1, 384, 256, generator=generator, device=device, dtype=torch.bfloat16)
            packed = _usp_call(graphed, x, w, "packed", 2).clone()
            expected = _usp_call(layer, x, w, "packed", 2)
            torch.testing.assert_close(packed, expected, msg=lambda m: f"call {call}: {m}")
            # The SDMA ops run between the graph's partitions, on the transport, not captured.
            before = len(calls)
            actual = _usp_call(graphed, x, w, "sdma", 2).clone()
            assert len(calls) == before + 1, f"call {call}: the SDMA transport did not run"
            torch.testing.assert_close(actual, packed, rtol=0, atol=0, msg=lambda m: f"call {call}, sdma: {m}")
        dist.barrier()
    finally:
        transport.PeerGroup.begin = original_begin
        # Graphs that captured RCCL kernels keep destroying their communicator from returning.
        torch.compiler.reset()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()


def test_compiled_with_cuda_graphs_the_ops_run_between_graphs(accelerator_ranks):
    accelerator_ranks(_cuda_graph_worker, world_size=4, timeout=900)


def _usp(q, k, v, mode, head_groups, **kwargs):
    from xfuser.model_executor.layers.usp import USP

    previous = _configure(mode, head_groups)
    try:
        return USP(q, k, v, **kwargs)
    finally:
        _set_config(previous)


def _usp_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        device = _setup(rank, world_size, init_method, world_size)
        generator = torch.Generator(device=device).manual_seed(rank)
        heads = 6 * world_size
        q, k, v = (_bhsd(generator, device, 1, 256, heads) for _ in range(3))
        jq, jk, jv = (_bhsd(generator, device, 1, 24, heads) for _ in range(3))
        calls = {
            "plain": {},
            "trailing pad": {"attention_kwargs": {"valid_kv_len": world_size * 256 - 7}},
            "joint front": dict(joint_query=jq, joint_key=jk, joint_value=jv, joint_strategy="front"),
            "joint rear": dict(joint_query=jq, joint_key=jk, joint_value=jv, joint_strategy="rear"),
            "joint front, trailing pad": dict(
                joint_query=jq,
                joint_key=jk,
                joint_value=jv,
                joint_strategy="front",
                attention_kwargs={"valid_kv_len": world_size * 256 - 7},
            ),
        }
        for name, kwargs in calls.items():
            expected = _usp(q, k, v, "default", 1, **kwargs)
            packed = _usp(q, k, v, "packed", 2, **kwargs)
            for mode in ("sdma", "packed"):
                actual = _usp(q, k, v, mode, 2, **kwargs)
                # The same layout either way, so SDMA must match it exactly.
                torch.testing.assert_close(
                    actual, packed, rtol=0, atol=0, msg=lambda m: f"{name}, {mode} vs packed: {m}"
                )
                # The backend reads bshd views on this path and contiguous bhsd on RCCL's, and may
                # round differently for the two.
                torch.testing.assert_close(actual, expected, msg=lambda m: f"{name}, {mode}: {m}")
        # Grouped-query K/V: 2 query heads per K/V head.
        kq, kk = _bhsd(generator, device, 1, 128, 4 * world_size), _bhsd(generator, device, 1, 128, 2 * world_size)
        kv = _bhsd(generator, device, 1, 128, 2 * world_size)
        expected = _usp(kq, kk, kv, "default", 1, kv_head_repeat=2, combine_qkv_a2a=True)
        actual = _usp(kq, kk, kv, "sdma", 2, kv_head_repeat=2)
        torch.testing.assert_close(actual, expected, msg=lambda m: f"grouped-query: {m}")
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_usp_on_sdma_matches_rccl(accelerator_ranks):
    accelerator_ranks(_usp_worker, world_size=4, timeout=600)


def _grouped_qkv_worker(rank, world_size, init_method):
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.layers.usp import USP

    class Processor:
        sdma_grouped_qkv = True

    class Attention(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads, self.processor = heads, Processor()
            self.to_q, self.to_k, self.to_v = (torch.nn.Linear(dim, heads * HEAD_DIM) for _ in range(3))

    def norm_rope(q, k, v):
        # Per head, as the models' QK norm and RoPE are: [b, s, h, d] with any subset of heads.
        q, k = (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) for x in (q, k))
        return q, k.flip(-1), v

    def plain(attn, x):
        q, k, v = (m(x).unflatten(-1, (-1, HEAD_DIM)) for m in (attn.to_q, attn.to_k, attn.to_v))
        q, k, v = norm_rope(q, k, v)
        return USP(*(t.transpose(1, 2) for t in (q, k, v)))

    def grouped(attn, x):
        return usp_sdma.grouped_qkv_attention(
            attn, x, norm_rope, backend=None, attention_kwargs={}, group=PROCESS_GROUP.ULYSSES_PG
        )

    try:
        device = _setup(rank, world_size, init_method, world_size)
        torch._dynamo.config.recompile_limit = 64
        torch.manual_seed(0)  # the same weights on every rank
        attn = Attention(256, 6 * world_size).to(device, torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(rank)
        for groups in (2, 3):
            previous = _configure("sdma", groups, overlap_qkv=True)
            try:
                usp_sdma.prepare(SimpleNamespace(attn=attn), ["attn"], cpu_offload=False)
                assert attn._xfuser_sdma_qkv_layout[0] == world_size
                for compiled in (False, True):
                    fn = torch.compile(grouped) if compiled else grouped
                    for call in range(3):
                        x = torch.randn(1, 192, 256, generator=generator, device=device, dtype=torch.bfloat16)
                        actual = fn(attn, x)
                        assert actual is not None, "the grouped path declined the call"
                        _configure("default", 1)
                        expected = plain(attn, x)
                        _configure("sdma", groups, overlap_qkv=True)
                        # Compiled, the norm stays in fp32 where the eager reference rounds to
                        # bf16 in between: differences of a bf16 step.
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=1e-2 if compiled else 1e-5,
                            rtol=1.6e-2,
                            msg=lambda m: f"{groups} groups, compiled {compiled}, call {call}: {m}",
                        )
            finally:
                _set_config(previous)
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_grouped_qkv_projection_matches_the_plain_path(accelerator_ranks):
    accelerator_ranks(_grouped_qkv_worker, world_size=4, timeout=900)


def _lost_flag_worker(rank, world_size, init_method):
    import os

    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    os.environ["XDIT_SDMA_WAIT_S"] = "2"
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sdma import transport

    # Before the transport's first use: abort mode would start a watchdog that ends the process.
    transport.set_on_timeout("raise")

    case = (1, 64, 64, 5, 5, 0, 0, True, 1, False)
    try:
        _setup(rank, world_size, init_method, world_size)
        exchange_mismatches(PROCESS_GROUP.ULYSSES_PG, case, 0)
        dist.barrier()
        # In rank 0's next call, its output push (the call's second issue) to its first peer,
        # rank 1, loses its flag: rank 1's wait for the outputs gives up, and no one waits on it.
        if rank == 0:
            transport.debug_drop_flag(1, 1, 0)
        exchange_mismatches(PROCESS_GROUP.ULYSSES_PG, case, 1)
        try:
            transport.raise_if_waits_gave_up()
            message = ""
        except RuntimeError as error:
            message = str(error)
        every = [None] * world_size
        dist.all_gather_object(every, message)
        assert every[0] == "" and "wait 1, flag words [0] never set" in every[1], every
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_lost_flag_is_reported_by_the_rank_that_waited(accelerator_ranks):
    accelerator_ranks(_lost_flag_worker, world_size=2, timeout=300)
