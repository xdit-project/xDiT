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


def _bhsd(generator, device, batch, seq, heads):
    """A [b, H, s, d] view of a [b, s, H, d] tensor, as models hand Q/K/V to USP."""
    import torch

    x = torch.randn(batch, seq, heads, HEAD_DIM, generator=generator, device=device, dtype=torch.bfloat16)
    return x.transpose(1, 2)


def exchange_mismatches(group, case, seed, rccl=False):
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
    from xfuser.model_executor.layers import usp_sdma

    previous = usp_sdma.configure(mode, head_groups)
    try:
        return fn(x, w)
    finally:
        usp_sdma.configure(*previous)


def test_compiled_sdma_matches_its_layout_over_rccl(accelerator_ranks):
    accelerator_ranks(_compiled_worker, world_size=4, timeout=900)


def _cuda_graph_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers.usp import USP

    try:
        device = _setup(rank, world_size, init_method, world_size)
        torch._dynamo.config.recompile_limit = 64

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
            actual = _usp_call(graphed, x, w, "packed", 2).clone()
            expected = _usp_call(layer, x, w, "packed", 2)
            torch.testing.assert_close(actual, expected, msg=lambda m: f"call {call}: {m}")
        dist.barrier()
    finally:
        # Graphs that captured RCCL kernels keep destroying their communicator from returning.
        torch.compiler.reset()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()


def test_packed_layout_runs_under_cuda_graphs(accelerator_ranks):
    accelerator_ranks(_cuda_graph_worker, world_size=4, timeout=900)


def _usp(q, k, v, mode, head_groups, **kwargs):
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.layers.usp import USP

    previous = usp_sdma.configure(mode, head_groups)
    try:
        return USP(q, k, v, **kwargs)
    finally:
        usp_sdma.configure(*previous)


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
    import torch
    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.layers.usp import USP

    class Processor:
        grouped_qkv = True

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
            previous = usp_sdma.configure("sdma", groups, overlap_qkv=True)
            try:
                assert usp_sdma.prepare_grouped_qkv(attn, world_size) == 1
                for compiled in (False, True):
                    fn = torch.compile(grouped) if compiled else grouped
                    for call in range(3):
                        x = torch.randn(1, 192, 256, generator=generator, device=device, dtype=torch.bfloat16)
                        actual = fn(attn, x)
                        assert actual is not None, "the grouped path declined the call"
                        usp_sdma.configure("default", 1)
                        expected = plain(attn, x)
                        usp_sdma.configure("sdma", groups, overlap_qkv=True)
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
                usp_sdma.configure(*previous)
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
