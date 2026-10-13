"""USP over gloo ignores a declared trailing sequence-parallel pad.

The pad is what a model appends so the sequence splits evenly across ranks. Its
keys must not be attended: in Wan they get a zero logit and act as an
attention sink. USP drops them given the real length (`valid_kv_len`); in ring
attention it does so inside the step that attends to the last rank's shard.
That needs the real Ulysses and ring exchanges, hence processes over gloo, and
torch.compile has to trace it without a graph break.
"""

import os
import queue
import shutil
import time
import traceback
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

HEADS, HEAD_DIM, PADDED_LEN = 4, 16, 48
CASES = {
    "no-pad": dict(real_len=PADDED_LEN),
    "declared-pad": dict(real_len=45, declared=True),
    "declared-pad-causal": dict(real_len=45, declared=True, causal=True),
}
# (Ulysses, ring) degrees. At ring degree 4 only one of a rank's four steps attends to the pad.
MODES = {"ulysses": (2, 1), "ring": (1, 2), "hybrid": (2, 2), "ring4": (1, 4), "hybrid-ring4": (2, 4)}
# Every variant computes in fp32 on the same CPU kernel; the leak of an attended pad is several orders larger.
TOLERANCE = 1e-5


def _inputs(torch, real_len):
    generator = torch.Generator().manual_seed(0)
    query, key, value = (torch.randn(1, HEADS, PADDED_LEN, HEAD_DIM, generator=generator) for _ in range(3))
    # As in Wan: the zero-padded RoPE tables zero the pad rows of Q and K, while V keeps a projection of them.
    query[:, :, real_len:] = 0
    key[:, :, real_len:] = 0
    value[:, :, real_len:] *= 3
    return query, key, value


def _wire_usp_to_gloo(torch, ulysses, ring):
    """Point usp at yunchang's gloo groups and a CPU kernel that returns the log-sum-exp ring attention merges on.

    xDiT hands yunchang's groups to usp only on CUDA or NPU machines. Each
    getter returns a precomputed value, as xDiT's own do: one that built or
    queried anything while being traced would be a graph break of the test's
    own making.
    """
    import torch.distributed as dist
    from yunchang import set_seq_parallel_pg
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.attention.constraints import ANY_CALL
    from xfuser.core.attention.requirements import ALWAYS
    from xfuser.core.attention.spec import AttentionBackendType, Spec
    from xfuser.model_executor.layers import usp

    set_seq_parallel_pg(ulysses, ring, dist.get_rank(), dist.get_world_size())

    def cpu_flash(query, key, value, call):
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, dropout_p=0.0, is_causal=call.is_causal
        )[:2]

    spec = Spec(AttentionBackendType.SDPA_FLASH, impl=cpu_flash, ring=ALWAYS, accepts=ANY_CALL, requires=ALWAYS)
    attention_function = usp.concat_joint_tensors_decorator(usp._spec_adapter(spec))
    ulysses_rank = dist.get_rank(PROCESS_GROUP.ULYSSES_PG)
    ring_rank = dist.get_rank(PROCESS_GROUP.RING_PG)
    runtime_state = SimpleNamespace(
        fp8_comms=None,
        attention_backend=AttentionBackendType.SDPA_FLASH,
        runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
    )
    usp.PROCESS_GROUP = PROCESS_GROUP
    usp.get_sequence_parallel_world_size = lambda: ulysses * ring
    usp.get_ulysses_parallel_world_size = lambda: ulysses
    usp.get_ring_parallel_world_size = lambda: ring
    usp.get_ulysses_parallel_rank = lambda: ulysses_rank
    usp.get_ring_parallel_rank = lambda: ring_rank
    usp.get_runtime_state = lambda: runtime_state
    usp._get_attention_function = lambda backend=None: attention_function
    return usp


def _real_rows_error(torch, rank, world_size, case, output):
    query, key, value = (tensor[:, :, : case["real_len"]] for tensor in _inputs(torch, case["real_len"]))
    expected = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        query, key, value, is_causal=case.get("causal", False)
    )[0]
    chunk = PADDED_LEN // world_size
    rows = range(rank * chunk, min((rank + 1) * chunk, case["real_len"]))
    if not rows:
        return 0.0
    local_rows = slice(rows.start - rank * chunk, rows.stop - rank * chunk)
    return (output[:, :, local_rows] - expected[:, :, rows.start : rows.stop]).abs().max().item()


def _local_inputs(torch, rank, world_size, real_len):
    chunk = PADDED_LEN // world_size
    return [tensor[:, :, rank * chunk : (rank + 1) * chunk].contiguous() for tensor in _inputs(torch, real_len)]


def _run_cases(torch, usp, rank, world_size, attention):
    errors = {}
    for name, case in CASES.items():
        local = _local_inputs(torch, rank, world_size, case["real_len"])
        attention_kwargs = {"valid_kv_len": case["real_len"]} if case.get("declared") else None
        output = attention(*local, is_causal=case.get("causal", False), attention_kwargs=attention_kwargs)
        errors[name] = _real_rows_error(torch, rank, world_size, case, output)
    return errors


def _packed_keys_error(torch, usp, rank, world_size):
    """What USP raises for packed keys with ring attention, which it cannot slice per step."""
    local = _local_inputs(torch, rank, world_size, 45)
    attention_kwargs = {
        "valid_kv_len": 45,
        "indices_k": torch.arange(45),
        "cu_seqlens_k": torch.tensor([0, 45], dtype=torch.int32),
        "max_seqlen_k": 45,
    }
    try:
        usp.USP(*local, attention_kwargs=attention_kwargs)
    except ValueError as error:
        return str(error)
    return None


def _worker(rank, world_size, init_method, result_queue, ulysses, ring, compile_backends):
    # The CPU path is the one under test, whatever accelerators the machine has.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    dist = None
    try:
        import torch
        import torch.distributed as dist

        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
        usp = _wire_usp_to_gloo(torch, ulysses, ring)
        report = {"errors": {"eager": _run_cases(torch, usp, rank, world_size, usp.USP)}, "graph_breaks": {}}
        for backend in compile_backends:
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            compiled = torch.compile(usp.USP, backend=backend)
            report["errors"][backend] = _run_cases(torch, usp, rank, world_size, compiled)
            report["graph_breaks"][backend] = sum(torch._dynamo.utils.counters["graph_break"].values())
        if ring > 1:
            report["packed_keys_error"] = _packed_keys_error(torch, usp, rank, world_size)
        result_queue.put(("returned", rank, report))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, world_size, args, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, init_method, result_queue, *args))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)
    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return [process.exitcode for process in processes], results


def _compile_backends():
    """aot_eager always: graph breaks are Dynamo's, whatever compiles the graph. Inductor too where it can build."""
    from torch._inductor import config

    return ["aot_eager"] + (["inductor"] if any(cxx and shutil.which(cxx) for cxx in config.cpp.cxx) else [])


@pytest.mark.parametrize("mode", MODES)
def test_usp_matches_attention_over_the_real_tokens(tmp_path, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="yunchang builds the Ulysses and ring process groups")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = MODES[mode]
    world_size = ulysses * ring
    compile_backends = _compile_backends()

    exit_codes, results = _run_spawned(
        torch, f"file://{tmp_path / 'usp-init'}", world_size, (ulysses, ring, compile_backends), timeout=900
    )

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert exit_codes == [0] * world_size
    assert sorted(result[1] for result in results) == list(range(world_size))
    for _, rank, report in results:
        for variant, case_errors in report["errors"].items():
            for case, error in case_errors.items():
                assert error <= TOLERANCE, f"rank {rank}, {variant}, {case}: max error {error:.3e}"
        assert report["graph_breaks"] == {backend: 0 for backend in compile_backends}, f"rank {rank}"
        if ring > 1:
            assert report["packed_keys_error"] is not None and "valid_kv_len must be in" in report["packed_keys_error"]
